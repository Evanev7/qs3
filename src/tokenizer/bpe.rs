use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap};

use super::TokenizerError;
use super::pretokenize::{byte_char, char_byte};

pub(super) struct BpeDefinition {
    vocab: HashMap<String, u32>,
    merges: Vec<(String, String)>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct BpeFingerprint {
    pub(super) vocab: u64,
    pub(super) merges: u64,
}

impl BpeDefinition {
    pub(super) fn new(vocab: HashMap<String, u32>, merges: Vec<(String, String)>) -> Self {
        Self { vocab, merges }
    }

    pub(super) fn vocab(&self) -> &HashMap<String, u32> {
        &self.vocab
    }

    pub(super) fn token_count(&self) -> usize {
        self.vocab.len()
    }

    pub(super) fn merge_count(&self) -> usize {
        self.merges.len()
    }

    pub(super) fn fingerprint(&self) -> Result<BpeFingerprint, TokenizerError> {
        let mut vocab = FNV_OFFSET;
        for token in ordered_vocab(&self.vocab)? {
            hash_token(&mut vocab, token);
        }

        let mut merges = FNV_OFFSET;
        for (left, right) in &self.merges {
            hash_token(&mut merges, left);
            hash_token(&mut merges, right);
        }
        Ok(BpeFingerprint { vocab, merges })
    }
}

const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
const FNV_PRIME: u64 = 0x100_0000_01b3;

#[derive(Clone, Copy, Debug)]
struct MergeRule {
    rank: u32,
    new_id: u32,
}

#[derive(Clone, Copy, Debug, Default)]
struct ByteRange {
    start: u32,
    len: u32,
}

#[derive(Clone, Copy, Debug)]
struct Symbol {
    id: u32,
    prev: isize,
    next: isize,
    live: bool,
}

pub(super) struct BpeModel {
    vocab: HashMap<String, u32>,
    merges: HashMap<(u32, u32), MergeRule>,
    byte_ids: [u32; 256],
    decoded_bytes: Vec<u8>,
    decoded_ranges: Vec<ByteRange>,
}

impl BpeModel {
    pub(super) fn new(definition: BpeDefinition) -> Result<Self, TokenizerError> {
        let BpeDefinition { vocab, merges } = definition;
        if vocab.len() > i32::MAX as usize {
            return Err(TokenizerError::invalid_vocabulary(
                "base vocabulary exceeds i32 token ID range",
            ));
        }
        ordered_vocab(&vocab)?;

        let mut byte_ids = [u32::MAX; 256];
        for byte in 0_u16..=255 {
            let symbol = byte_char(byte as u8).to_string();
            byte_ids[usize::from(byte)] = *vocab.get(&symbol).ok_or_else(|| {
                TokenizerError::invalid_vocabulary(format!(
                    "missing byte-alphabet symbol {symbol:?} for byte 0x{byte:02x}"
                ))
            })?;
        }

        let (decoded_bytes, decoded_ranges) = build_decode_table(&vocab)?;
        let mut merge_map = HashMap::with_capacity(merges.len());
        for (rank, (left, right)) in merges.into_iter().enumerate() {
            let rank = u32::try_from(rank).map_err(|_| {
                TokenizerError::invalid_vocabulary("merge table exceeds u32 rank range")
            })?;
            let left_id = *vocab.get(&left).ok_or_else(|| {
                TokenizerError::invalid_vocabulary(format!(
                    "merge rank {rank} references missing left token {left:?}"
                ))
            })?;
            let right_id = *vocab.get(&right).ok_or_else(|| {
                TokenizerError::invalid_vocabulary(format!(
                    "merge rank {rank} references missing right token {right:?}"
                ))
            })?;
            let merged = format!("{left}{right}");
            let new_id = *vocab.get(&merged).ok_or_else(|| {
                TokenizerError::invalid_vocabulary(format!(
                    "merge rank {rank} produces missing token {merged:?}"
                ))
            })?;
            if merge_map
                .insert((left_id, right_id), MergeRule { rank, new_id })
                .is_some()
            {
                return Err(TokenizerError::invalid_vocabulary(format!(
                    "merge rank {rank} duplicates pair {left:?} + {right:?}"
                )));
            }
        }

        Ok(Self {
            vocab,
            merges: merge_map,
            byte_ids,
            decoded_bytes,
            decoded_ranges,
        })
    }

    pub(super) fn encode_piece(&self, piece: &str, output: &mut Vec<i32>) {
        let mut symbols: Vec<Symbol> = Vec::with_capacity(piece.chars().count());
        for ch in piece.chars() {
            let byte = char_byte(ch).expect("pre-tokenized pieces use only byte-alphabet chars");
            let index = symbols.len() as isize;
            if let Some(previous) = symbols.last_mut() {
                previous.next = index;
            }
            symbols.push(Symbol {
                id: self.byte_ids[usize::from(byte)],
                prev: index - 1,
                next: -1,
                live: true,
            });
        }
        if symbols.len() > 1 {
            self.merge_all(&mut symbols);
        }
        output.extend(
            symbols
                .into_iter()
                .filter(|symbol| symbol.live)
                .map(|symbol| symbol.id as i32),
        );
    }

    pub(super) fn base_token_bytes(&self, id: i32) -> Option<&[u8]> {
        let id = usize::try_from(id).ok()?;
        let range = *self.decoded_ranges.get(id)?;
        let start = range.start as usize;
        let end = start + range.len as usize;
        Some(&self.decoded_bytes[start..end])
    }

    pub(super) fn token_id(&self, token: &str) -> Option<i32> {
        self.vocab.get(token).and_then(|id| i32::try_from(*id).ok())
    }

    pub(super) fn token_count(&self) -> usize {
        self.decoded_ranges.len()
    }

    fn merge_all(&self, symbols: &mut [Symbol]) {
        // Hugging Face BPE orders individual merge positions by rank and then
        // by their left index. Newly formed pairs re-enter the same queue;
        // stale entries are accepted only when their replacement ID still
        // matches the pair currently occupying that position.
        let mut queue = BinaryHeap::with_capacity(symbols.len());
        for position in 0..symbols.len() - 1 {
            self.push_candidate(symbols, position, &mut queue);
        }

        while let Some(Reverse((_, position, queued_new_id))) = queue.pop() {
            if !symbols[position].live || symbols[position].next < 0 {
                continue;
            }
            let right_position = symbols[position].next as usize;
            let pair = (symbols[position].id, symbols[right_position].id);
            let Some(rule) = self.merges.get(&pair) else {
                continue;
            };
            if rule.new_id != queued_new_id {
                continue;
            }

            let previous = symbols[position].prev;
            let next = symbols[right_position].next;
            symbols[position].id = rule.new_id;
            symbols[position].next = next;
            symbols[right_position].live = false;
            if next >= 0 {
                symbols[next as usize].prev = position as isize;
            }
            if previous >= 0 {
                self.push_candidate(symbols, previous as usize, &mut queue);
            }
            self.push_candidate(symbols, position, &mut queue);
        }
    }

    fn push_candidate(
        &self,
        symbols: &[Symbol],
        left: usize,
        queue: &mut BinaryHeap<Reverse<(u32, usize, u32)>>,
    ) {
        if !symbols[left].live || symbols[left].next < 0 {
            return;
        }
        let right = symbols[left].next as usize;
        if let Some(rule) = self.merges.get(&(symbols[left].id, symbols[right].id)) {
            queue.push(Reverse((rule.rank, left, rule.new_id)));
        }
    }
}

fn ordered_vocab(vocab: &HashMap<String, u32>) -> Result<Vec<&str>, TokenizerError> {
    let mut tokens_by_id = vec![None; vocab.len()];
    for (token, id) in vocab {
        let index = usize::try_from(*id).map_err(|_| {
            TokenizerError::invalid_vocabulary(format!("token {token:?} ID {id} overflows usize"))
        })?;
        let Some(slot) = tokens_by_id.get_mut(index) else {
            return Err(TokenizerError::invalid_vocabulary(format!(
                "token {token:?} ID {id} is outside contiguous range 0..{}",
                vocab.len()
            )));
        };
        if let Some(other) = slot.replace(token) {
            return Err(TokenizerError::invalid_vocabulary(format!(
                "tokens {other:?} and {token:?} share ID {id}"
            )));
        }
    }
    if let Some(missing) = tokens_by_id.iter().position(Option::is_none) {
        return Err(TokenizerError::invalid_vocabulary(format!(
            "base vocabulary is missing ID {missing}"
        )));
    }
    Ok(tokens_by_id
        .into_iter()
        .map(|token| token.expect("missing IDs checked"))
        .map(String::as_str)
        .collect())
}

fn hash_token(hash: &mut u64, token: &str) {
    for byte in (token.len() as u64).to_le_bytes() {
        *hash ^= u64::from(byte);
        *hash = hash.wrapping_mul(FNV_PRIME);
    }
    for &byte in token.as_bytes() {
        *hash ^= u64::from(byte);
        *hash = hash.wrapping_mul(FNV_PRIME);
    }
}

fn build_decode_table(
    vocab: &HashMap<String, u32>,
) -> Result<(Vec<u8>, Vec<ByteRange>), TokenizerError> {
    let estimated_bytes = vocab.keys().map(String::len).sum();
    let mut decoded_bytes = Vec::with_capacity(estimated_bytes);
    let mut decoded_ranges = vec![ByteRange::default(); vocab.len()];
    for (token, id) in vocab {
        let start = u32::try_from(decoded_bytes.len()).map_err(|_| {
            TokenizerError::invalid_vocabulary("decoded vocabulary exceeds u32 byte range")
        })?;
        for ch in token.chars() {
            decoded_bytes.push(char_byte(ch).ok_or_else(|| {
                TokenizerError::invalid_vocabulary(format!(
                    "base token {token:?} contains non-byte-alphabet character {ch:?}"
                ))
            })?);
        }
        let len = u32::try_from(decoded_bytes.len() - start as usize).map_err(|_| {
            TokenizerError::invalid_vocabulary(format!("base token {token:?} is too long"))
        })?;
        decoded_ranges[*id as usize] = ByteRange { start, len };
    }
    Ok((decoded_bytes, decoded_ranges))
}
