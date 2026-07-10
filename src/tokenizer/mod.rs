mod bpe;
mod error;
mod format;
mod pretokenize;

pub use error::TokenizerError;

use std::collections::HashMap;
use std::fmt;
use std::path::Path;

use unicode_normalization::UnicodeNormalization;

use bpe::BpeModel;
use format::TokenizerDefinition;
use pretokenize::{byte_pieces, char_byte};

/// The exact host-side tokenizer used by the supported Qwen3.6 artifact.
///
/// Construction validates the complete tokenizer pipeline and vocabulary.
/// Encoding therefore has no unknown-token path and never inserts BOS or EOS.
pub struct QwenTokenizer {
    model: BpeModel,
    added_tokens: Vec<AddedToken>,
    added_token_ids: HashMap<String, i32>,
}

struct AddedToken {
    id: i32,
    content: String,
}

impl QwenTokenizer {
    /// Load `tokenizer.json` from a model snapshot directory.
    pub fn from_model_dir(path: impl AsRef<Path>) -> Result<Self, TokenizerError> {
        Self::from_file(path.as_ref().join("tokenizer.json"))
    }

    /// Load and validate a Qwen3.6 `tokenizer.json` artifact.
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self, TokenizerError> {
        Self::from_definition(TokenizerDefinition::read(path)?)
    }

    /// Encode text without adding any control tokens.
    ///
    /// Added-token literals from the artifact are recognized before NFC
    /// normalization, exactly as declared by their `normalized: false` policy.
    pub fn encode(&self, text: &str) -> Vec<i32> {
        let mut output = Vec::new();
        let mut cursor = 0;
        while let Some((start, end, id)) = self.next_added_token(text, cursor) {
            self.encode_ordinary(&text[cursor..start], &mut output);
            output.push(id);
            cursor = end;
        }
        self.encode_ordinary(&text[cursor..], &mut output);
        output
    }

    /// Decode token IDs while preserving all added and special-token literals.
    pub fn decode(&self, token_ids: &[i32]) -> Result<String, TokenizerError> {
        let mut bytes = Vec::new();
        for &id in token_ids {
            if let Some(token_bytes) = self.model.base_token_bytes(id) {
                bytes.extend_from_slice(token_bytes);
                continue;
            }
            let Some(offset) = id
                .checked_sub(self.model.token_count() as i32)
                .and_then(|offset| usize::try_from(offset).ok())
            else {
                return Err(TokenizerError::UnknownTokenId(id));
            };
            let Some(token) = self.added_tokens.get(offset) else {
                return Err(TokenizerError::UnknownTokenId(id));
            };
            if token.id != id {
                return Err(TokenizerError::UnknownTokenId(id));
            }
            append_byte_level_decoded(&token.content, &mut bytes);
        }
        Ok(String::from_utf8_lossy(&bytes).into_owned())
    }

    /// Look up a base or added-token literal.
    pub fn token_id(&self, token: &str) -> Option<i32> {
        self.added_token_ids
            .get(token)
            .copied()
            .or_else(|| self.model.token_id(token))
    }

    /// Number of addressable tokenizer IDs, excluding padded model-logit slots.
    pub fn token_count(&self) -> usize {
        self.model.token_count() + self.added_tokens.len()
    }

    fn from_definition(definition: TokenizerDefinition) -> Result<Self, TokenizerError> {
        let (bpe, added_tokens) = definition.into_runtime_parts();
        let model = BpeModel::new(bpe)?;
        let mut contents = Vec::with_capacity(added_tokens.len());
        let mut ids = HashMap::with_capacity(added_tokens.len());
        for token in added_tokens {
            let id = i32::try_from(token.id).map_err(|_| {
                TokenizerError::invalid_vocabulary(format!(
                    "added token {:?} ID {} exceeds i32",
                    token.content, token.id
                ))
            })?;
            ids.insert(token.content.clone(), id);
            contents.push(AddedToken {
                id,
                content: token.content,
            });
        }
        Ok(Self {
            model,
            added_tokens: contents,
            added_token_ids: ids,
        })
    }

    fn encode_ordinary(&self, text: &str, output: &mut Vec<i32>) {
        if text.is_empty() {
            return;
        }
        let normalized: String = text.nfc().collect();
        for piece in byte_pieces(&normalized) {
            self.model.encode_piece(&piece, output);
        }
    }

    fn next_added_token(&self, text: &str, cursor: usize) -> Option<(usize, usize, i32)> {
        // The artifact has only 26 added tokens, so a direct scan keeps the
        // leftmost-longest rule explicit without importing a trie package.
        let remaining = &text[cursor..];
        let mut best: Option<(usize, usize, i32)> = None;
        for token in &self.added_tokens {
            let Some(relative_start) = remaining.find(&token.content) else {
                continue;
            };
            let start = cursor + relative_start;
            let candidate = (start, token.content.len(), token.id);
            if best.is_none_or(|current| {
                candidate.0 < current.0 || (candidate.0 == current.0 && candidate.1 > current.1)
            }) {
                best = Some(candidate);
            }
        }
        best.map(|(start, len, id)| (start, start + len, id))
    }
}

impl fmt::Debug for QwenTokenizer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("QwenTokenizer")
            .field("base_tokens", &self.model.token_count())
            .field("added_tokens", &self.added_tokens.len())
            .finish()
    }
}

fn append_byte_level_decoded(token: &str, output: &mut Vec<u8>) {
    let start = output.len();
    for ch in token.chars() {
        let Some(byte) = char_byte(ch) else {
            output.truncate(start);
            output.extend_from_slice(token.as_bytes());
            return;
        };
        output.push(byte);
    }
}

#[cfg(test)]
mod tests;
