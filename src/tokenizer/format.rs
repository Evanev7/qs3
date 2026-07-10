use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io::Read;
use std::path::Path;

use tinyjson::JsonValue;

use super::TokenizerError;
use super::bpe::BpeDefinition;
use super::pretokenize::QWEN_SPLIT_PATTERN;

const MAX_TOKENIZER_JSON_BYTES: usize = 64 << 20;
const QWEN36_BASE_TOKEN_COUNT: usize = 248_044;
const QWEN36_MERGE_COUNT: usize = 247_587;
const QWEN36_VOCAB_FINGERPRINT: u64 = 0x3816_dfff_87e4_58d6;
const QWEN36_MERGES_FINGERPRINT: u64 = 0x5744_a130_3ff2_f51d;

const QWEN36_ADDED_TOKENS: &[(u32, &str, bool)] = &[
    (248_044, "<|endoftext|>", true),
    (248_045, "<|im_start|>", true),
    (248_046, "<|im_end|>", true),
    (248_047, "<|object_ref_start|>", true),
    (248_048, "<|object_ref_end|>", true),
    (248_049, "<|box_start|>", true),
    (248_050, "<|box_end|>", true),
    (248_051, "<|quad_start|>", true),
    (248_052, "<|quad_end|>", true),
    (248_053, "<|vision_start|>", true),
    (248_054, "<|vision_end|>", true),
    (248_055, "<|vision_pad|>", true),
    (248_056, "<|image_pad|>", true),
    (248_057, "<|video_pad|>", true),
    (248_058, "<tool_call>", false),
    (248_059, "</tool_call>", false),
    (248_060, "<|fim_prefix|>", false),
    (248_061, "<|fim_middle|>", false),
    (248_062, "<|fim_suffix|>", false),
    (248_063, "<|fim_pad|>", false),
    (248_064, "<|repo_name|>", false),
    (248_065, "<|file_sep|>", false),
    (248_066, "<tool_response>", false),
    (248_067, "</tool_response>", false),
    (248_068, "<think>", false),
    (248_069, "</think>", false),
];

#[derive(Debug)]
pub(super) struct AddedTokenDefinition {
    pub(super) id: u32,
    pub(super) content: String,
    pub(super) special: bool,
}

pub(super) struct TokenizerDefinition {
    bpe: BpeDefinition,
    added_tokens: Vec<AddedTokenDefinition>,
}

impl TokenizerDefinition {
    pub(super) fn read(path: impl AsRef<Path>) -> Result<Self, TokenizerError> {
        let path = path.as_ref();
        let mut file = File::open(path).map_err(|error| {
            TokenizerError::Io(format!("failed to open {}: {error}", path.display()))
        })?;
        let len = file
            .metadata()
            .map_err(|error| {
                TokenizerError::Io(format!("failed to stat {}: {error}", path.display()))
            })?
            .len();
        if len > MAX_TOKENIZER_JSON_BYTES as u64 {
            return Err(TokenizerError::json(format!(
                "{} is {len} bytes; limit is {MAX_TOKENIZER_JSON_BYTES}",
                path.display()
            )));
        }
        let mut source = String::with_capacity(len as usize);
        file.read_to_string(&mut source).map_err(|error| {
            TokenizerError::Io(format!("failed to read {}: {error}", path.display()))
        })?;
        let definition = Self::parse_json(&source)?;
        definition.validate_qwen36_identity()?;
        Ok(definition)
    }

    pub(super) fn parse_json(source: &str) -> Result<Self, TokenizerError> {
        validate_u32_number_literals(source)?;
        let value = source
            .parse::<JsonValue>()
            .map_err(|error| TokenizerError::json(error.to_string()))?;
        let mut root = JsonObject::new("tokenizer", value)?;
        require_string(&mut root, "version", "1.0")?;
        require_null(&mut root, "truncation")?;
        require_null(&mut root, "padding")?;
        let added_tokens = root.take("added_tokens")?;
        validate_normalizer(root.take("normalizer")?)?;
        validate_pre_tokenizer(root.take("pre_tokenizer")?)?;
        validate_byte_level("post_processor", root.take("post_processor")?)?;
        validate_byte_level("decoder", root.take("decoder")?)?;
        let bpe = parse_bpe(root.take("model")?)?;
        root.finish()?;

        let added_tokens = parse_added_tokens(added_tokens, bpe.vocab())?;
        Ok(Self { bpe, added_tokens })
    }

    pub(super) fn validate_qwen36_identity(&self) -> Result<(), TokenizerError> {
        if self.bpe.token_count() != QWEN36_BASE_TOKEN_COUNT {
            return Err(TokenizerError::invalid_vocabulary(format!(
                "expected {QWEN36_BASE_TOKEN_COUNT} base tokens, found {}",
                self.bpe.token_count()
            )));
        }
        if self.bpe.merge_count() != QWEN36_MERGE_COUNT {
            return Err(TokenizerError::invalid_vocabulary(format!(
                "expected {QWEN36_MERGE_COUNT} BPE merges, found {}",
                self.bpe.merge_count()
            )));
        }
        let fingerprint = self.bpe.fingerprint()?;
        if fingerprint.vocab != QWEN36_VOCAB_FINGERPRINT {
            return Err(TokenizerError::invalid_vocabulary(format!(
                "base vocabulary fingerprint is {:016x}, expected {QWEN36_VOCAB_FINGERPRINT:016x}",
                fingerprint.vocab
            )));
        }
        if fingerprint.merges != QWEN36_MERGES_FINGERPRINT {
            return Err(TokenizerError::invalid_vocabulary(format!(
                "BPE merge fingerprint is {:016x}, expected {QWEN36_MERGES_FINGERPRINT:016x}",
                fingerprint.merges
            )));
        }
        if self.added_tokens.len() != QWEN36_ADDED_TOKENS.len() {
            return Err(TokenizerError::invalid_vocabulary(format!(
                "expected {} Qwen added tokens, found {}",
                QWEN36_ADDED_TOKENS.len(),
                self.added_tokens.len()
            )));
        }
        for (actual, &(id, content, special)) in self.added_tokens.iter().zip(QWEN36_ADDED_TOKENS) {
            if actual.id != id || actual.content != content || actual.special != special {
                return Err(TokenizerError::invalid_vocabulary(format!(
                    "added token ID {id} must be {content:?} with special={special}"
                )));
            }
        }
        Ok(())
    }

    #[cfg(test)]
    pub(super) fn base_token_count(&self) -> usize {
        self.bpe.token_count()
    }

    #[cfg(test)]
    pub(super) fn added_token_count(&self) -> usize {
        self.added_tokens.len()
    }

    pub(super) fn into_runtime_parts(self) -> (BpeDefinition, Vec<AddedTokenDefinition>) {
        (self.bpe, self.added_tokens)
    }
}

fn validate_u32_number_literals(source: &str) -> Result<(), TokenizerError> {
    // tinyjson intentionally represents every JSON number as f64. This exact
    // artifact has only u32 token IDs, so reject non-canonical spellings before
    // f64 rounding can turn a fraction, exponent, or negative value into one.
    let bytes = source.as_bytes();
    let mut cursor = 0;
    while cursor < bytes.len() {
        if bytes[cursor] == b'"' {
            cursor = string_end(bytes, cursor);
            continue;
        }
        if bytes[cursor] == b'-' || bytes[cursor].is_ascii_digit() {
            let start = cursor;
            cursor += 1;
            while cursor < bytes.len()
                && matches!(
                    bytes[cursor],
                    b'0'..=b'9' | b'.' | b'e' | b'E' | b'+' | b'-'
                )
            {
                cursor += 1;
            }
            let literal = &source[start..cursor];
            let canonical = literal.bytes().all(|byte| byte.is_ascii_digit())
                && (literal.len() == 1 || !literal.starts_with('0'));
            if !canonical || literal.parse::<u32>().is_err() {
                return Err(TokenizerError::json(format!(
                    "numeric literal {literal:?} must be a canonical unsigned u32 token ID"
                )));
            }
            continue;
        }
        cursor += 1;
    }
    Ok(())
}

fn string_end(bytes: &[u8], start: usize) -> usize {
    let mut cursor = start + 1;
    while cursor < bytes.len() {
        match bytes[cursor] {
            b'\\' => cursor = (cursor + 2).min(bytes.len()),
            b'"' => return cursor + 1,
            _ => cursor += 1,
        }
    }
    cursor
}

struct JsonObject {
    context: String,
    fields: HashMap<String, JsonValue>,
}

impl JsonObject {
    fn new(context: impl Into<String>, value: JsonValue) -> Result<Self, TokenizerError> {
        let context = context.into();
        let fields = value
            .try_into()
            .map_err(|_| TokenizerError::json(format!("{context} must be a JSON object")))?;
        Ok(Self { context, fields })
    }

    fn take(&mut self, name: &str) -> Result<JsonValue, TokenizerError> {
        self.fields
            .remove(name)
            .ok_or_else(|| TokenizerError::json(format!("{}.{} is required", self.context, name)))
    }

    fn finish(self) -> Result<(), TokenizerError> {
        if self.fields.is_empty() {
            return Ok(());
        }
        let mut names: Vec<_> = self.fields.into_keys().collect();
        names.sort();
        Err(TokenizerError::unsupported(format!(
            "{} has unsupported fields: {}",
            self.context,
            names.join(", ")
        )))
    }
}

fn validate_normalizer(value: JsonValue) -> Result<(), TokenizerError> {
    let mut normalizer = JsonObject::new("tokenizer.normalizer", value)?;
    require_string(&mut normalizer, "type", "NFC")?;
    normalizer.finish()
}

fn validate_pre_tokenizer(value: JsonValue) -> Result<(), TokenizerError> {
    let mut sequence = JsonObject::new("tokenizer.pre_tokenizer", value)?;
    require_string(&mut sequence, "type", "Sequence")?;
    let components = array(
        sequence.take("pretokenizers")?,
        "tokenizer.pre_tokenizer.pretokenizers",
    )?;
    sequence.finish()?;
    if components.len() != 2 {
        return Err(TokenizerError::unsupported(format!(
            "tokenizer.pre_tokenizer must contain exactly two components, found {}",
            components.len()
        )));
    }
    let mut components = components.into_iter();
    validate_split(components.next().expect("length checked"))?;
    validate_byte_level(
        "tokenizer.pre_tokenizer.pretokenizers[1]",
        components.next().expect("length checked"),
    )
}

fn validate_split(value: JsonValue) -> Result<(), TokenizerError> {
    let mut split = JsonObject::new("tokenizer.pre_tokenizer.pretokenizers[0]", value)?;
    require_string(&mut split, "type", "Split")?;
    let mut pattern = JsonObject::new(
        "tokenizer.pre_tokenizer.pretokenizers[0].pattern",
        split.take("pattern")?,
    )?;
    require_string(&mut pattern, "Regex", QWEN_SPLIT_PATTERN)?;
    pattern.finish()?;
    require_string(&mut split, "behavior", "Isolated")?;
    require_bool(&mut split, "invert", false)?;
    split.finish()
}

fn validate_byte_level(context: &str, value: JsonValue) -> Result<(), TokenizerError> {
    let mut byte_level = JsonObject::new(context, value)?;
    require_string(&mut byte_level, "type", "ByteLevel")?;
    require_bool(&mut byte_level, "add_prefix_space", false)?;
    require_bool(&mut byte_level, "trim_offsets", false)?;
    require_bool(&mut byte_level, "use_regex", false)?;
    byte_level.finish()
}

fn parse_bpe(value: JsonValue) -> Result<BpeDefinition, TokenizerError> {
    let mut model = JsonObject::new("tokenizer.model", value)?;
    require_string(&mut model, "type", "BPE")?;
    require_null(&mut model, "dropout")?;
    require_null(&mut model, "unk_token")?;
    require_string(&mut model, "continuing_subword_prefix", "")?;
    require_string(&mut model, "end_of_word_suffix", "")?;
    require_bool(&mut model, "fuse_unk", false)?;
    require_bool(&mut model, "byte_fallback", false)?;
    require_bool(&mut model, "ignore_merges", false)?;
    let vocab = object_map(model.take("vocab")?, "tokenizer.model.vocab")?;
    let merges = array(model.take("merges")?, "tokenizer.model.merges")?;
    model.finish()?;

    let mut parsed_vocab = HashMap::with_capacity(vocab.len());
    for (token, value) in vocab {
        let id = json_u32(value, format!("tokenizer.model.vocab[{token:?}]"))?;
        parsed_vocab.insert(token, id);
    }
    let mut parsed_merges = Vec::with_capacity(merges.len());
    for (rank, value) in merges.into_iter().enumerate() {
        let merge = json_string(value, format!("tokenizer.model.merges[{rank}]"))?;
        let Some((left, right)) = merge.split_once(' ') else {
            return Err(TokenizerError::invalid_vocabulary(format!(
                "merge rank {rank} must contain exactly two tokens"
            )));
        };
        if left.is_empty() || right.is_empty() || right.contains(' ') {
            return Err(TokenizerError::invalid_vocabulary(format!(
                "merge rank {rank} must contain exactly two non-empty tokens"
            )));
        }
        parsed_merges.push((left.to_owned(), right.to_owned()));
    }
    Ok(BpeDefinition::new(parsed_vocab, parsed_merges))
}

fn parse_added_tokens(
    value: JsonValue,
    vocab: &HashMap<String, u32>,
) -> Result<Vec<AddedTokenDefinition>, TokenizerError> {
    let values = array(value, "tokenizer.added_tokens")?;
    let mut tokens = Vec::with_capacity(values.len());
    let mut ids = HashSet::with_capacity(values.len());
    let mut contents = HashSet::with_capacity(values.len());
    for (index, value) in values.into_iter().enumerate() {
        let context = format!("tokenizer.added_tokens[{index}]");
        let mut token = JsonObject::new(&context, value)?;
        let id = take_u32(&mut token, "id")?;
        let content = take_string(&mut token, "content")?;
        let special = take_bool(&mut token, "special")?;
        require_bool(&mut token, "single_word", false)?;
        require_bool(&mut token, "lstrip", false)?;
        require_bool(&mut token, "rstrip", false)?;
        require_bool(&mut token, "normalized", false)?;
        token.finish()?;

        if content.is_empty() {
            return Err(TokenizerError::invalid_vocabulary(format!(
                "{context}.content must not be empty"
            )));
        }
        if vocab.contains_key(&content) {
            return Err(TokenizerError::invalid_vocabulary(format!(
                "added token {content:?} duplicates a base-vocabulary token"
            )));
        }
        if !ids.insert(id) {
            return Err(TokenizerError::invalid_vocabulary(format!(
                "added token ID {id} is duplicated"
            )));
        }
        if !contents.insert(content.clone()) {
            return Err(TokenizerError::invalid_vocabulary(format!(
                "added token content {content:?} is duplicated"
            )));
        }
        tokens.push(AddedTokenDefinition {
            id,
            content,
            special,
        });
    }

    tokens.sort_unstable_by_key(|token| token.id);
    for (offset, token) in tokens.iter().enumerate() {
        let expected = vocab
            .len()
            .checked_add(offset)
            .and_then(|id| u32::try_from(id).ok())
            .ok_or_else(|| {
                TokenizerError::invalid_vocabulary("combined vocabulary exceeds u32 ID range")
            })?;
        if token.id != expected {
            return Err(TokenizerError::invalid_vocabulary(format!(
                "added token IDs must be contiguous after base vocabulary: expected {expected}, found {}",
                token.id
            )));
        }
    }
    Ok(tokens)
}

fn require_string(
    object: &mut JsonObject,
    name: &str,
    expected: &str,
) -> Result<(), TokenizerError> {
    let actual = take_string(object, name)?;
    if actual == expected {
        Ok(())
    } else {
        Err(TokenizerError::unsupported(format!(
            "{}.{} must be {expected:?}, found {actual:?}",
            object.context, name
        )))
    }
}

fn require_bool(object: &mut JsonObject, name: &str, expected: bool) -> Result<(), TokenizerError> {
    let actual = take_bool(object, name)?;
    if actual == expected {
        Ok(())
    } else {
        Err(TokenizerError::unsupported(format!(
            "{}.{} must be {expected}, found {actual}",
            object.context, name
        )))
    }
}

fn require_null(object: &mut JsonObject, name: &str) -> Result<(), TokenizerError> {
    if object.take(name)?.is_null() {
        Ok(())
    } else {
        Err(TokenizerError::unsupported(format!(
            "{}.{} must be null",
            object.context, name
        )))
    }
}

fn take_string(object: &mut JsonObject, name: &str) -> Result<String, TokenizerError> {
    json_string(object.take(name)?, format!("{}.{}", object.context, name))
}

fn take_bool(object: &mut JsonObject, name: &str) -> Result<bool, TokenizerError> {
    object
        .take(name)?
        .try_into()
        .map_err(|_| TokenizerError::json(format!("{}.{} must be a boolean", object.context, name)))
}

fn take_u32(object: &mut JsonObject, name: &str) -> Result<u32, TokenizerError> {
    json_u32(object.take(name)?, format!("{}.{}", object.context, name))
}

fn json_string(value: JsonValue, context: impl AsRef<str>) -> Result<String, TokenizerError> {
    value
        .try_into()
        .map_err(|_| TokenizerError::json(format!("{} must be a string", context.as_ref())))
}

fn json_u32(value: JsonValue, context: impl AsRef<str>) -> Result<u32, TokenizerError> {
    let context = context.as_ref();
    let number: f64 = value
        .try_into()
        .map_err(|_| TokenizerError::json(format!("{context} must be an integer")))?;
    if !number.is_finite() || number < 0.0 || number.fract() != 0.0 || number > u32::MAX as f64 {
        return Err(TokenizerError::json(format!(
            "{context} must be a non-negative integer that fits u32"
        )));
    }
    Ok(number as u32)
}

fn array(value: JsonValue, context: &str) -> Result<Vec<JsonValue>, TokenizerError> {
    value
        .try_into()
        .map_err(|_| TokenizerError::json(format!("{context} must be an array")))
}

fn object_map(
    value: JsonValue,
    context: &str,
) -> Result<HashMap<String, JsonValue>, TokenizerError> {
    value
        .try_into()
        .map_err(|_| TokenizerError::json(format!("{context} must be an object")))
}
