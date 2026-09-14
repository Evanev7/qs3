use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io::Read;
use std::path::Path;

use tinyjson::JsonValue;

use super::TokenizerError;
use super::bpe::BpeDefinition;
use super::pretokenize::QWEN_SPLIT_PATTERN;

const MAX_TOKENIZER_JSON_BYTES: usize = 64 << 20;

#[derive(Debug)]
pub(super) struct AddedTokenDefinition {
    pub(super) id: u32,
    pub(super) content: String,
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
        Self::parse_json(&source)
    }

    pub(super) fn parse_json(source: &str) -> Result<Self, TokenizerError> {
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

        let added_tokens = parse_added_tokens(added_tokens, bpe.vocab())?;
        Ok(Self { bpe, added_tokens })
    }

    pub(super) fn into_runtime_parts(self) -> (BpeDefinition, Vec<AddedTokenDefinition>) {
        (self.bpe, self.added_tokens)
    }
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
}

fn validate_normalizer(value: JsonValue) -> Result<(), TokenizerError> {
    let mut normalizer = JsonObject::new("tokenizer.normalizer", value)?;
    require_string(&mut normalizer, "type", "NFC")?;
    Ok(())
}

fn validate_pre_tokenizer(value: JsonValue) -> Result<(), TokenizerError> {
    let mut sequence = JsonObject::new("tokenizer.pre_tokenizer", value)?;
    require_string(&mut sequence, "type", "Sequence")?;
    let components = array(
        sequence.take("pretokenizers")?,
        "tokenizer.pre_tokenizer.pretokenizers",
    )?;

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

    require_string(&mut split, "behavior", "Isolated")?;
    require_bool(&mut split, "invert", false)?;
    Ok(())
}

fn validate_byte_level(context: &str, value: JsonValue) -> Result<(), TokenizerError> {
    let mut byte_level = JsonObject::new(context, value)?;
    require_string(&mut byte_level, "type", "ByteLevel")?;
    require_bool(&mut byte_level, "add_prefix_space", false)?;
    require_bool(&mut byte_level, "trim_offsets", false)?;
    require_bool(&mut byte_level, "use_regex", false)?;
    Ok(())
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
    let mut contents = HashSet::with_capacity(values.len());
    for (index, value) in values.into_iter().enumerate() {
        let context = format!("tokenizer.added_tokens[{index}]");
        let mut token = JsonObject::new(&context, value)?;
        let id = take_u32(&mut token, "id")?;
        let content = take_string(&mut token, "content")?;
        // Decode preserves every added token, including special tokens.
        take_bool(&mut token, "special")?;
        require_bool(&mut token, "single_word", false)?;
        require_bool(&mut token, "lstrip", false)?;
        require_bool(&mut token, "rstrip", false)?;
        require_bool(&mut token, "normalized", false)?;

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
        if !contents.insert(content.clone()) {
            return Err(TokenizerError::invalid_vocabulary(format!(
                "added token content {content:?} is duplicated"
            )));
        }
        tokens.push(AddedTokenDefinition { id, content });
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
