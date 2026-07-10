use std::fmt;

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum TokenizerError {
    Io(String),
    Json(String),
    UnsupportedDefinition(String),
    InvalidVocabulary(String),
    UnknownTokenId(i32),
}

impl TokenizerError {
    pub(super) fn json(message: impl Into<String>) -> Self {
        Self::Json(message.into())
    }

    pub(super) fn unsupported(message: impl Into<String>) -> Self {
        Self::UnsupportedDefinition(message.into())
    }

    pub(super) fn invalid_vocabulary(message: impl Into<String>) -> Self {
        Self::InvalidVocabulary(message.into())
    }
}

impl fmt::Display for TokenizerError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(message) => write!(f, "tokenizer I/O error: {message}"),
            Self::Json(message) => write!(f, "invalid tokenizer JSON: {message}"),
            Self::UnsupportedDefinition(message) => {
                write!(f, "unsupported Qwen3.6 tokenizer definition: {message}")
            }
            Self::InvalidVocabulary(message) => {
                write!(f, "invalid Qwen3.6 tokenizer vocabulary: {message}")
            }
            Self::UnknownTokenId(id) => write!(f, "unknown tokenizer token ID {id}"),
        }
    }
}

impl std::error::Error for TokenizerError {}

impl From<std::io::Error> for TokenizerError {
    fn from(error: std::io::Error) -> Self {
        Self::Io(error.to_string())
    }
}
