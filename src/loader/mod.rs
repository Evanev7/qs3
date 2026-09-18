#![allow(dead_code)]

mod materialize;
mod plan;
mod prepare;
mod quantization;
mod schema;

const DEFAULT_MAX_JSON_BYTES: usize = 64 << 20;
const DEFAULT_MAX_HEADER_BYTES: usize = 256 << 20;
const CONFIG_FILE: &str = "config.json";
const SAFETENSORS_INDEX_FILE: &str = "model.safetensors.index.json";
const PINNED_UPLOAD_BUFFER_COUNT: usize = 4;
const PINNED_UPLOAD_BUFFER_BYTES: usize = 1 << 30;

mod transfer;

pub(crate) mod benchmark;
mod format;
#[cfg(test)]
mod tests;
