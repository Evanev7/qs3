#![allow(dead_code)]

mod materialize;
mod plan;

use crate::engine::{DynDType, Status};
use crate::{
    QWEN36_FULL_ATTN_HEAD_DIM, QWEN36_FULL_ATTN_KV_HEADS, QWEN36_FULL_ATTN_KV_HIDDEN,
    QWEN36_FULL_ATTN_Q_HEADS, QWEN36_FULL_ATTN_Q_HIDDEN, QWEN36_FULL_ATTN_Q_PROJ_OUT,
    QWEN36_FULL_ATTN_ROTARY_DIM, QWEN36_GDN_CONV_WIDTH, QWEN36_GDN_KEY_DIM, QWEN36_GDN_NUM_K_HEADS,
    QWEN36_GDN_NUM_V_HEADS, QWEN36_GDN_OUTPUT_DIM, QWEN36_GDN_PACKED_DIM, QWEN36_GDN_VALUE_DIM,
    QWEN36_HIDDEN_SIZE, QWEN36_MOE_INTERMEDIATE_SIZE, QWEN36_MOE_NUM_EXPERTS,
    QWEN36_MOE_SHARED_EXPERT_INTERMEDIATE_SIZE, QWEN36_MOE_TOP_K,
};

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::ffi::c_void;
use std::fmt;
use std::fs::File;
use std::io::{self, Read};
use std::os::fd::AsRawFd;
use std::path::{Component, Path, PathBuf};
use std::ptr;
use std::time::Instant;
use tinyjson::JsonValue;

const DEFAULT_MAX_JSON_BYTES: usize = 64 << 20;
const DEFAULT_MAX_HEADER_BYTES: usize = 256 << 20;
const CONFIG_FILE: &str = "config.json";
const SAFETENSORS_INDEX_FILE: &str = "model.safetensors.index.json";
const TEXT_PREFIX: &str = "model.language_model.";
const PINNED_UPLOAD_BUFFER_COUNT: usize = 4;
const PINNED_UPLOAD_BUFFER_BYTES: usize = 1 << 30;

mod transfer;

#[cfg(test)]
use crate::ffi;
#[cfg(test)]
use transfer::{ManagedUmaBackend, PinnedUploadBackend, WeightLoadMemory, result_from_cuda};
use transfer::{WeightFileRange, WeightLoadBackend, WeightLoadSpan, WeightTensorDesc};
mod format;

use format::*;
use plan::*;
#[cfg(test)]
mod tests;
