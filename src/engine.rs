use crate::{ffi, runtime};
use std::collections::HashSet;

pub struct Engine {
    inner: Box<runtime::EngineInner>,
}

impl Engine {
    pub(crate) fn kernel_ops(&mut self) -> runtime::kernels::KernelOps<'_> {
        self.inner.kernel_ops()
    }
}

pub trait EngineTrait {
    fn new(config: EngineConfig) -> Result<Self, Status>
    where
        Self: Sized;

    fn reset(&mut self) -> Result<(), Status>;
    fn release_requests(&mut self, request_ids: &[RequestId]) -> Result<(), Status>;
    fn state(&self) -> Result<CoreState<'_>, Status>;
    fn begin_append(&mut self, batch: AppendBatch<'_>) -> Result<(), Status>;
    unsafe fn append_layer(&mut self, layer: &EngineLayer) -> Result<(), Status>;
    fn begin_decode(&mut self, batch: DecodeBatch<'_>) -> Result<(), Status>;
    unsafe fn decode_layer(&mut self, layer: &EngineLayer) -> Result<(), Status>;
    fn commit_batch(&mut self, commit: Commit<'_>) -> Result<(), Status>;
    fn abort_batch(&mut self) -> Result<(), Status>;
}

impl EngineTrait for Engine {
    fn new(config: EngineConfig) -> Result<Self, Status> {
        runtime::EngineInner::new(config).map(|inner| Self { inner })
    }

    fn reset(&mut self) -> Result<(), Status> {
        self.inner.core.reset()
    }

    fn release_requests(&mut self, request_ids: &[RequestId]) -> Result<(), Status> {
        self.inner.core.release_requests(request_ids)
    }

    fn state(&self) -> Result<CoreState<'_>, Status> {
        self.inner.core.state()
    }

    fn begin_append(&mut self, batch: AppendBatch<'_>) -> Result<(), Status> {
        self.inner.prepare_append(batch)
    }

    unsafe fn append_layer(&mut self, layer: &EngineLayer) -> Result<(), Status> {
        unsafe { self.inner.execute_append_layer(layer) }
    }

    fn begin_decode(&mut self, batch: DecodeBatch<'_>) -> Result<(), Status> {
        self.inner.prepare_decode(batch)
    }

    unsafe fn decode_layer(&mut self, layer: &EngineLayer) -> Result<(), Status> {
        unsafe { self.inner.execute_decode_layer(layer) }
    }

    fn commit_batch(&mut self, commit: Commit<'_>) -> Result<(), Status> {
        self.inner.commit_batch(commit)
    }

    fn abort_batch(&mut self) -> Result<(), Status> {
        self.inner.core.abort_batch()
    }
}

pub type RequestId = u64;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Status {
    InvalidArgument,
    Unsupported,
    OutOfMemory,
    CudaError,
    BackendError,
    InternalError,
    Unreachable,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DType {
    F32,
    F16,
    BF16,
    FP8E4M3,
    FP8E5M2,
    NVFP4E2M1,
    MXFP4E2M1,
    MXFP8E4M3,
    I32,
    U32,
    I8,
    U8,
}

impl DType {
    pub fn bits(self) -> usize {
        match self {
            DType::F32 | DType::I32 | DType::U32 => 32,
            DType::F16 | DType::BF16 => 16,
            DType::FP8E4M3 | DType::FP8E5M2 | DType::MXFP8E4M3 | DType::I8 | DType::U8 => 8,
            DType::NVFP4E2M1 | DType::MXFP4E2M1 => 4,
        }
    }

    pub fn storage_bytes_for(self, elements: usize) -> Result<usize, Status> {
        let bits = elements
            .checked_mul(self.bits())
            .ok_or(Status::InvalidArgument)?;
        bits.checked_add(7)
            .ok_or(Status::InvalidArgument)
            .map(|bits| bits / 8)
    }

    fn is_runtime_supported(self) -> bool {
        matches!(self, DType::F16 | DType::BF16)
    }

    pub(crate) fn to_raw(self) -> ffi::DTypeRaw {
        match self {
            DType::F32 => ffi::DTYPE_F32,
            DType::F16 => ffi::DTYPE_F16,
            DType::BF16 => ffi::DTYPE_BF16,
            DType::FP8E4M3 => ffi::DTYPE_FP8_E4M3,
            DType::FP8E5M2 => ffi::DTYPE_FP8_E5M2,
            DType::NVFP4E2M1 => ffi::DTYPE_NVFP4_E2M1,
            DType::MXFP4E2M1 => ffi::DTYPE_MXFP4_E2M1,
            DType::MXFP8E4M3 => ffi::DTYPE_MXFP8_E4M3,
            DType::I32 => ffi::DTYPE_I32,
            DType::U32 => ffi::DTYPE_U32,
            DType::I8 => ffi::DTYPE_I8,
            DType::U8 => ffi::DTYPE_U8,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum KvLayout {
    NHD,
    HND,
}

impl KvLayout {
    pub(crate) fn to_raw(self) -> ffi::KvLayoutRaw {
        match self {
            KvLayout::NHD => ffi::KV_LAYOUT_NHD,
            KvLayout::HND => ffi::KV_LAYOUT_HND,
        }
    }
}

pub(crate) fn validate_supported_attention_grouping(
    num_q_heads: u32,
    num_kv_heads: u32,
) -> Result<(), Status> {
    if num_q_heads == 0 || num_kv_heads == 0 || !num_q_heads.is_multiple_of(num_kv_heads) {
        return Err(Status::InvalidArgument);
    }
    if num_q_heads != num_kv_heads {
        return Err(Status::Unsupported);
    }
    Ok(())
}

pub(crate) fn validate_supported_attention_head_dim(head_dim: u32) -> Result<(), Status> {
    if head_dim == 0 {
        return Err(Status::InvalidArgument);
    }
    if head_dim != 64 {
        return Err(Status::Unsupported);
    }
    Ok(())
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BatchKind {
    Append,
    Decode,
}

#[derive(Clone, Copy, Debug)]
pub struct EngineConfig {
    pub device_ordinal: i32,
    pub stream: *mut std::ffi::c_void,
    pub num_layers: u32,
    pub max_live_requests: u32,
    pub max_batch_size: u32,
    pub max_seq_len: u32,
    pub max_pages: u32,
    pub page_size: u32,
    pub hidden_size: u32,
    pub intermediate_size: u32,
    pub vocab_size: u32,
    pub num_q_heads: u32,
    pub num_kv_heads: u32,
    pub head_dim: u32,
    pub activation_dtype: DType,
    pub kv_dtype: DType,
    pub kv_layout: KvLayout,
    pub rope_theta: f32,
    pub rope_scale: f32,
    pub logits_soft_cap: f32,
    pub qsfi_float_workspace_bytes: usize,
    pub qsfi_int_workspace_bytes: usize,
    pub qsfi_host_int_workspace_bytes: usize,
}

#[derive(Clone, Copy, Debug)]
pub struct AppendBatch<'a> {
    pub request_ids: &'a [RequestId],
    pub token_indptr: &'a [i32],
    pub tokens: &'a [i32],
}

#[derive(Clone, Copy, Debug)]
pub struct DecodeBatch<'a> {
    pub request_ids: &'a [RequestId],
    pub tokens: &'a [i32],
}

#[derive(Clone, Copy, Debug, Default)]
pub struct Commit<'a> {
    pub accepted_token_counts: Option<&'a [u32]>,
}

#[derive(Clone, Copy, Debug)]
pub struct EngineLayer {
    pub layer_idx: u32,
    pub q: ffi::Tensor3,
    pub k: ffi::Tensor3,
    pub v: ffi::Tensor3,
    pub o: ffi::Tensor3,
    pub q_rope_offset: ffi::DevicePtr,
    pub lse: ffi::DevicePtr,
    pub q_scale: f32,
    pub k_scale: f32,
    pub v_scale: f32,
}

impl EngineLayer {
    pub(crate) fn bf16_attention(
        layer_idx: u32,
        q: runtime::kernels::Bf16Heads,
        k: runtime::kernels::Bf16Heads,
        v: runtime::kernels::Bf16Heads,
        o: runtime::kernels::Bf16Heads,
        q_rope_offset: ffi::DevicePtr,
    ) -> Self {
        Self {
            layer_idx,
            q: q.tensor(),
            k: k.tensor(),
            v: v.tensor(),
            o: o.tensor(),
            q_rope_offset,
            lse: std::ptr::null_mut(),
            q_scale: 0.0,
            k_scale: 0.0,
            v_scale: 0.0,
        }
    }
}

#[derive(Clone, Debug)]
struct Request {
    id: RequestId,
    seq_len: u32,
    tokens: Vec<i32>,
    pages: Vec<i32>,
}

#[derive(Clone, Debug)]
struct LiveViews {
    request_ids: Vec<RequestId>,
    seq_lens: Vec<i32>,
    token_indptr: Vec<i32>,
    tokens: Vec<i32>,
    kv_indptr: Vec<i32>,
    kv_indices: Vec<i32>,
    last_page_len: Vec<i32>,
}

#[derive(Clone, Debug)]
struct StagedRow {
    id: RequestId,
    request_index: Option<usize>,
    old_seq_len: u32,
    old_page_count: usize,
    token_count: u32,
    tokens: Vec<i32>,
    pages: Vec<i32>,
}

#[derive(Clone, Debug)]
struct BatchState {
    size: u32,
    token_count: u32,
    rows: Vec<StagedRow>,
    layers: LayerProgress,
}

impl BatchState {
    fn new(size: u32, token_count: u32, rows: Vec<StagedRow>, layer_count: u32) -> Self {
        Self {
            size,
            token_count,
            rows,
            layers: LayerProgress::new(layer_count),
        }
    }
}

#[derive(Clone, Debug)]
enum ActiveBatch {
    Append(BatchState),
    Decode(BatchState),
}

impl ActiveBatch {
    fn append(size: u32, token_count: u32, rows: Vec<StagedRow>, layer_count: u32) -> Self {
        Self::Append(BatchState::new(size, token_count, rows, layer_count))
    }

    fn decode(size: u32, token_count: u32, rows: Vec<StagedRow>, layer_count: u32) -> Self {
        Self::Decode(BatchState::new(size, token_count, rows, layer_count))
    }

    fn kind(&self) -> BatchKind {
        match self {
            Self::Append(_) => BatchKind::Append,
            Self::Decode(_) => BatchKind::Decode,
        }
    }

    fn state(&self) -> &BatchState {
        match self {
            Self::Append(state) | Self::Decode(state) => state,
        }
    }

    fn is_decode(&self) -> bool {
        matches!(self, Self::Decode(_))
    }
}

#[derive(Clone, Debug)]
struct LayerProgress {
    next_layer: u32,
    layer_count: u32,
}

impl LayerProgress {
    fn new(layer_count: u32) -> Self {
        Self {
            next_layer: 0,
            layer_count,
        }
    }

    fn pending(&self, layer_idx: u32) -> Result<PendingLayer, Status> {
        if self.next_layer >= self.layer_count || layer_idx != self.next_layer {
            return Err(Status::InvalidArgument);
        }
        Ok(PendingLayer { layer_idx })
    }

    fn complete(&mut self, pending: PendingLayer) -> Result<(), Status> {
        if pending.layer_idx != self.next_layer {
            return Err(Status::InternalError);
        }
        self.next_layer = self
            .next_layer
            .checked_add(1)
            .ok_or(Status::InternalError)?;
        Ok(())
    }

    fn is_complete(&self) -> bool {
        self.next_layer == self.layer_count
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct PendingLayer {
    layer_idx: u32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct PendingAppendLayer(PendingLayer);

impl PendingAppendLayer {
    pub(crate) fn layer_idx(self) -> u32 {
        self.0.layer_idx
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct PendingDecodeLayer(PendingLayer);

impl PendingDecodeLayer {
    pub(crate) fn layer_idx(self) -> u32 {
        self.0.layer_idx
    }
}

#[derive(Debug)]
pub(crate) struct EngineCore {
    config: EngineConfig,
    requests: Vec<Request>,
    free_pages: Vec<i32>,

    live_request_ids: Vec<RequestId>,
    live_seq_lens: Vec<i32>,
    live_token_indptr: Vec<i32>,
    live_tokens: Vec<i32>,
    live_kv_indptr: Vec<i32>,
    live_kv_indices: Vec<i32>,
    live_last_page_len: Vec<i32>,

    active_batch: Option<ActiveBatch>,

    batch_request_ids: Vec<RequestId>,
    batch_tokens: Vec<i32>,
    batch_qo_indptr: Vec<i32>,
    batch_kv_indptr: Vec<i32>,
    batch_kv_indices: Vec<i32>,
    batch_last_page_len: Vec<i32>,
    batch_rope_pos_offset: Vec<i32>,
    batch_append_batch_indices: Vec<i32>,
    batch_append_positions: Vec<i32>,
}

#[derive(Clone, Copy)]
pub struct CoreState<'a> {
    pub batch_kind: Option<BatchKind>,
    pub live_request_count: u32,
    pub batch_size: u32,
    pub batch_token_count: u32,
    pub live_num_indices: u32,
    pub allocated_pages: u32,
    pub free_page_count: u32,
    pub max_pages: u32,
    pub page_size: u32,
    pub live_request_ids: &'a [RequestId],
    pub live_seq_lens: &'a [i32],
    pub live_token_indptr: &'a [i32],
    pub live_tokens: &'a [i32],
    pub live_kv_indptr: &'a [i32],
    pub live_kv_indices: &'a [i32],
    pub live_last_page_len: &'a [i32],
    pub free_pages: &'a [i32],
    pub batch_request_ids: &'a [RequestId],
    pub batch_tokens: &'a [i32],
    pub batch_qo_indptr: &'a [i32],
    pub batch_kv_indptr: &'a [i32],
    pub batch_kv_indices: &'a [i32],
    pub batch_last_page_len: &'a [i32],
    pub batch_rope_pos_offset: &'a [i32],
    pub batch_append_batch_indices: &'a [i32],
    pub batch_append_positions: &'a [i32],
}

fn try_reserve<T>(vec: &mut Vec<T>, additional: usize) -> Result<(), Status> {
    vec.try_reserve(additional).map_err(|_| Status::OutOfMemory)
}

fn try_vec_with_capacity<T>(capacity: usize) -> Result<Vec<T>, Status> {
    let mut vec = Vec::new();
    try_reserve(&mut vec, capacity)?;
    Ok(vec)
}

pub(crate) fn try_clone_slice<T: Copy>(slice: &[T]) -> Result<Vec<T>, Status> {
    let mut out = try_vec_with_capacity(slice.len())?;
    out.extend_from_slice(slice);
    Ok(out)
}

fn ceil_div_u32(a: u32, b: u32) -> Result<u32, Status> {
    a.checked_add(b.checked_sub(1).ok_or(Status::InvalidArgument)?)
        .ok_or(Status::InvalidArgument)?
        .checked_div(b)
        .ok_or(Status::InvalidArgument)
}

fn page_count_for_len(seq_len: u32, page_size: u32) -> Result<u32, Status> {
    if seq_len == 0 {
        Ok(0)
    } else {
        ceil_div_u32(seq_len, page_size)
    }
}

fn last_page_len_for_seq(seq_len: u32, page_size: u32) -> i32 {
    if seq_len == 0 {
        0
    } else {
        let rem = seq_len % page_size;
        if rem == 0 {
            page_size as i32
        } else {
            rem as i32
        }
    }
}

fn u32_to_i32(value: u32) -> Result<i32, Status> {
    i32::try_from(value).map_err(|_| Status::InvalidArgument)
}

fn usize_to_u32(value: usize) -> Result<u32, Status> {
    u32::try_from(value).map_err(|_| Status::InvalidArgument)
}

fn usize_to_i32(value: usize) -> Result<i32, Status> {
    i32::try_from(value).map_err(|_| Status::InvalidArgument)
}

fn validate_config(config: &EngineConfig) -> Result<(), Status> {
    if config.num_layers == 0
        || config.max_live_requests == 0
        || config.max_batch_size == 0
        || config.max_seq_len == 0
        || config.max_pages == 0
        || config.page_size == 0
    {
        return Err(Status::InvalidArgument);
    }
    if config.num_q_heads == 0 || config.num_kv_heads == 0 || config.head_dim == 0 {
        return Err(Status::InvalidArgument);
    }
    if !config.num_q_heads.is_multiple_of(config.num_kv_heads) {
        return Err(Status::InvalidArgument);
    }
    if !config.activation_dtype.is_runtime_supported() || !config.kv_dtype.is_runtime_supported() {
        return Err(Status::Unsupported);
    }
    if config.activation_dtype != config.kv_dtype {
        return Err(Status::Unsupported);
    }
    if !matches!(config.kv_layout, KvLayout::NHD | KvLayout::HND) {
        return Err(Status::InvalidArgument);
    }
    let capacity = config
        .max_pages
        .checked_mul(config.page_size)
        .ok_or(Status::InvalidArgument)?;
    if config.max_seq_len > capacity {
        return Err(Status::InvalidArgument);
    }
    if config.qsfi_float_workspace_bytes == 0
        || config.qsfi_int_workspace_bytes == 0
        || config.qsfi_host_int_workspace_bytes == 0
    {
        return Err(Status::InvalidArgument);
    }
    Ok(())
}

impl EngineCore {
    pub fn new(config: EngineConfig) -> Result<Self, Status> {
        validate_config(&config)?;

        let mut free_pages = try_vec_with_capacity(config.max_pages as usize)?;
        for page in (0..config.max_pages).rev() {
            free_pages.push(u32_to_i32(page)?);
        }

        let mut session = Self {
            config,
            requests: Vec::new(),
            free_pages,
            live_request_ids: Vec::new(),
            live_seq_lens: Vec::new(),
            live_token_indptr: Vec::new(),
            live_tokens: Vec::new(),
            live_kv_indptr: Vec::new(),
            live_kv_indices: Vec::new(),
            live_last_page_len: Vec::new(),
            active_batch: None,
            batch_request_ids: Vec::new(),
            batch_tokens: Vec::new(),
            batch_qo_indptr: Vec::new(),
            batch_kv_indptr: Vec::new(),
            batch_kv_indices: Vec::new(),
            batch_last_page_len: Vec::new(),
            batch_rope_pos_offset: Vec::new(),
            batch_append_batch_indices: Vec::new(),
            batch_append_positions: Vec::new(),
        };
        session.rebuild_live_views()?;
        #[cfg(debug_assertions)]
        session.check_allocator_invariants()?;
        Ok(session)
    }

    pub fn config(&self) -> EngineConfig {
        self.config
    }

    pub fn state(&self) -> Result<CoreState<'_>, Status> {
        let active_batch = self.active_batch.as_ref();
        let active_state = active_batch.map(ActiveBatch::state);
        Ok(CoreState {
            batch_kind: active_batch.map(ActiveBatch::kind),
            live_request_count: usize_to_u32(self.requests.len())?,
            batch_size: active_state.map_or(0, |state| state.size),
            batch_token_count: active_state.map_or(0, |state| state.token_count),
            live_num_indices: usize_to_u32(self.live_kv_indices.len())?,
            allocated_pages: self
                .config
                .max_pages
                .checked_sub(usize_to_u32(self.free_pages.len())?)
                .ok_or(Status::InternalError)?,
            free_page_count: usize_to_u32(self.free_pages.len())?,
            max_pages: self.config.max_pages,
            page_size: self.config.page_size,
            live_request_ids: &self.live_request_ids,
            live_seq_lens: &self.live_seq_lens,
            live_token_indptr: &self.live_token_indptr,
            live_tokens: &self.live_tokens,
            live_kv_indptr: &self.live_kv_indptr,
            live_kv_indices: &self.live_kv_indices,
            live_last_page_len: &self.live_last_page_len,
            free_pages: &self.free_pages,
            batch_request_ids: &self.batch_request_ids,
            batch_tokens: &self.batch_tokens,
            batch_qo_indptr: &self.batch_qo_indptr,
            batch_kv_indptr: &self.batch_kv_indptr,
            batch_kv_indices: &self.batch_kv_indices,
            batch_last_page_len: &self.batch_last_page_len,
            batch_rope_pos_offset: &self.batch_rope_pos_offset,
            batch_append_batch_indices: &self.batch_append_batch_indices,
            batch_append_positions: &self.batch_append_positions,
        })
    }

    pub fn reset(&mut self) -> Result<(), Status> {
        let mut next_free_pages = try_vec_with_capacity(self.config.max_pages as usize)?;
        for page in (0..self.config.max_pages).rev() {
            next_free_pages.push(u32_to_i32(page)?);
        }
        let next_requests = Vec::new();
        let live_views = self.build_live_views_for(&next_requests)?;

        self.requests = next_requests;
        self.free_pages = next_free_pages;
        self.clear_active_batch();
        self.install_live_views(live_views);
        #[cfg(debug_assertions)]
        self.check_allocator_invariants()?;
        Ok(())
    }

    pub fn release_requests(&mut self, request_ids: &[RequestId]) -> Result<(), Status> {
        if self.active_batch.is_some() {
            return Err(Status::InvalidArgument);
        }
        let mut seen = HashSet::new();
        seen.try_reserve(request_ids.len())
            .map_err(|_| Status::OutOfMemory)?;
        let mut release_indices = try_vec_with_capacity(request_ids.len())?;
        for id in request_ids {
            if !seen.insert(*id) {
                return Err(Status::InvalidArgument);
            }
            release_indices.push(
                self.find_request_index(*id)
                    .ok_or(Status::InvalidArgument)?,
            );
        }

        let mut release_set = HashSet::new();
        release_set
            .try_reserve(release_indices.len())
            .map_err(|_| Status::OutOfMemory)?;
        for idx in &release_indices {
            release_set.insert(*idx);
        }

        let mut next_requests =
            try_vec_with_capacity(self.requests.len().saturating_sub(release_set.len()))?;
        let mut next_free_pages = try_clone_slice(&self.free_pages)?;
        let released_page_count = release_indices
            .iter()
            .try_fold(0usize, |acc, idx| {
                acc.checked_add(self.requests[*idx].pages.len())
            })
            .ok_or(Status::InvalidArgument)?;
        try_reserve(&mut next_free_pages, released_page_count)?;

        for (idx, req) in self.requests.iter().enumerate() {
            if release_set.contains(&idx) {
                next_free_pages.extend_from_slice(&req.pages);
            } else {
                next_requests.push(Request {
                    id: req.id,
                    seq_len: req.seq_len,
                    tokens: try_clone_slice(&req.tokens)?,
                    pages: try_clone_slice(&req.pages)?,
                });
            }
        }

        let live_views = self.build_live_views_for(&next_requests)?;
        self.requests = next_requests;
        self.free_pages = next_free_pages;
        self.install_live_views(live_views);
        #[cfg(debug_assertions)]
        self.check_allocator_invariants()?;
        Ok(())
    }

    pub fn begin_append(
        &mut self,
        request_ids: &[RequestId],
        token_indptr: &[i32],
        tokens: &[i32],
    ) -> Result<(), Status> {
        if self.active_batch.is_some() {
            return Err(Status::InvalidArgument);
        }
        let batch_size = usize_to_u32(request_ids.len())?;
        let token_count = usize_to_u32(tokens.len())?;
        if batch_size == 0 || batch_size > self.config.max_batch_size || token_count == 0 {
            return Err(Status::InvalidArgument);
        }
        if token_indptr.len() != request_ids.len() + 1 {
            return Err(Status::InvalidArgument);
        }
        if token_indptr[0] != 0 {
            return Err(Status::InvalidArgument);
        }

        let mut seen = HashSet::new();
        seen.try_reserve(request_ids.len())
            .map_err(|_| Status::OutOfMemory)?;
        for i in 0..request_ids.len() {
            if token_indptr[i] < 0 || token_indptr[i + 1] < token_indptr[i] {
                return Err(Status::InvalidArgument);
            }
            if !seen.insert(request_ids[i]) {
                return Err(Status::InvalidArgument);
            }
        }
        if token_indptr[request_ids.len()] != usize_to_i32(tokens.len())? {
            return Err(Status::InvalidArgument);
        }

        let mut staged_rows = try_vec_with_capacity(request_ids.len())?;
        let mut new_request_count = 0usize;
        let mut extra_pages = 0usize;

        for i in 0..request_ids.len() {
            let token_begin =
                u32::try_from(token_indptr[i]).map_err(|_| Status::InvalidArgument)?;
            let token_end =
                u32::try_from(token_indptr[i + 1]).map_err(|_| Status::InvalidArgument)?;
            let token_begin_usize =
                usize::try_from(token_begin).map_err(|_| Status::InvalidArgument)?;
            let token_end_usize =
                usize::try_from(token_end).map_err(|_| Status::InvalidArgument)?;
            let row_token_count = token_end
                .checked_sub(token_begin)
                .ok_or(Status::InvalidArgument)?;
            if row_token_count == 0 {
                return Err(Status::InvalidArgument);
            }
            let request_index = self.find_request_index(request_ids[i]);
            let old_seq_len = request_index.map_or(0, |idx| self.requests[idx].seq_len);
            let new_seq_len = old_seq_len
                .checked_add(row_token_count)
                .ok_or(Status::InvalidArgument)?;
            if new_seq_len > self.config.max_seq_len {
                return Err(Status::InvalidArgument);
            }
            if request_index.is_none() && row_token_count != 0 {
                new_request_count = new_request_count
                    .checked_add(1)
                    .ok_or(Status::InvalidArgument)?;
            }

            let old_pages: &[i32] =
                request_index.map_or(&[], |idx| self.requests[idx].pages.as_slice());
            let old_page_count = old_pages.len();
            let needed_pages = page_count_for_len(new_seq_len, self.config.page_size)? as usize;
            extra_pages = extra_pages
                .checked_add(
                    needed_pages
                        .checked_sub(old_page_count)
                        .ok_or(Status::InternalError)?,
                )
                .ok_or(Status::InvalidArgument)?;

            staged_rows.push(StagedRow {
                id: request_ids[i],
                request_index,
                old_seq_len,
                old_page_count,
                token_count: row_token_count,
                tokens: try_clone_slice(&tokens[token_begin_usize..token_end_usize])?,
                pages: try_clone_slice(old_pages)?,
            });
        }

        if self.requests.len() + new_request_count > self.config.max_live_requests as usize {
            return Err(Status::InvalidArgument);
        }
        if extra_pages > self.free_pages.len() {
            return Err(Status::OutOfMemory);
        }

        let mut next_free_pages = try_clone_slice(&self.free_pages)?;
        let batch_request_ids = try_clone_slice(request_ids)?;
        let batch_tokens = try_clone_slice(tokens)?;
        let batch_qo_indptr = try_clone_slice(token_indptr)?;
        let mut batch_kv_indptr = try_vec_with_capacity(request_ids.len() + 1)?;
        let mut batch_kv_indices =
            try_vec_with_capacity(staged_rows.iter().try_fold(0usize, |acc, row| {
                let new_seq_len = row
                    .old_seq_len
                    .checked_add(row.token_count)
                    .ok_or(Status::InvalidArgument)?;
                acc.checked_add(page_count_for_len(new_seq_len, self.config.page_size)? as usize)
                    .ok_or(Status::InvalidArgument)
            })?)?;
        let mut batch_last_page_len = try_vec_with_capacity(request_ids.len())?;
        let mut batch_rope_pos_offset = try_vec_with_capacity(request_ids.len())?;
        let mut batch_append_batch_indices = try_vec_with_capacity(tokens.len())?;
        let mut batch_append_positions = try_vec_with_capacity(tokens.len())?;
        batch_kv_indptr.push(0);

        for (idx, row) in staged_rows.iter_mut().enumerate() {
            let new_seq_len = row
                .old_seq_len
                .checked_add(row.token_count)
                .ok_or(Status::InvalidArgument)?;
            let needed_pages = page_count_for_len(new_seq_len, self.config.page_size)? as usize;
            while row.pages.len() < needed_pages {
                let page = next_free_pages.pop().ok_or(Status::OutOfMemory)?;
                row.pages.push(page);
            }

            batch_kv_indices.extend_from_slice(&row.pages);
            batch_kv_indptr.push(usize_to_i32(batch_kv_indices.len())?);
            batch_last_page_len.push(last_page_len_for_seq(new_seq_len, self.config.page_size));
            batch_rope_pos_offset.push(0);

            for j in 0..row.token_count {
                batch_append_batch_indices.push(usize_to_i32(idx)?);
                let pos = row
                    .old_seq_len
                    .checked_add(j)
                    .ok_or(Status::InvalidArgument)?;
                batch_append_positions.push(u32_to_i32(pos)?);
            }
        }

        self.free_pages = next_free_pages;
        self.batch_request_ids = batch_request_ids;
        self.batch_tokens = batch_tokens;
        self.batch_qo_indptr = batch_qo_indptr;
        self.batch_kv_indptr = batch_kv_indptr;
        self.batch_kv_indices = batch_kv_indices;
        self.batch_last_page_len = batch_last_page_len;
        self.batch_rope_pos_offset = batch_rope_pos_offset;
        self.batch_append_batch_indices = batch_append_batch_indices;
        self.batch_append_positions = batch_append_positions;
        self.active_batch = Some(ActiveBatch::append(
            batch_size,
            token_count,
            staged_rows,
            self.config.num_layers,
        ));
        #[cfg(debug_assertions)]
        self.check_allocator_invariants()?;
        Ok(())
    }

    pub fn begin_decode(
        &mut self,
        request_ids: &[RequestId],
        tokens: &[i32],
    ) -> Result<(), Status> {
        if self.active_batch.is_some() {
            return Err(Status::InvalidArgument);
        }
        if request_ids.is_empty()
            || request_ids.len() != tokens.len()
            || request_ids.len() > self.config.max_batch_size as usize
        {
            return Err(Status::InvalidArgument);
        }

        let mut seen = HashSet::new();
        seen.try_reserve(request_ids.len())
            .map_err(|_| Status::OutOfMemory)?;
        for id in request_ids {
            if !seen.insert(*id) {
                return Err(Status::InvalidArgument);
            }
        }

        let mut staged_rows = try_vec_with_capacity(request_ids.len())?;
        let mut extra_pages = 0usize;

        for (row_idx, id) in request_ids.iter().enumerate() {
            let request_index = self
                .find_request_index(*id)
                .ok_or(Status::InvalidArgument)?;
            let req = &self.requests[request_index];
            let new_seq_len = req.seq_len.checked_add(1).ok_or(Status::InvalidArgument)?;
            if new_seq_len > self.config.max_seq_len {
                return Err(Status::InvalidArgument);
            }
            let needed_pages = page_count_for_len(new_seq_len, self.config.page_size)? as usize;
            extra_pages = extra_pages
                .checked_add(
                    needed_pages
                        .checked_sub(req.pages.len())
                        .ok_or(Status::InternalError)?,
                )
                .ok_or(Status::InvalidArgument)?;
            staged_rows.push(StagedRow {
                id: *id,
                request_index: Some(request_index),
                old_seq_len: req.seq_len,
                old_page_count: req.pages.len(),
                token_count: 1,
                tokens: {
                    let mut row_tokens = try_vec_with_capacity(1)?;
                    row_tokens.push(tokens[row_idx]);
                    row_tokens
                },
                pages: try_clone_slice(&req.pages)?,
            });
        }

        if extra_pages > self.free_pages.len() {
            return Err(Status::OutOfMemory);
        }

        let mut next_free_pages = try_clone_slice(&self.free_pages)?;
        let batch_request_ids = try_clone_slice(request_ids)?;
        let batch_tokens = try_clone_slice(tokens)?;
        let mut batch_kv_indptr = try_vec_with_capacity(request_ids.len() + 1)?;
        let mut batch_kv_indices =
            try_vec_with_capacity(staged_rows.iter().try_fold(0usize, |acc, row| {
                let new_seq_len = row
                    .old_seq_len
                    .checked_add(1)
                    .ok_or(Status::InvalidArgument)?;
                acc.checked_add(page_count_for_len(new_seq_len, self.config.page_size)? as usize)
                    .ok_or(Status::InvalidArgument)
            })?)?;
        let mut batch_last_page_len = try_vec_with_capacity(request_ids.len())?;
        let mut batch_rope_pos_offset = try_vec_with_capacity(request_ids.len())?;
        batch_kv_indptr.push(0);

        for row in staged_rows.iter_mut() {
            let new_seq_len = row
                .old_seq_len
                .checked_add(1)
                .ok_or(Status::InvalidArgument)?;
            let needed_pages = page_count_for_len(new_seq_len, self.config.page_size)? as usize;
            while row.pages.len() < needed_pages {
                let page = next_free_pages.pop().ok_or(Status::OutOfMemory)?;
                row.pages.push(page);
            }
            batch_kv_indices.extend_from_slice(&row.pages);
            batch_kv_indptr.push(usize_to_i32(batch_kv_indices.len())?);
            batch_last_page_len.push(last_page_len_for_seq(new_seq_len, self.config.page_size));
            batch_rope_pos_offset.push(0);
        }

        self.free_pages = next_free_pages;
        self.batch_request_ids = batch_request_ids;
        self.batch_tokens = batch_tokens;
        self.batch_qo_indptr.clear();
        self.batch_kv_indptr = batch_kv_indptr;
        self.batch_kv_indices = batch_kv_indices;
        self.batch_last_page_len = batch_last_page_len;
        self.batch_rope_pos_offset = batch_rope_pos_offset;
        self.batch_append_batch_indices.clear();
        self.batch_append_positions.clear();
        self.active_batch = Some(ActiveBatch::decode(
            usize_to_u32(request_ids.len())?,
            usize_to_u32(request_ids.len())?,
            staged_rows,
            self.config.num_layers,
        ));
        #[cfg(debug_assertions)]
        self.check_allocator_invariants()?;
        Ok(())
    }

    pub fn commit_batch(&mut self, accepted_token_counts: Option<&[u32]>) -> Result<(), Status> {
        let active_batch = self.active_batch.as_ref().ok_or(Status::InvalidArgument)?;
        let active_state = active_batch.state();
        if !active_state.layers.is_complete() {
            return Err(Status::InvalidArgument);
        }
        if let Some(counts) = accepted_token_counts
            && counts.len() != active_state.rows.len()
        {
            return Err(Status::InvalidArgument);
        }

        let mut accepted = try_vec_with_capacity(active_state.rows.len())?;
        for (i, row) in active_state.rows.iter().enumerate() {
            let count = accepted_token_counts.map_or(row.token_count, |counts| counts[i]);
            if count > row.token_count {
                return Err(Status::InvalidArgument);
            }
            if active_batch.is_decode() && count > 1 {
                return Err(Status::InvalidArgument);
            }
            accepted.push(count);
        }

        let mut next_requests = self.clone_requests()?;
        let mut next_free_pages = try_clone_slice(&self.free_pages)?;

        for (i, row) in active_state.rows.iter().enumerate() {
            let new_seq_len = row
                .old_seq_len
                .checked_add(accepted[i])
                .ok_or(Status::InvalidArgument)?;
            let accepted_len = usize::try_from(accepted[i]).map_err(|_| Status::InvalidArgument)?;
            let needed_pages = page_count_for_len(new_seq_len, self.config.page_size)? as usize;
            if needed_pages > row.pages.len() {
                return Err(Status::InternalError);
            }
            if accepted_len > row.tokens.len() {
                return Err(Status::InternalError);
            }

            match row.request_index {
                Some(request_index) => {
                    let req = next_requests
                        .get_mut(request_index)
                        .ok_or(Status::InternalError)?;
                    req.seq_len = new_seq_len;
                    let old_token_len =
                        usize::try_from(row.old_seq_len).map_err(|_| Status::InvalidArgument)?;
                    if req.tokens.len() != old_token_len {
                        return Err(Status::InternalError);
                    }
                    try_reserve(&mut req.tokens, accepted_len)?;
                    req.tokens.extend_from_slice(&row.tokens[..accepted_len]);
                    req.pages = try_clone_slice(&row.pages[..needed_pages])?;
                }
                None => {
                    if accepted[i] != 0 {
                        next_requests.push(Request {
                            id: row.id,
                            seq_len: new_seq_len,
                            tokens: try_clone_slice(&row.tokens[..accepted_len])?,
                            pages: try_clone_slice(&row.pages[..needed_pages])?,
                        });
                    }
                }
            }
            try_reserve(&mut next_free_pages, row.pages.len() - needed_pages)?;
            next_free_pages.extend_from_slice(&row.pages[needed_pages..]);
        }

        let live_views = self.build_live_views_for(&next_requests)?;
        self.requests = next_requests;
        self.free_pages = next_free_pages;
        self.clear_active_batch();
        self.install_live_views(live_views);
        #[cfg(debug_assertions)]
        self.check_allocator_invariants()?;
        Ok(())
    }

    pub fn abort_batch(&mut self) -> Result<(), Status> {
        let Some(active_batch) = self.active_batch.as_ref() else {
            return Ok(());
        };
        let mut next_free_pages = try_clone_slice(&self.free_pages)?;
        Self::return_staged_pages(active_batch, &mut next_free_pages)?;
        let live_views = self.build_live_views_for(&self.requests)?;

        self.free_pages = next_free_pages;
        self.clear_active_batch();
        self.install_live_views(live_views);
        #[cfg(debug_assertions)]
        self.check_allocator_invariants()?;
        Ok(())
    }

    pub fn batch_size(&self) -> u32 {
        self.active_batch
            .as_ref()
            .map_or(0, |batch| batch.state().size)
    }

    pub fn batch_token_count(&self) -> u32 {
        self.active_batch
            .as_ref()
            .map_or(0, |batch| batch.state().token_count)
    }

    pub fn batch_tokens(&self) -> &[i32] {
        &self.batch_tokens
    }

    pub fn batch_qo_indptr(&self) -> &[i32] {
        &self.batch_qo_indptr
    }

    pub fn batch_kv_indptr(&self) -> &[i32] {
        &self.batch_kv_indptr
    }

    pub fn batch_kv_indices(&self) -> &[i32] {
        &self.batch_kv_indices
    }

    pub fn batch_last_page_len(&self) -> &[i32] {
        &self.batch_last_page_len
    }

    pub fn batch_rope_pos_offset(&self) -> &[i32] {
        &self.batch_rope_pos_offset
    }

    pub fn batch_append_batch_indices(&self) -> &[i32] {
        &self.batch_append_batch_indices
    }

    pub fn batch_append_positions(&self) -> &[i32] {
        &self.batch_append_positions
    }

    pub(crate) fn pending_append_layer(
        &self,
        layer_idx: u32,
    ) -> Result<PendingAppendLayer, Status> {
        match self.active_batch.as_ref() {
            Some(ActiveBatch::Append(batch)) => {
                batch.layers.pending(layer_idx).map(PendingAppendLayer)
            }
            _ => Err(Status::InvalidArgument),
        }
    }

    pub(crate) fn complete_append_layer(
        &mut self,
        pending: PendingAppendLayer,
    ) -> Result<(), Status> {
        match self.active_batch.as_mut() {
            Some(ActiveBatch::Append(batch)) => batch.layers.complete(pending.0),
            _ => Err(Status::InternalError),
        }
    }

    pub(crate) fn pending_decode_layer(
        &self,
        layer_idx: u32,
    ) -> Result<PendingDecodeLayer, Status> {
        match self.active_batch.as_ref() {
            Some(ActiveBatch::Decode(batch)) => {
                batch.layers.pending(layer_idx).map(PendingDecodeLayer)
            }
            _ => Err(Status::InvalidArgument),
        }
    }

    pub(crate) fn complete_decode_layer(
        &mut self,
        pending: PendingDecodeLayer,
    ) -> Result<(), Status> {
        match self.active_batch.as_mut() {
            Some(ActiveBatch::Decode(batch)) => batch.layers.complete(pending.0),
            _ => Err(Status::InternalError),
        }
    }

    fn clone_requests(&self) -> Result<Vec<Request>, Status> {
        let mut out = try_vec_with_capacity(self.requests.len())?;
        for req in &self.requests {
            out.push(Request {
                id: req.id,
                seq_len: req.seq_len,
                tokens: try_clone_slice(&req.tokens)?,
                pages: try_clone_slice(&req.pages)?,
            });
        }
        Ok(out)
    }

    fn find_request_index(&self, id: RequestId) -> Option<usize> {
        self.requests.iter().position(|req| req.id == id)
    }

    fn return_staged_pages(
        active_batch: &ActiveBatch,
        free_pages: &mut Vec<i32>,
    ) -> Result<(), Status> {
        let rows = &active_batch.state().rows;
        let return_count = rows
            .iter()
            .map(|row| row.pages.len().saturating_sub(row.old_page_count))
            .sum();
        try_reserve(free_pages, return_count)?;
        for row in rows {
            free_pages.extend_from_slice(&row.pages[row.old_page_count..]);
        }
        Ok(())
    }

    fn clear_batch_views(&mut self) {
        self.batch_request_ids.clear();
        self.batch_tokens.clear();
        self.batch_qo_indptr.clear();
        self.batch_kv_indptr.clear();
        self.batch_kv_indices.clear();
        self.batch_last_page_len.clear();
        self.batch_rope_pos_offset.clear();
        self.batch_append_batch_indices.clear();
        self.batch_append_positions.clear();
    }

    fn clear_active_batch(&mut self) {
        self.active_batch = None;
        self.clear_batch_views();
    }

    fn rebuild_live_views(&mut self) -> Result<(), Status> {
        let live_views = self.build_live_views_for(&self.requests)?;
        self.install_live_views(live_views);
        Ok(())
    }

    fn build_live_views_for(&self, requests: &[Request]) -> Result<LiveViews, Status> {
        let total_pages = requests
            .iter()
            .try_fold(0usize, |acc, req| acc.checked_add(req.pages.len()))
            .ok_or(Status::InvalidArgument)?;
        let total_tokens = requests
            .iter()
            .try_fold(0usize, |acc, req| acc.checked_add(req.tokens.len()))
            .ok_or(Status::InvalidArgument)?;
        let mut live_request_ids = try_vec_with_capacity(requests.len())?;
        let mut live_seq_lens = try_vec_with_capacity(requests.len())?;
        let mut live_token_indptr = try_vec_with_capacity(requests.len() + 1)?;
        let mut live_tokens = try_vec_with_capacity(total_tokens)?;
        let mut live_kv_indptr = try_vec_with_capacity(requests.len() + 1)?;
        let mut live_kv_indices = try_vec_with_capacity(total_pages)?;
        let mut live_last_page_len = try_vec_with_capacity(requests.len())?;

        live_token_indptr.push(0);
        live_kv_indptr.push(0);
        for req in requests {
            let seq_len = usize::try_from(req.seq_len).map_err(|_| Status::InvalidArgument)?;
            if req.tokens.len() != seq_len {
                return Err(Status::InternalError);
            }
            live_request_ids.push(req.id);
            live_seq_lens.push(u32_to_i32(req.seq_len)?);
            live_tokens.extend_from_slice(&req.tokens);
            live_token_indptr.push(usize_to_i32(live_tokens.len())?);
            live_kv_indices.extend_from_slice(&req.pages);
            live_kv_indptr.push(usize_to_i32(live_kv_indices.len())?);
            live_last_page_len.push(last_page_len_for_seq(req.seq_len, self.config.page_size));
        }

        Ok(LiveViews {
            request_ids: live_request_ids,
            seq_lens: live_seq_lens,
            token_indptr: live_token_indptr,
            tokens: live_tokens,
            kv_indptr: live_kv_indptr,
            kv_indices: live_kv_indices,
            last_page_len: live_last_page_len,
        })
    }

    fn install_live_views(&mut self, live_views: LiveViews) {
        self.live_request_ids = live_views.request_ids;
        self.live_seq_lens = live_views.seq_lens;
        self.live_token_indptr = live_views.token_indptr;
        self.live_tokens = live_views.tokens;
        self.live_kv_indptr = live_views.kv_indptr;
        self.live_kv_indices = live_views.kv_indices;
        self.live_last_page_len = live_views.last_page_len;
    }

    #[cfg(debug_assertions)]
    fn check_allocator_invariants(&self) -> Result<(), Status> {
        let max_pages =
            usize::try_from(self.config.max_pages).map_err(|_| Status::InvalidArgument)?;
        let mut owners: Vec<Option<&'static str>> = vec![None; max_pages];

        for &page in &self.free_pages {
            Self::claim_page(&mut owners, page, "free")?;
        }

        for req in &self.requests {
            for &page in &req.pages {
                Self::claim_page(&mut owners, page, "live")?;
            }
        }

        if let Some(active_batch) = &self.active_batch {
            for row in &active_batch.state().rows {
                if row.old_page_count > row.pages.len() {
                    return Err(Status::InternalError);
                }
                if let Some(request_index) = row.request_index {
                    let req = self
                        .requests
                        .get(request_index)
                        .ok_or(Status::InternalError)?;
                    if req.id != row.id || row.old_page_count != req.pages.len() {
                        return Err(Status::InternalError);
                    }
                    if row.pages[..row.old_page_count] != req.pages[..] {
                        return Err(Status::InternalError);
                    }
                } else if row.old_page_count != 0 {
                    return Err(Status::InternalError);
                }
                for &page in &row.pages[row.old_page_count..] {
                    Self::claim_page(&mut owners, page, "staged")?;
                }
            }
        }

        if owners.iter().any(Option::is_none) {
            return Err(Status::InternalError);
        }
        Ok(())
    }

    #[cfg(debug_assertions)]
    fn claim_page(
        owners: &mut [Option<&'static str>],
        page: i32,
        owner: &'static str,
    ) -> Result<(), Status> {
        let page = usize::try_from(page).map_err(|_| Status::InternalError)?;
        let slot = owners.get_mut(page).ok_or(Status::InternalError)?;
        if slot.replace(owner).is_some() {
            return Err(Status::InternalError);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny_config() -> EngineConfig {
        EngineConfig {
            device_ordinal: 0,
            stream: std::ptr::null_mut(),
            num_layers: 1,
            max_live_requests: 4,
            max_batch_size: 3,
            max_seq_len: 8,
            max_pages: 8,
            page_size: 4,
            hidden_size: 128,
            intermediate_size: 0,
            vocab_size: 0,
            num_q_heads: 2,
            num_kv_heads: 2,
            head_dim: 64,
            activation_dtype: DType::F16,
            kv_dtype: DType::F16,
            kv_layout: KvLayout::NHD,
            rope_theta: 10000.0,
            rope_scale: 1.0,
            logits_soft_cap: 0.0,
            qsfi_float_workspace_bytes: 64 << 20,
            qsfi_int_workspace_bytes: 64 << 20,
            qsfi_host_int_workspace_bytes: 64 << 20,
        }
    }

    fn complete_all_layers(session: &mut EngineCore, batch_kind: BatchKind) {
        let num_layers = session.config().num_layers;
        for layer_idx in 0..num_layers {
            complete_layer(session, batch_kind, layer_idx).unwrap();
        }
    }

    fn complete_layer(
        session: &mut EngineCore,
        batch_kind: BatchKind,
        layer_idx: u32,
    ) -> Result<(), Status> {
        match batch_kind {
            BatchKind::Append => {
                let pending = session.pending_append_layer(layer_idx)?;
                session.complete_append_layer(pending)
            }
            BatchKind::Decode => {
                let pending = session.pending_decode_layer(layer_idx)?;
                session.complete_decode_layer(pending)
            }
        }
    }

    fn tiny_config_with_layers(num_layers: u32) -> EngineConfig {
        let mut config = tiny_config();
        config.num_layers = num_layers;
        config
    }

    #[test]
    fn dtype_bits_and_storage_bytes_are_total() {
        assert_eq!(DType::F32.bits(), 32);
        assert_eq!(DType::F16.bits(), 16);
        assert_eq!(DType::BF16.bits(), 16);
        assert_eq!(DType::FP8E4M3.bits(), 8);
        assert_eq!(DType::FP8E5M2.bits(), 8);
        assert_eq!(DType::NVFP4E2M1.bits(), 4);
        assert_eq!(DType::MXFP4E2M1.bits(), 4);
        assert_eq!(DType::MXFP8E4M3.bits(), 8);
        assert_eq!(DType::I32.bits(), 32);
        assert_eq!(DType::U32.bits(), 32);
        assert_eq!(DType::I8.bits(), 8);
        assert_eq!(DType::U8.bits(), 8);
        assert_eq!(DType::F16.storage_bytes_for(3), Ok(6));
        assert_eq!(DType::FP8E4M3.storage_bytes_for(3), Ok(3));
        assert_eq!(DType::NVFP4E2M1.storage_bytes_for(1), Ok(1));
        assert_eq!(DType::NVFP4E2M1.storage_bytes_for(2), Ok(1));
        assert_eq!(DType::NVFP4E2M1.storage_bytes_for(3), Ok(2));
    }

    #[test]
    fn dtype_and_layout_raw_values_match_qsfi() {
        assert_eq!(DType::F32.to_raw(), ffi::DTYPE_F32);
        assert_eq!(DType::F16.to_raw(), ffi::DTYPE_F16);
        assert_eq!(DType::BF16.to_raw(), ffi::DTYPE_BF16);
        assert_eq!(DType::FP8E4M3.to_raw(), ffi::DTYPE_FP8_E4M3);
        assert_eq!(DType::FP8E5M2.to_raw(), ffi::DTYPE_FP8_E5M2);
        assert_eq!(DType::NVFP4E2M1.to_raw(), ffi::DTYPE_NVFP4_E2M1);
        assert_eq!(DType::MXFP4E2M1.to_raw(), ffi::DTYPE_MXFP4_E2M1);
        assert_eq!(DType::MXFP8E4M3.to_raw(), ffi::DTYPE_MXFP8_E4M3);
        assert_eq!(DType::I32.to_raw(), ffi::DTYPE_I32);
        assert_eq!(DType::U32.to_raw(), ffi::DTYPE_U32);
        assert_eq!(DType::I8.to_raw(), ffi::DTYPE_I8);
        assert_eq!(DType::U8.to_raw(), ffi::DTYPE_U8);
        assert_eq!(KvLayout::NHD.to_raw(), ffi::KV_LAYOUT_NHD);
        assert_eq!(KvLayout::HND.to_raw(), ffi::KV_LAYOUT_HND);
    }

    #[test]
    fn append_commit_decode_release() {
        let mut session = EngineCore::new(tiny_config()).unwrap();
        session
            .begin_append(&[10, 11], &[0, 5, 6], &[100, 101, 102, 103, 104, 200])
            .unwrap();
        let state = session.state().unwrap();
        assert_eq!(state.batch_kind, Some(BatchKind::Append));
        assert_eq!(state.batch_size, 2);
        assert_eq!(state.batch_token_count, 6);
        assert_eq!(state.batch_qo_indptr, &[0, 5, 6]);
        assert_eq!(state.batch_kv_indptr, &[0, 2, 3]);
        assert_eq!(state.batch_last_page_len, &[1, 1]);
        assert_eq!(state.batch_append_batch_indices[4], 0);
        assert_eq!(state.batch_append_batch_indices[5], 1);
        assert_eq!(state.batch_append_positions[0], 0);
        assert_eq!(state.batch_append_positions[4], 4);
        assert_eq!(state.batch_append_positions[5], 0);
        assert_eq!(state.free_page_count, 5);

        complete_all_layers(&mut session, BatchKind::Append);
        session.commit_batch(None).unwrap();
        let state = session.state().unwrap();
        assert_eq!(state.batch_kind, None);
        assert_eq!(state.live_request_ids, &[10, 11]);
        assert_eq!(state.live_seq_lens, &[5, 1]);
        assert_eq!(state.live_token_indptr, &[0, 5, 6]);
        assert_eq!(state.live_tokens, &[100, 101, 102, 103, 104, 200]);
        assert_eq!(state.live_kv_indptr, &[0, 2, 3]);
        assert_eq!(state.free_page_count, 5);

        session
            .begin_append(&[10, 12], &[0, 2, 6], &[105, 106, 300, 301, 302, 303])
            .unwrap();
        complete_all_layers(&mut session, BatchKind::Append);
        session.commit_batch(Some(&[1, 0])).unwrap();
        let state = session.state().unwrap();
        assert_eq!(state.live_request_ids, &[10, 11]);
        assert_eq!(state.live_seq_lens, &[6, 1]);
        assert_eq!(state.live_token_indptr, &[0, 6, 7]);
        assert_eq!(state.live_tokens, &[100, 101, 102, 103, 104, 105, 200]);
        assert_eq!(state.free_page_count, 5);

        session.begin_decode(&[10, 11], &[107, 201]).unwrap();
        complete_all_layers(&mut session, BatchKind::Decode);
        session.commit_batch(Some(&[0, 1])).unwrap();
        let state = session.state().unwrap();
        assert_eq!(state.live_seq_lens, &[6, 2]);
        assert_eq!(state.live_token_indptr, &[0, 6, 8]);
        assert_eq!(state.live_tokens, &[100, 101, 102, 103, 104, 105, 200, 201]);
        assert_eq!(state.free_page_count, 5);

        session.release_requests(&[10]).unwrap();
        let state = session.state().unwrap();
        assert_eq!(state.live_request_ids, &[11]);
        assert_eq!(state.live_seq_lens, &[2]);
        assert_eq!(state.live_token_indptr, &[0, 2]);
        assert_eq!(state.live_tokens, &[200, 201]);
        assert_eq!(state.free_page_count, 7);
    }

    #[test]
    fn zero_attention_layers_are_invalid() {
        let mut config = tiny_config_with_layers(0);
        assert_eq!(
            EngineCore::new(config).map(|_| ()),
            Err(Status::InvalidArgument)
        );

        config.num_q_heads = 0;
        config.num_kv_heads = 0;
        config.head_dim = 0;
        assert_eq!(
            EngineCore::new(config).map(|_| ()),
            Err(Status::InvalidArgument)
        );
    }

    #[test]
    fn rejects_bad_append_shapes_without_allocating_pages() {
        let mut session = EngineCore::new(tiny_config()).unwrap();
        assert_eq!(
            session.begin_append(&[41, 42], &[1, 2, 3], &[1, 2, 3]),
            Err(Status::InvalidArgument)
        );
        assert_eq!(
            session.begin_append(&[41, 42], &[0, 2, 1], &[1]),
            Err(Status::InvalidArgument)
        );
        assert_eq!(
            session.begin_append(&[41, 42], &[0, 1, 2], &[1, 2, 3]),
            Err(Status::InvalidArgument)
        );
        assert_eq!(
            session.begin_append(&[43, 43], &[0, 1, 2], &[1, 2]),
            Err(Status::InvalidArgument)
        );
        assert_eq!(
            session.begin_append(&[44, 45], &[0, 1, 1], &[1]),
            Err(Status::InvalidArgument)
        );
        let state = session.state().unwrap();
        assert_eq!(state.batch_kind, None);
        assert_eq!(state.live_request_count, 0);
        assert_eq!(state.free_page_count, 8);
    }

    #[test]
    fn release_rejects_unknown_and_duplicate_ids_without_mutation() {
        let mut session = EngineCore::new(tiny_config()).unwrap();
        session
            .begin_append(&[71, 72], &[0, 1, 2], &[10, 20])
            .unwrap();
        complete_all_layers(&mut session, BatchKind::Append);
        session.commit_batch(None).unwrap();

        assert_eq!(
            session.release_requests(&[71, 73]),
            Err(Status::InvalidArgument)
        );
        let state = session.state().unwrap();
        assert_eq!(state.live_request_ids, &[71, 72]);
        assert_eq!(state.live_tokens, &[10, 20]);
        assert_eq!(state.free_page_count, 6);

        assert_eq!(
            session.release_requests(&[71, 71]),
            Err(Status::InvalidArgument)
        );
        let state = session.state().unwrap();
        assert_eq!(state.live_request_ids, &[71, 72]);
        assert_eq!(state.live_tokens, &[10, 20]);
        assert_eq!(state.free_page_count, 6);
    }

    #[test]
    fn abort_and_reset_restore_pages() {
        let mut session = EngineCore::new(tiny_config()).unwrap();
        session.begin_append(&[31], &[0, 4], &[1, 2, 3, 4]).unwrap();
        complete_layer(&mut session, BatchKind::Append, 0).unwrap();
        assert_eq!(
            session.release_requests(&[31]),
            Err(Status::InvalidArgument)
        );
        assert_eq!(
            session.begin_decode(&[31], &[5]),
            Err(Status::InvalidArgument)
        );
        session.abort_batch().unwrap();
        let state = session.state().unwrap();
        assert_eq!(state.batch_kind, None);
        assert_eq!(state.live_request_count, 0);
        assert_eq!(state.free_page_count, 8);

        session.begin_append(&[31], &[0, 4], &[1, 2, 3, 4]).unwrap();
        complete_layer(&mut session, BatchKind::Append, 0).unwrap();
        session.reset().unwrap();
        let state = session.state().unwrap();
        assert_eq!(state.batch_kind, None);
        assert_eq!(state.live_request_count, 0);
        assert_eq!(state.free_page_count, 8);
    }

    #[test]
    fn decode_validation_and_invalid_commit_keep_batch_active() {
        let mut session = EngineCore::new(tiny_config()).unwrap();
        session.begin_append(&[51], &[0, 1], &[1]).unwrap();
        complete_all_layers(&mut session, BatchKind::Append);
        session.commit_batch(None).unwrap();

        assert_eq!(
            session.begin_decode(&[52], &[9]),
            Err(Status::InvalidArgument)
        );
        assert_eq!(
            session.begin_decode(&[51, 51], &[2, 3]),
            Err(Status::InvalidArgument)
        );

        session.begin_decode(&[51], &[4]).unwrap();
        complete_all_layers(&mut session, BatchKind::Decode);
        assert_eq!(
            session.commit_batch(Some(&[2])),
            Err(Status::InvalidArgument)
        );
        let state = session.state().unwrap();
        assert_eq!(state.batch_kind, Some(BatchKind::Decode));
        assert_eq!(state.live_seq_lens, &[1]);
        session.abort_batch().unwrap();
        assert_eq!(session.state().unwrap().live_seq_lens, &[1]);
    }

    #[test]
    fn commit_rejects_batch_without_layer_execution() {
        let mut session = EngineCore::new(tiny_config()).unwrap();
        session.begin_append(&[61], &[0, 2], &[1, 2]).unwrap();

        assert_eq!(session.commit_batch(None), Err(Status::InvalidArgument));
        let state = session.state().unwrap();
        assert_eq!(state.batch_kind, Some(BatchKind::Append));
        assert_eq!(state.live_request_count, 0);

        complete_layer(&mut session, BatchKind::Append, 0).unwrap();
        session.commit_batch(None).unwrap();
        assert_eq!(session.state().unwrap().live_seq_lens, &[2]);
    }

    #[test]
    fn layer_completion_rejects_wrong_duplicate_and_extra_layers() {
        let mut session = EngineCore::new(tiny_config_with_layers(2)).unwrap();
        session.begin_append(&[62], &[0, 1], &[1]).unwrap();

        assert_eq!(
            complete_layer(&mut session, BatchKind::Append, 1),
            Err(Status::InvalidArgument)
        );
        assert_eq!(
            complete_layer(&mut session, BatchKind::Append, 2),
            Err(Status::InvalidArgument)
        );
        complete_layer(&mut session, BatchKind::Append, 0).unwrap();
        assert_eq!(
            complete_layer(&mut session, BatchKind::Append, 0),
            Err(Status::InvalidArgument)
        );
        complete_layer(&mut session, BatchKind::Append, 1).unwrap();
        assert_eq!(
            complete_layer(&mut session, BatchKind::Append, 1),
            Err(Status::InvalidArgument)
        );

        session.commit_batch(None).unwrap();
        assert_eq!(session.state().unwrap().live_seq_lens, &[1]);
    }

    #[test]
    fn commit_rejects_missing_layer_on_multi_layer_batch() {
        let mut session = EngineCore::new(tiny_config_with_layers(2)).unwrap();
        session.begin_append(&[63], &[0, 1], &[1]).unwrap();
        complete_layer(&mut session, BatchKind::Append, 0).unwrap();

        assert_eq!(session.commit_batch(None), Err(Status::InvalidArgument));
        let state = session.state().unwrap();
        assert_eq!(state.batch_kind, Some(BatchKind::Append));
        assert_eq!(state.live_request_count, 0);

        complete_layer(&mut session, BatchKind::Append, 1).unwrap();
        session.commit_batch(None).unwrap();
        assert_eq!(session.state().unwrap().live_seq_lens, &[1]);
    }

    #[test]
    fn config_overflow_is_rejected() {
        let mut config = tiny_config();
        config.max_pages = u32::MAX;
        config.page_size = 2;
        assert_eq!(
            EngineCore::new(config).map(|_| ()),
            Err(Status::InvalidArgument)
        );
    }
}
