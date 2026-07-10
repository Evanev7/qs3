use super::{BatchKind, EngineConfig, KvLayout, RequestId, Status};
use crate::ext::{Cast, SafeHashSet, SafeVec, try_clone_slice};

use std::collections::HashSet;

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

impl StagedRow {
    fn new_seq_len(&self) -> Result<u32, Status> {
        self.old_seq_len
            .checked_add(self.token_count)
            .ok_or(Status::InvalidArgument)
    }
}

#[derive(Clone, Debug)]
struct BatchCandidate {
    batch: Batch,
    free_pages: Vec<i32>,
    request_ids: Vec<RequestId>,
    tokens: Vec<i32>,
    qo_indptr: Vec<i32>,
    kv_indptr: Vec<i32>,
    kv_indices: Vec<i32>,
    last_page_len: Vec<i32>,
    rope_pos_offset: Vec<i32>,
    append_batch_indices: Vec<i32>,
    append_positions: Vec<i32>,
}

struct BatchPlanner<'a> {
    config: &'a EngineConfig,
    requests: &'a [Request],
    free_pages: &'a [i32],
}

impl<'a> BatchPlanner<'a> {
    fn new(config: &'a EngineConfig, requests: &'a [Request], free_pages: &'a [i32]) -> Self {
        Self {
            config,
            requests,
            free_pages,
        }
    }

    fn plan_append(
        &self,
        request_ids: &[RequestId],
        token_indptr: &[i32],
        tokens: &[i32],
    ) -> Result<BatchCandidate, Status> {
        let batch_size = usize_to_u32(request_ids.len())?;
        let token_count = usize_to_u32(tokens.len())?;
        if batch_size == 0
            || batch_size > self.config.max_batch_rows
            || token_count == 0
            || token_count > self.config.max_batch_tokens
        {
            return Err(Status::InvalidArgument);
        }

        self.validate_unique_request_ids(request_ids)?;
        self.validate_append_indptr(request_ids, token_indptr, tokens)?;
        let staged_rows = self.stage_append_rows(request_ids, token_indptr, tokens)?;
        let new_request_count = staged_rows
            .iter()
            .filter(|row| row.request_index.is_none())
            .count();
        if self.requests.len() + new_request_count > self.config.max_live_requests as usize {
            return Err(Status::InvalidArgument);
        }

        self.build_candidate(Some(token_indptr), request_ids, tokens, staged_rows)
    }

    fn plan_decode(
        &self,
        request_ids: &[RequestId],
        tokens: &[i32],
    ) -> Result<BatchCandidate, Status> {
        if request_ids.is_empty()
            || request_ids.len() != tokens.len()
            || request_ids.len() > self.config.max_batch_rows as usize
        {
            return Err(Status::InvalidArgument);
        }

        self.validate_unique_request_ids(request_ids)?;
        let staged_rows = self.stage_decode_rows(request_ids, tokens)?;
        self.build_candidate(None, request_ids, tokens, staged_rows)
    }

    fn validate_unique_request_ids(&self, request_ids: &[RequestId]) -> Result<(), Status> {
        let mut seen = HashSet::new();
        seen.safe_reserve(request_ids.len())?;
        for id in request_ids {
            if !seen.insert(*id) {
                return Err(Status::InvalidArgument);
            }
        }
        Ok(())
    }

    fn validate_append_indptr(
        &self,
        request_ids: &[RequestId],
        token_indptr: &[i32],
        tokens: &[i32],
    ) -> Result<(), Status> {
        if token_indptr.len() != request_ids.len() + 1 || token_indptr[0] != 0 {
            return Err(Status::InvalidArgument);
        }
        for i in 0..request_ids.len() {
            if token_indptr[i] < 0 || token_indptr[i + 1] < token_indptr[i] {
                return Err(Status::InvalidArgument);
            }
        }
        if token_indptr[request_ids.len()] != usize_to_i32(tokens.len())? {
            return Err(Status::InvalidArgument);
        }
        Ok(())
    }

    fn stage_append_rows(
        &self,
        request_ids: &[RequestId],
        token_indptr: &[i32],
        tokens: &[i32],
    ) -> Result<Vec<StagedRow>, Status> {
        let mut staged_rows = Vec::safe_new(request_ids.len())?;
        for (i, &id) in request_ids.iter().enumerate() {
            let token_range = token_indptr[i].cast()..token_indptr[i + 1].cast();
            let row_token_count = usize_to_u32(token_range.len())?;
            staged_rows.push(self.stage_row(
                id,
                self.find_request_index(id),
                row_token_count,
                &tokens[token_range],
            )?);
        }
        Ok(staged_rows)
    }

    fn stage_decode_rows(
        &self,
        request_ids: &[RequestId],
        tokens: &[i32],
    ) -> Result<Vec<StagedRow>, Status> {
        let mut staged_rows = Vec::safe_new(request_ids.len())?;
        for (i, &id) in request_ids.iter().enumerate() {
            let request_index = self.find_request_index(id).ok_or(Status::InvalidArgument)?;
            staged_rows.push(self.stage_row(id, Some(request_index), 1, &tokens[i..i + 1])?);
        }
        Ok(staged_rows)
    }

    fn stage_row(
        &self,
        id: RequestId,
        request_index: Option<usize>,
        token_count: u32,
        tokens: &[i32],
    ) -> Result<StagedRow, Status> {
        let old_seq_len = request_index.map_or(0, |idx| self.requests[idx].seq_len);
        let new_seq_len = old_seq_len
            .checked_add(token_count)
            .ok_or(Status::InvalidArgument)?;
        if token_count == 0 || new_seq_len > self.config.max_seq_len {
            return Err(Status::InvalidArgument);
        }

        let old_pages: &[i32] =
            request_index.map_or(&[], |idx| self.requests[idx].pages.as_slice());
        Ok(StagedRow {
            id,
            request_index,
            old_seq_len,
            old_page_count: old_pages.len(),
            token_count,
            tokens: try_clone_slice(tokens)?,
            pages: try_clone_slice(old_pages)?,
        })
    }

    fn build_candidate(
        &self,
        append_qo_indptr: Option<&[i32]>,
        request_ids: &[RequestId],
        tokens: &[i32],
        mut staged_rows: Vec<StagedRow>,
    ) -> Result<BatchCandidate, Status> {
        let extra_pages = self.extra_page_count(&staged_rows)?;
        if extra_pages > self.free_pages.len() {
            return Err(Status::OutOfMemory);
        }

        let mut next_free_pages = try_clone_slice(self.free_pages)?;
        let batch_request_ids = try_clone_slice(request_ids)?;
        let batch_tokens = try_clone_slice(tokens)?;
        let is_append = append_qo_indptr.is_some();
        let batch_qo_indptr = if let Some(qo_indptr) = append_qo_indptr {
            try_clone_slice(qo_indptr)?
        } else {
            Vec::new()
        };
        let mut batch_kv_indptr = Vec::safe_new(request_ids.len() + 1)?;
        let mut batch_kv_indices =
            Vec::safe_new(staged_rows.iter().try_fold(0usize, |acc, row| {
                let new_seq_len = row.new_seq_len()?;
                acc.checked_add(page_count_for_len(new_seq_len, self.config.page_size)? as usize)
                    .ok_or(Status::InvalidArgument)
            })?)?;
        let mut batch_last_page_len = Vec::safe_new(request_ids.len())?;
        let mut batch_rope_pos_offset = Vec::safe_new(request_ids.len())?;
        let append_capacity = if is_append { tokens.len() } else { 0 };
        let mut batch_append_batch_indices = Vec::safe_new(append_capacity)?;
        let mut batch_append_positions = Vec::safe_new(append_capacity)?;
        batch_kv_indptr.push(0);

        for (idx, row) in staged_rows.iter_mut().enumerate() {
            let new_seq_len = row.new_seq_len()?;
            let needed_pages = page_count_for_len(new_seq_len, self.config.page_size)? as usize;
            while row.pages.len() < needed_pages {
                let page = next_free_pages.pop().ok_or(Status::OutOfMemory)?;
                row.pages.push(page);
            }
            batch_kv_indices.extend_from_slice(&row.pages);
            batch_kv_indptr.push(usize_to_i32(batch_kv_indices.len())?);
            batch_last_page_len.push(last_page_len_for_seq(new_seq_len, self.config.page_size));
            batch_rope_pos_offset.push(0);

            if is_append {
                for j in 0..row.token_count {
                    batch_append_batch_indices.push(usize_to_i32(idx)?);
                    let pos = row
                        .old_seq_len
                        .checked_add(j)
                        .ok_or(Status::InvalidArgument)?;
                    batch_append_positions.push(u32_to_i32(pos)?);
                }
            }
        }

        let batch_size = usize_to_u32(request_ids.len())?;
        let token_count = usize_to_u32(tokens.len())?;
        let batch_kind = if is_append {
            BatchKind::Append
        } else {
            BatchKind::Decode
        };
        let batch = Batch::new(
            batch_kind,
            batch_size,
            token_count,
            staged_rows,
            self.config.num_layers,
        );

        Ok(BatchCandidate {
            batch,
            free_pages: next_free_pages,
            request_ids: batch_request_ids,
            tokens: batch_tokens,
            qo_indptr: batch_qo_indptr,
            kv_indptr: batch_kv_indptr,
            kv_indices: batch_kv_indices,
            last_page_len: batch_last_page_len,
            rope_pos_offset: batch_rope_pos_offset,
            append_batch_indices: batch_append_batch_indices,
            append_positions: batch_append_positions,
        })
    }

    fn extra_page_count(&self, staged_rows: &[StagedRow]) -> Result<usize, Status> {
        staged_rows.iter().try_fold(0usize, |acc, row| {
            let needed_pages =
                page_count_for_len(row.new_seq_len()?, self.config.page_size)? as usize;
            let new_pages = needed_pages
                .checked_sub(row.old_page_count)
                .ok_or(Status::InternalError)?;
            acc.checked_add(new_pages).ok_or(Status::InvalidArgument)
        })
    }

    fn find_request_index(&self, id: RequestId) -> Option<usize> {
        self.requests.iter().position(|req| req.id == id)
    }
}

#[derive(Clone, Debug)]
struct Batch {
    kind: BatchKind,
    size: u32,
    token_count: u32,
    rows: Vec<StagedRow>,
    layers: LayerProgress,
}

impl Batch {
    fn new(
        kind: BatchKind,
        size: u32,
        token_count: u32,
        rows: Vec<StagedRow>,
        layer_count: u32,
    ) -> Self {
        Self {
            kind,
            size,
            token_count,
            rows,
            layers: LayerProgress::new(layer_count),
        }
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

    batch: Option<Batch>,

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

/// A coherent borrowed view of the candidate transaction installed by
/// `begin_append` or `begin_decode`.
///
/// Device metadata upload and attention planning must consume this view as a
/// unit: mixing fields from different engine states would break the cache
/// transaction contract.
#[derive(Clone, Copy, Debug)]
pub(crate) struct ActiveBatch<'a> {
    pub kind: BatchKind,
    pub size: u32,
    pub token_count: u32,
    pub request_ids: &'a [RequestId],
    pub tokens: &'a [i32],
    pub qo_indptr: &'a [i32],
    pub kv_indptr: &'a [i32],
    pub kv_indices: &'a [i32],
    pub last_page_len: &'a [i32],
    pub rope_pos_offset: &'a [i32],
    pub append_batch_indices: &'a [i32],
    pub append_positions: &'a [i32],
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
        || config.max_batch_rows == 0
        || config.max_batch_tokens == 0
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
    if config.max_batch_tokens < config.max_batch_rows {
        return Err(Status::InvalidArgument);
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

        let mut free_pages = Vec::safe_new(config.max_pages as usize)?;
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
            batch: None,
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
        let batch = self.batch.as_ref();
        Ok(CoreState {
            batch_kind: batch.map(|batch| batch.kind),
            live_request_count: usize_to_u32(self.requests.len())?,
            batch_size: batch.map_or(0, |batch| batch.size),
            batch_token_count: batch.map_or(0, |batch| batch.token_count),
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

    pub(crate) fn active_batch(&self) -> Result<ActiveBatch<'_>, Status> {
        let batch = self.batch.as_ref().ok_or(Status::InvalidArgument)?;
        Ok(ActiveBatch {
            kind: batch.kind,
            size: batch.size,
            token_count: batch.token_count,
            request_ids: &self.batch_request_ids,
            tokens: &self.batch_tokens,
            qo_indptr: &self.batch_qo_indptr,
            kv_indptr: &self.batch_kv_indptr,
            kv_indices: &self.batch_kv_indices,
            last_page_len: &self.batch_last_page_len,
            rope_pos_offset: &self.batch_rope_pos_offset,
            append_batch_indices: &self.batch_append_batch_indices,
            append_positions: &self.batch_append_positions,
        })
    }

    pub fn reset(&mut self) -> Result<(), Status> {
        let mut next_free_pages = Vec::safe_new(self.config.max_pages as usize)?;
        for page in (0..self.config.max_pages).rev() {
            next_free_pages.push(u32_to_i32(page)?);
        }
        let next_requests = Vec::new();
        let live_views = self.build_live_views_for(&next_requests)?;

        self.requests = next_requests;
        self.free_pages = next_free_pages;
        self.clear_batch();
        self.install_live_views(live_views);
        #[cfg(debug_assertions)]
        self.check_allocator_invariants()?;
        Ok(())
    }

    pub fn release_requests(&mut self, request_ids: &[RequestId]) -> Result<(), Status> {
        if self.batch.is_some() {
            return Err(Status::InvalidArgument);
        }
        let mut seen = HashSet::safe_new(request_ids.len())?;
        let mut release_indices = Vec::safe_new(request_ids.len())?;
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
        release_set.safe_reserve(release_indices.len())?;
        for idx in &release_indices {
            release_set.insert(*idx);
        }

        let mut next_requests =
            Vec::safe_new(self.requests.len().saturating_sub(release_set.len()))?;
        let mut next_free_pages = try_clone_slice(&self.free_pages)?;
        let released_page_count = release_indices
            .iter()
            .try_fold(0usize, |acc, idx| {
                acc.checked_add(self.requests[*idx].pages.len())
            })
            .ok_or(Status::InvalidArgument)?;
        next_free_pages.safe_reserve(released_page_count)?;

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
        if self.batch.is_some() {
            return Err(Status::InvalidArgument);
        }
        let candidate = BatchPlanner::new(&self.config, &self.requests, &self.free_pages)
            .plan_append(request_ids, token_indptr, tokens)?;
        self.install_batch_candidate(candidate);
        #[cfg(debug_assertions)]
        self.check_allocator_invariants()?;
        Ok(())
    }

    pub fn begin_decode(
        &mut self,
        request_ids: &[RequestId],
        tokens: &[i32],
    ) -> Result<(), Status> {
        if self.batch.is_some() {
            return Err(Status::InvalidArgument);
        }
        let candidate = BatchPlanner::new(&self.config, &self.requests, &self.free_pages)
            .plan_decode(request_ids, tokens)?;
        self.install_batch_candidate(candidate);
        #[cfg(debug_assertions)]
        self.check_allocator_invariants()?;
        Ok(())
    }

    pub fn commit_batch(&mut self, accepted_token_counts: Option<&[u32]>) -> Result<(), Status> {
        let batch = self.batch.as_ref().ok_or(Status::InvalidArgument)?;
        if !batch.layers.is_complete() {
            return Err(Status::InvalidArgument);
        }
        if let Some(counts) = accepted_token_counts
            && counts.len() != batch.rows.len()
        {
            return Err(Status::InvalidArgument);
        }

        let mut accepted = Vec::safe_new(batch.rows.len())?;
        for (i, row) in batch.rows.iter().enumerate() {
            let count = accepted_token_counts.map_or(row.token_count, |counts| counts[i]);
            if count > row.token_count {
                return Err(Status::InvalidArgument);
            }
            if batch.kind == BatchKind::Decode && count > 1 {
                return Err(Status::InvalidArgument);
            }
            accepted.push(count);
        }

        let mut next_requests = self.clone_requests()?;
        let mut next_free_pages = try_clone_slice(&self.free_pages)?;

        for (i, row) in batch.rows.iter().enumerate() {
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
                    req.tokens.safe_reserve(accepted_len)?;
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
            next_free_pages.safe_reserve(row.pages.len() - needed_pages)?;
            next_free_pages.extend_from_slice(&row.pages[needed_pages..]);
        }

        let live_views = self.build_live_views_for(&next_requests)?;
        self.requests = next_requests;
        self.free_pages = next_free_pages;
        self.clear_batch();
        self.install_live_views(live_views);
        #[cfg(debug_assertions)]
        self.check_allocator_invariants()?;
        Ok(())
    }

    pub fn abort_batch(&mut self) -> Result<(), Status> {
        let Some(batch) = self.batch.as_ref() else {
            return Ok(());
        };
        let mut next_free_pages = try_clone_slice(&self.free_pages)?;
        Self::return_staged_pages(batch, &mut next_free_pages)?;
        let live_views = self.build_live_views_for(&self.requests)?;

        self.free_pages = next_free_pages;
        self.clear_batch();
        self.install_live_views(live_views);
        #[cfg(debug_assertions)]
        self.check_allocator_invariants()?;
        Ok(())
    }

    pub(crate) fn pending_append_layer(
        &self,
        layer_idx: u32,
    ) -> Result<PendingAppendLayer, Status> {
        match self.batch.as_ref() {
            Some(batch) if batch.kind == BatchKind::Append => {
                batch.layers.pending(layer_idx).map(PendingAppendLayer)
            }
            _ => Err(Status::InvalidArgument),
        }
    }

    pub(crate) fn complete_append_layer(
        &mut self,
        pending: PendingAppendLayer,
    ) -> Result<(), Status> {
        match self.batch.as_mut() {
            Some(batch) if batch.kind == BatchKind::Append => batch.layers.complete(pending.0),
            _ => Err(Status::InternalError),
        }
    }

    pub(crate) fn pending_decode_layer(
        &self,
        layer_idx: u32,
    ) -> Result<PendingDecodeLayer, Status> {
        match self.batch.as_ref() {
            Some(batch) if batch.kind == BatchKind::Decode => {
                batch.layers.pending(layer_idx).map(PendingDecodeLayer)
            }
            _ => Err(Status::InvalidArgument),
        }
    }

    pub(crate) fn complete_decode_layer(
        &mut self,
        pending: PendingDecodeLayer,
    ) -> Result<(), Status> {
        match self.batch.as_mut() {
            Some(batch) if batch.kind == BatchKind::Decode => batch.layers.complete(pending.0),
            _ => Err(Status::InternalError),
        }
    }

    fn clone_requests(&self) -> Result<Vec<Request>, Status> {
        let mut out = Vec::safe_new(self.requests.len())?;
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

    fn return_staged_pages(batch: &Batch, free_pages: &mut Vec<i32>) -> Result<(), Status> {
        let rows = &batch.rows;
        let return_count = rows
            .iter()
            .map(|row| row.pages.len().saturating_sub(row.old_page_count))
            .sum();
        free_pages.safe_reserve(return_count)?;
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

    fn clear_batch(&mut self) {
        self.batch = None;
        self.clear_batch_views();
    }

    fn install_batch_candidate(&mut self, candidate: BatchCandidate) {
        self.free_pages = candidate.free_pages;
        self.batch = Some(candidate.batch);
        self.batch_request_ids = candidate.request_ids;
        self.batch_tokens = candidate.tokens;
        self.batch_qo_indptr = candidate.qo_indptr;
        self.batch_kv_indptr = candidate.kv_indptr;
        self.batch_kv_indices = candidate.kv_indices;
        self.batch_last_page_len = candidate.last_page_len;
        self.batch_rope_pos_offset = candidate.rope_pos_offset;
        self.batch_append_batch_indices = candidate.append_batch_indices;
        self.batch_append_positions = candidate.append_positions;
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
        let mut live_request_ids = Vec::safe_new(requests.len())?;
        let mut live_seq_lens = Vec::safe_new(requests.len())?;
        let mut live_token_indptr = Vec::safe_new(requests.len() + 1)?;
        let mut live_tokens = Vec::safe_new(total_tokens)?;
        let mut live_kv_indptr = Vec::safe_new(requests.len() + 1)?;
        let mut live_kv_indices = Vec::safe_new(total_pages)?;
        let mut live_last_page_len = Vec::safe_new(requests.len())?;

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

        if let Some(batch) = &self.batch {
            for row in &batch.rows {
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
    use super::EngineCore;
    use crate::{
        QWEN36_FULL_ATTN_HEAD_DIM, QWEN36_FULL_ATTN_KV_HEADS, QWEN36_FULL_ATTN_Q_HEADS,
        QWEN36_HIDDEN_SIZE,
        engine::{BatchKind, DynDType, EngineConfig, KvLayout, Status},
        ffi,
    };

    fn tiny_config() -> EngineConfig {
        EngineConfig {
            device_ordinal: 0,
            stream: std::ptr::null_mut(),
            num_layers: 1,
            max_live_requests: 4,
            max_batch_rows: 3,
            max_batch_tokens: 8,
            max_seq_len: 8,
            max_pages: 8,
            page_size: 4,
            hidden_size: QWEN36_HIDDEN_SIZE,
            intermediate_size: 0,
            vocab_size: 0,
            num_q_heads: QWEN36_FULL_ATTN_Q_HEADS,
            num_kv_heads: QWEN36_FULL_ATTN_KV_HEADS,
            head_dim: QWEN36_FULL_ATTN_HEAD_DIM,
            activation_dtype: DynDType::F16,
            kv_dtype: DynDType::F16,
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
        assert_eq!(DynDType::F32.bits(), 32);
        assert_eq!(DynDType::F16.bits(), 16);
        assert_eq!(DynDType::BF16.bits(), 16);
        assert_eq!(DynDType::FP8E4M3.bits(), 8);
        assert_eq!(DynDType::FP8E5M2.bits(), 8);
        assert_eq!(DynDType::NVFP4E2M1.bits(), 4);
        assert_eq!(DynDType::MXFP4E2M1.bits(), 4);
        assert_eq!(DynDType::MXFP8E4M3.bits(), 8);
        assert_eq!(DynDType::I32.bits(), 32);
        assert_eq!(DynDType::U32.bits(), 32);
        assert_eq!(DynDType::I8.bits(), 8);
        assert_eq!(DynDType::U8.bits(), 8);
        assert_eq!(DynDType::F16.storage_bytes_for(3), Ok(6));
        assert_eq!(DynDType::FP8E4M3.storage_bytes_for(3), Ok(3));
        assert_eq!(DynDType::NVFP4E2M1.storage_bytes_for(1), Ok(1));
        assert_eq!(DynDType::NVFP4E2M1.storage_bytes_for(2), Ok(1));
        assert_eq!(DynDType::NVFP4E2M1.storage_bytes_for(3), Ok(2));
    }

    #[test]
    fn dtype_and_layout_raw_values_match_qsfi() {
        assert_eq!(DynDType::F32.to_raw(), ffi::DTYPE_F32);
        assert_eq!(DynDType::F16.to_raw(), ffi::DTYPE_F16);
        assert_eq!(DynDType::BF16.to_raw(), ffi::DTYPE_BF16);
        assert_eq!(DynDType::FP8E4M3.to_raw(), ffi::DTYPE_FP8_E4M3);
        assert_eq!(DynDType::FP8E5M2.to_raw(), ffi::DTYPE_FP8_E5M2);
        assert_eq!(DynDType::NVFP4E2M1.to_raw(), ffi::DTYPE_NVFP4_E2M1);
        assert_eq!(DynDType::MXFP4E2M1.to_raw(), ffi::DTYPE_MXFP4_E2M1);
        assert_eq!(DynDType::MXFP8E4M3.to_raw(), ffi::DTYPE_MXFP8_E4M3);
        assert_eq!(DynDType::I32.to_raw(), ffi::DTYPE_I32);
        assert_eq!(DynDType::U32.to_raw(), ffi::DTYPE_U32);
        assert_eq!(DynDType::I8.to_raw(), ffi::DTYPE_I8);
        assert_eq!(DynDType::U8.to_raw(), ffi::DTYPE_U8);
        assert_eq!(KvLayout::NHD.to_raw(), ffi::KV_LAYOUT_NHD);
        assert_eq!(KvLayout::HND.to_raw(), ffi::KV_LAYOUT_HND);
    }

    #[test]
    fn active_batch_is_one_coherent_transaction_view() {
        let mut core = EngineCore::new(tiny_config()).unwrap();
        core.begin_append(&[7], &[0, 2], &[10, 11]).unwrap();

        let batch = core.active_batch().unwrap();
        assert_eq!(batch.kind, BatchKind::Append);
        assert_eq!(batch.size, 1);
        assert_eq!(batch.token_count, 2);
        assert_eq!(batch.request_ids, &[7]);
        assert_eq!(batch.tokens, &[10, 11]);
        assert_eq!(batch.qo_indptr, &[0, 2]);
        assert_eq!(batch.kv_indptr.len(), 2);
        assert_eq!(batch.last_page_len, &[2]);
        assert_eq!(batch.append_batch_indices, &[0, 0]);
        assert_eq!(batch.append_positions, &[0, 1]);
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
    fn append_rejects_batches_above_token_cap_without_allocating_pages() {
        let mut config = tiny_config();
        config.max_batch_tokens = 3;
        let mut session = EngineCore::new(config).unwrap();

        assert_eq!(
            session.begin_append(&[41], &[0, 4], &[1, 2, 3, 4]),
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
