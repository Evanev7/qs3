use super::{QwenConfig, checked_usize_product, weights::MoeShape};
use crate::{
    QWEN36_GDN_STATE_SLOTS_PER_LAYER,
    backend::{DVec, gdn_prefill::GdnPrefillScratch},
    constants::{
        attention::PACKED_Q_GATE_WIDTH,
        gdn::{KEY_HEAD_DIM, NUM_KEY_HEADS, NUM_VALUE_HEADS, OUTPUT_WIDTH, PACKED_QKV_CHANNELS},
    },
    dtype::{BF16, DType, F32, I32, U8},
    engine::Status,
    ffi::DevicePtr,
    memory::{CudaCtx, DeviceBuffer, HostBuffer},
};
use std::rc::Rc;

pub(super) struct RunnerScratch {
    config: QwenConfig,
    row_capacity: u32,
    pub(super) token_ids: DeviceBuffer<I32>,
    pub(super) positions: DeviceBuffer<I32>,
    pub(super) residual: DeviceBuffer<BF16>,
    pub(super) norm: DeviceBuffer<BF16>,
    pub(super) q_proj_out: DeviceBuffer<BF16>,
    pub(super) q: DeviceBuffer<BF16>,
    pub(super) k: DeviceBuffer<BF16>,
    pub(super) v: DeviceBuffer<BF16>,
    pub(super) attn_out: DeviceBuffer<BF16>,
    pub(super) attn_proj: DeviceBuffer<BF16>,
    pub(super) attn_gate: DeviceBuffer<BF16>,
    pub(super) mlp_out: DeviceBuffer<BF16>,
    pub(super) mlp: MlpScratch,
    pub(super) gdn: Option<GdnScratch>,
    pub(super) logits: DeviceBuffer<F32>,
    pub(super) next_token_ids: DeviceBuffer<I32>,
}

pub(super) enum MlpScratch {
    Dense(DenseScratch),
    Moe(MoeScratch),
}

pub(super) struct DenseScratch {
    pub(super) gate: DeviceBuffer<BF16>,
    pub(super) up: DeviceBuffer<BF16>,
    pub(super) activated: DeviceBuffer<BF16>,
}

pub(super) struct MoeScratch {
    pub(super) router_logits: DeviceBuffer<BF16>,
    pub(super) topk_ids: DeviceBuffer<I32>,
    pub(super) topk_weights: DeviceBuffer<F32>,
    pub(super) workspace: DeviceBuffer<U8>,
    pub(super) shared: Option<SharedExpertScratch>,
}

pub(super) struct SharedExpertScratch {
    pub(super) gate: DeviceBuffer<BF16>,
    pub(super) up: DeviceBuffer<BF16>,
    pub(super) activated: DeviceBuffer<BF16>,
    pub(super) out: DeviceBuffer<BF16>,
    pub(super) gate_logits: DeviceBuffer<F32>,
}

pub(super) struct GdnScratch {
    pub(super) prefill: GdnPrefillScratch,
    pub(super) packed: DeviceBuffer<BF16>,
    pub(super) conv_out: DeviceBuffer<BF16>,
    pub(super) a: DeviceBuffer<BF16>,
    pub(super) b: DeviceBuffer<BF16>,
    pub(super) q: DeviceBuffer<BF16>,
    pub(super) k: DeviceBuffer<BF16>,
    pub(super) v: DeviceBuffer<BF16>,
    pub(super) recurrent_out: DeviceBuffer<BF16>,
    pub(super) gate: DeviceBuffer<BF16>,
    pub(super) norm_out: DeviceBuffer<BF16>,
    pub(super) seq_indptr: DeviceBuffer<I32>,
    // Immutable identity table: select an entry instead of uploading a slot ID.
    slot_indices: DeviceBuffer<I32>,
}

impl RunnerScratch {
    /// Allocates all scratch for the initial token capacity.
    ///
    /// Prepare the MoE plan first and pass its workspace size for the maximum
    /// planned token count; dense models pass zero. Growth does not resize this
    /// workspace. Full-attention-only fixtures omit GDN scratch, and models
    /// without a shared expert omit its scratch. All present buffers own storage.
    /// Full-model validation belongs to ModelRunner; kernel fixtures also use
    /// this storage with reduced dimensions or without a cuBLAS workspace.
    pub(super) fn new(
        ctx: Rc<CudaCtx>,
        config: &QwenConfig,
        moe_workspace_bytes: usize,
    ) -> Result<Self, Status> {
        if config.moe_config().is_some() != (moe_workspace_bytes != 0) {
            return Err(Status::InvalidArgument);
        }
        let hidden = config.hidden_size() as usize;
        let q_hidden = config.q_hidden_size()? as usize;
        let q_proj_out = PACKED_Q_GATE_WIDTH as usize;
        let kv_hidden = config.kv_hidden_size()? as usize;
        let mlp = match config.moe_config() {
            Some(moe) => MlpScratch::Moe(MoeScratch::new(
                &ctx,
                config.hidden_size(),
                moe,
                moe_workspace_bytes,
            )?),
            None => MlpScratch::Dense(DenseScratch::new(&ctx, config.intermediate_size())?),
        };
        let gdn = config
            .has_gdn_layers()
            .then(|| GdnScratch::new(&ctx, config.gdn_layer_count(), 1))
            .transpose()?;
        Ok(Self {
            config: *config,
            row_capacity: 1,
            token_ids: DeviceBuffer::with_capacity(ctx.clone(), 1)?,
            positions: DeviceBuffer::with_capacity(ctx.clone(), 1)?,
            residual: DeviceBuffer::with_capacity(ctx.clone(), hidden)?,
            norm: DeviceBuffer::with_capacity(ctx.clone(), hidden)?,
            q_proj_out: DeviceBuffer::with_capacity(ctx.clone(), q_proj_out)?,
            q: DeviceBuffer::with_capacity(ctx.clone(), q_hidden)?,
            k: DeviceBuffer::with_capacity(ctx.clone(), kv_hidden)?,
            v: DeviceBuffer::with_capacity(ctx.clone(), kv_hidden)?,
            attn_out: DeviceBuffer::with_capacity(ctx.clone(), q_hidden)?,
            attn_proj: DeviceBuffer::with_capacity(ctx.clone(), hidden)?,
            attn_gate: DeviceBuffer::with_capacity(ctx.clone(), q_hidden)?,
            mlp_out: DeviceBuffer::with_capacity(ctx.clone(), hidden)?,
            mlp,
            gdn,
            logits: DeviceBuffer::with_capacity(ctx.clone(), config.vocab_size() as usize)?,
            next_token_ids: DeviceBuffer::with_capacity(ctx, 1)?,
        })
    }

    /// Grows token-dependent storage before preparing views for execution.
    /// Smaller batches retain capacity; logits and the prepared MoE workspace
    /// retain their allocations. An allocation failure can leave some larger
    /// buffers, but does not advance the recorded token capacity. This does not
    /// promise recovery from asynchronous CUDA execution errors.
    pub(super) fn reserve(&mut self, rows: u32) -> Result<(), Status> {
        if rows == 0 {
            return Err(Status::InvalidArgument);
        }
        if rows <= self.row_capacity {
            return Ok(());
        }
        let config = &self.config;
        let row_count = rows as usize;
        let hidden = checked_usize_product(&[rows, config.hidden_size()])?;
        let q_hidden = checked_usize_product(&[rows, config.q_hidden_size()?])?;
        let q_proj_out = checked_usize_product(&[rows, PACKED_Q_GATE_WIDTH])?;
        let kv_hidden = checked_usize_product(&[rows, config.kv_hidden_size()?])?;
        self.token_ids.realloc(row_count)?;
        self.positions.realloc(row_count)?;
        self.residual.realloc(hidden)?;
        self.norm.realloc(hidden)?;
        self.q_proj_out.realloc(q_proj_out)?;
        self.q.realloc(q_hidden)?;
        self.k.realloc(kv_hidden)?;
        self.v.realloc(kv_hidden)?;
        self.attn_out.realloc(q_hidden)?;
        self.attn_proj.realloc(hidden)?;
        self.attn_gate.realloc(q_hidden)?;
        self.mlp_out.realloc(hidden)?;
        match &mut self.mlp {
            MlpScratch::Dense(scratch) => scratch.realloc(config.intermediate_size(), rows)?,
            MlpScratch::Moe(scratch) => scratch.realloc(
                config.hidden_size(),
                config.moe_config().ok_or(Status::InternalError)?,
                rows,
            )?,
        }
        if let Some(gdn) = &mut self.gdn {
            gdn.realloc(rows)?;
        }
        self.row_capacity = rows;
        Ok(())
    }
}

impl DenseScratch {
    fn new(ctx: &Rc<CudaCtx>, intermediate: u32) -> Result<Self, Status> {
        let len = intermediate as usize;
        Ok(Self {
            gate: DeviceBuffer::with_capacity(ctx.clone(), len)?,
            up: DeviceBuffer::with_capacity(ctx.clone(), len)?,
            activated: DeviceBuffer::with_capacity(ctx.clone(), len)?,
        })
    }

    fn realloc(&mut self, intermediate: u32, rows: u32) -> Result<(), Status> {
        let len = checked_usize_product(&[rows, intermediate])?;
        self.gate.realloc(len)?;
        self.up.realloc(len)?;
        self.activated.realloc(len)?;
        Ok(())
    }
}

impl MoeScratch {
    fn new(
        ctx: &Rc<CudaCtx>,
        hidden: u32,
        moe: MoeShape,
        workspace_bytes: usize,
    ) -> Result<Self, Status> {
        let logits = moe.num_experts as usize;
        let topk = moe.num_experts_per_tok as usize;
        let shared = (moe.shared_expert_intermediate_size != 0)
            .then(|| SharedExpertScratch::new(ctx, hidden, moe.shared_expert_intermediate_size))
            .transpose()?;
        Ok(Self {
            router_logits: DeviceBuffer::with_capacity(ctx.clone(), logits)?,
            topk_ids: DeviceBuffer::with_capacity(ctx.clone(), topk)?,
            topk_weights: DeviceBuffer::with_capacity(ctx.clone(), topk)?,
            workspace: DeviceBuffer::with_capacity(ctx.clone(), workspace_bytes)?,
            shared,
        })
    }

    fn realloc(&mut self, hidden: u32, moe: MoeShape, rows: u32) -> Result<(), Status> {
        let logits = checked_usize_product(&[rows, moe.num_experts])?;
        let topk = checked_usize_product(&[rows, moe.num_experts_per_tok])?;
        self.router_logits.realloc(logits)?;
        self.topk_ids.realloc(topk)?;
        self.topk_weights.realloc(topk)?;
        if let Some(shared) = &mut self.shared {
            shared.realloc(hidden, moe.shared_expert_intermediate_size, rows)?;
        }
        Ok(())
    }
}

impl SharedExpertScratch {
    fn new(ctx: &Rc<CudaCtx>, hidden: u32, intermediate: u32) -> Result<Self, Status> {
        let intermediate = intermediate as usize;
        let hidden = hidden as usize;
        Ok(Self {
            gate: DeviceBuffer::with_capacity(ctx.clone(), intermediate)?,
            up: DeviceBuffer::with_capacity(ctx.clone(), intermediate)?,
            activated: DeviceBuffer::with_capacity(ctx.clone(), intermediate)?,
            out: DeviceBuffer::with_capacity(ctx.clone(), hidden)?,
            gate_logits: DeviceBuffer::with_capacity(ctx.clone(), 1)?,
        })
    }

    fn realloc(&mut self, hidden: u32, intermediate: u32, rows: u32) -> Result<(), Status> {
        let intermediate = checked_usize_product(&[rows, intermediate])?;
        let hidden = checked_usize_product(&[rows, hidden])?;
        self.gate.realloc(intermediate)?;
        self.up.realloc(intermediate)?;
        self.activated.realloc(intermediate)?;
        self.out.realloc(hidden)?;
        self.gate_logits.realloc(rows as usize)?;
        Ok(())
    }
}

impl GdnScratch {
    fn new(ctx: &Rc<CudaCtx>, layer_count: u32, rows: u32) -> Result<Self, Status> {
        let slot_count = layer_count
            .checked_mul(QWEN36_GDN_STATE_SLOTS_PER_LAYER)
            .ok_or(Status::InvalidArgument)?;
        let slot_count = i32::try_from(slot_count).map_err(|_| Status::InvalidArgument)?;
        let mut indices = HostBuffer::<I32>::new(slot_count as usize)?;
        for (bytes, slot) in indices.as_mut().chunks_exact_mut(4).zip(0..slot_count) {
            bytes.copy_from_slice(&slot.to_ne_bytes());
        }
        let slot_indices = indices.upload(ctx.clone())?;
        let packed = checked_usize_product(&[rows, PACKED_QKV_CHANNELS])?;
        let heads = checked_usize_product(&[rows, NUM_VALUE_HEADS])?;
        let qk = checked_usize_product(&[rows, NUM_KEY_HEADS, KEY_HEAD_DIM])?;
        let output = checked_usize_product(&[rows, OUTPUT_WIDTH])?;
        Ok(Self {
            prefill: GdnPrefillScratch::new(ctx.clone(), rows)?,
            packed: DeviceBuffer::with_capacity(ctx.clone(), packed)?,
            conv_out: DeviceBuffer::with_capacity(ctx.clone(), packed)?,
            a: DeviceBuffer::with_capacity(ctx.clone(), heads)?,
            b: DeviceBuffer::with_capacity(ctx.clone(), heads)?,
            q: DeviceBuffer::with_capacity(ctx.clone(), qk)?,
            k: DeviceBuffer::with_capacity(ctx.clone(), qk)?,
            v: DeviceBuffer::with_capacity(ctx.clone(), output)?,
            recurrent_out: DeviceBuffer::with_capacity(ctx.clone(), output)?,
            gate: DeviceBuffer::with_capacity(ctx.clone(), output)?,
            norm_out: DeviceBuffer::with_capacity(ctx.clone(), output)?,
            seq_indptr: DeviceBuffer::with_capacity(ctx.clone(), 2)?,
            slot_indices,
        })
    }

    pub(super) fn state_index(&self, slot: u32) -> Result<DVec<I32>, Status> {
        if slot as usize >= self.slot_indices.len() {
            return Err(Status::InvalidArgument);
        }
        let offset = I32::size_of(slot as usize)?;
        // SAFETY: the checked slot lies within the immutable owned index table.
        let ptr = unsafe { self.slot_indices.as_raw().add(offset) };
        DVec::contiguous(DevicePtr::new(ptr).ok_or(Status::InvalidArgument)?, 1)
    }

    fn realloc(&mut self, rows: u32) -> Result<(), Status> {
        let packed = checked_usize_product(&[rows, PACKED_QKV_CHANNELS])?;
        let heads = checked_usize_product(&[rows, NUM_VALUE_HEADS])?;
        let qk = checked_usize_product(&[rows, NUM_KEY_HEADS, KEY_HEAD_DIM])?;
        let output = checked_usize_product(&[rows, OUTPUT_WIDTH])?;
        self.prefill.reserve(rows)?;
        self.packed.realloc(packed)?;
        self.conv_out.realloc(packed)?;
        self.a.realloc(heads)?;
        self.b.realloc(heads)?;
        self.q.realloc(qk)?;
        self.k.realloc(qk)?;
        self.v.realloc(output)?;
        self.recurrent_out.realloc(output)?;
        self.gate.realloc(output)?;
        self.norm_out.realloc(output)?;
        Ok(())
    }
}

#[cfg(test)]
mod view_tests {
    use super::*;
    use crate::{
        backend::DMat,
        dtype::{BF16, F32},
        model::state::GdnSlotMap,
    };

    #[test]
    fn gdn_index_table_survives_growth_commit_and_reset() {
        let ctx = Rc::new(CudaCtx::default().unwrap());
        let mut scratch = GdnScratch::new(&ctx, 3, 1).unwrap();
        let mut slots = GdnSlotMap::new(3).unwrap();
        let indices_ptr = scratch.slot_indices.erase();
        let sequence_ptr = scratch.seq_indptr.erase();
        let initial: Vec<_> = (0..3)
            .map(|layer| {
                let pair = slots.layer_slots(layer).unwrap();
                (
                    scratch.state_index(pair.live_slot).unwrap(),
                    scratch.state_index(pair.staged_slot).unwrap(),
                )
            })
            .collect();
        // Every layer and both sides of its transaction need distinct entries.
        for (layer, &(read, write)) in initial.iter().enumerate() {
            assert_ne!(read, write);
            for &(other_read, other_write) in &initial[..layer] {
                assert_ne!(read, other_read);
                assert_ne!(read, other_write);
                assert_ne!(write, other_read);
                assert_ne!(write, other_write);
            }
        }
        scratch.realloc(4).unwrap();
        assert_eq!(scratch.slot_indices.erase(), indices_ptr);
        assert_eq!(scratch.seq_indptr.erase(), sequence_ptr);
        assert_eq!(scratch.state_index(6), Err(Status::InvalidArgument));
        assert_eq!(scratch.state_index(u32::MAX), Err(Status::InvalidArgument));

        for (committed, swapped) in [(true, true), (false, true), (true, false)] {
            if committed {
                slots.commit();
            }
            for (layer, &(read, write)) in initial.iter().enumerate() {
                let pair = slots.layer_slots(layer as u32).unwrap();
                let actual = (
                    scratch.state_index(pair.live_slot).unwrap(),
                    scratch.state_index(pair.staged_slot).unwrap(),
                );
                let expected = if swapped {
                    (write, read)
                } else {
                    (read, write)
                };
                assert_eq!(actual, expected);
            }
        }
        slots.reset(3).unwrap();
        for (layer, &expected) in initial.iter().enumerate() {
            let pair = slots.layer_slots(layer as u32).unwrap();
            assert_eq!(
                (
                    scratch.state_index(pair.live_slot).unwrap(),
                    scratch.state_index(pair.staged_slot).unwrap(),
                ),
                expected,
            );
        }

        let mut values = HostBuffer::<I32>::new(6).unwrap();
        unsafe { scratch.slot_indices.download(&mut values).unwrap() };
        ctx.synchronize().unwrap();
        let values: Vec<_> = values
            .as_ref()
            .chunks_exact(4)
            .map(|bytes| i32::from_ne_bytes(bytes.try_into().unwrap()))
            .collect();
        assert_eq!(values, [0, 1, 2, 3, 4, 5]);
    }

    #[test]
    fn dense_scratch_retains_capacity_and_fixed_storage() {
        let ctx = Rc::new(CudaCtx::default().unwrap());
        let config = QwenConfig::randomized_dense_tiny_fixture();
        let mut scratch = RunnerScratch::new(ctx.clone(), &config, 0).unwrap();
        scratch.reserve(4).unwrap();
        assert!(scratch.gdn.is_none());
        let MlpScratch::Dense(dense) = &scratch.mlp else {
            panic!("dense fixture allocated MoE scratch");
        };
        assert_eq!(dense.activated.len, 4 * config.intermediate_size() as usize);
        let residual = scratch.residual.erase();
        let logits = scratch.logits.erase();
        let next_token = scratch.next_token_ids.erase();

        scratch.reserve(1).unwrap();
        assert_eq!(scratch.residual.erase(), residual);
        assert_eq!(scratch.row_capacity, 4);
        scratch.reserve(8).unwrap();
        assert_eq!(scratch.row_capacity, 8);
        assert_eq!(scratch.residual.len, 8 * config.hidden_size() as usize);
        assert_eq!(scratch.logits.erase(), logits);
        assert_eq!(scratch.next_token_ids.erase(), next_token);
        let MlpScratch::Dense(dense) = &scratch.mlp else {
            unreachable!();
        };
        assert_eq!(dense.activated.len, 8 * config.intermediate_size() as usize);
        assert_eq!(scratch.reserve(0), Err(Status::InvalidArgument));
        ctx.synchronize().unwrap();
    }

    #[test]
    fn moe_scratch_groups_optional_storage_and_retains_workspace() {
        let ctx = Rc::new(CudaCtx::default().unwrap());
        for config in [
            QwenConfig::randomized_moe_tiny_fixture(),
            QwenConfig::randomized_shared_moe_tiny_fixture(),
            QwenConfig::randomized_qwen36_moe_gdn_one_block_fixture(),
        ] {
            config.validate().unwrap();
            // Only allocation ownership is exercised here; no MoE kernel uses
            // this deliberately small workspace.
            let mut scratch = RunnerScratch::new(ctx.clone(), &config, 64).unwrap();
            let shape = config.moe_config().unwrap();
            let MlpScratch::Moe(moe) = &scratch.mlp else {
                panic!("MoE fixture allocated dense scratch");
            };
            let workspace = moe.workspace.erase();
            assert_eq!(
                moe.shared.is_some(),
                shape.shared_expert_intermediate_size != 0
            );
            assert_eq!(scratch.gdn.is_some(), config.has_gdn_layers());

            scratch.reserve(4).unwrap();
            let MlpScratch::Moe(moe) = &scratch.mlp else {
                unreachable!();
            };
            assert_eq!(moe.topk_ids.len, 4 * shape.num_experts_per_tok as usize);
            assert_eq!(moe.workspace.erase(), workspace);
            assert_eq!(moe.workspace.len, 64);
            if let Some(shared) = &moe.shared {
                assert_eq!(
                    shared.activated.len,
                    4 * shape.shared_expert_intermediate_size as usize
                );
            }
            if let Some(gdn) = &scratch.gdn {
                assert_eq!(gdn.packed.len, 4 * PACKED_QKV_CHANNELS as usize);
                assert_eq!(gdn.seq_indptr.len, 2);
            }
        }
        ctx.synchronize().unwrap();
    }

    #[test]
    fn typed_views_check_capacity_offsets_and_shape_overflow() {
        let ctx = Rc::new(CudaCtx::default().unwrap());
        let buffer = DeviceBuffer::<BF16>::with_capacity(ctx.clone(), 8).unwrap();
        let matrix: DMat<BF16> = buffer.matrix(2, 4).unwrap();
        // SAFETY: buffer owns the entire matrix and remains live for both calls.
        unsafe {
            assert_eq!(
                matrix.row(1).unwrap(),
                DMat::contiguous(DevicePtr::new(buffer.as_raw().add(8)).unwrap(), 1, 4).unwrap()
            );
            assert!(matrix.row(2).is_err());
        }
        assert!(buffer.matrix(3, 3).is_err());
        assert!(buffer.matrix(u32::MAX, u32::MAX).is_err());
        assert!(buffer.matrix(0, 4).is_err());
        assert!(buffer.vector(8).is_ok());
        assert!(buffer.vector(9).is_err());
        assert!(buffer.tensor3(2, 2, 2).is_ok());
        assert!(buffer.tensor3(2, 2, 3).is_err());
        assert!(buffer.heads(1, 2, 4).is_ok());
        assert!(buffer.heads(2, 2, 4).is_err());

        let buffer = DeviceBuffer::<F32>::with_capacity(ctx, 8).unwrap();
        let matrix: DMat<F32> = buffer.matrix(2, 4).unwrap();
        // SAFETY: buffer owns the entire matrix and remains live here.
        assert_eq!(
            unsafe { matrix.row(1) }.unwrap(),
            DMat::contiguous(
                DevicePtr::new(buffer.as_raw().wrapping_add(16)).unwrap(),
                1,
                4
            )
            .unwrap()
        );
    }

    #[test]
    fn workspace_view_checks_requested_bytes() {
        let ctx = Rc::new(CudaCtx::default().unwrap());
        let buffer = DeviceBuffer::<U8>::with_capacity(ctx, 32).unwrap();
        assert!(buffer.workspace(0).is_ok());
        assert!(buffer.workspace(16).is_ok());
        assert!(buffer.workspace(32).is_ok());
        assert!(buffer.workspace(33).is_err());
    }
}
