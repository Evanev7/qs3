use super::{QwenConfig, checked_usize_product, weights::MoeShape};
use crate::{
    constants::{
        attention::PACKED_Q_GATE_WIDTH,
        gdn::{KEY_HEAD_DIM, NUM_KEY_HEADS, NUM_VALUE_HEADS, OUTPUT_WIDTH, PACKED_QKV_CHANNELS},
    },
    engine::Status,
    memory::{CudaCtx, DeviceBuffer},
};
use std::rc::Rc;

pub(super) struct RunnerScratch {
    config: QwenConfig,
    row_capacity: u32,
    pub(super) token_ids: DeviceBuffer<i32>,
    pub(super) positions: DeviceBuffer<i32>,
    pub(super) residual: DeviceBuffer<u16>,
    pub(super) norm: DeviceBuffer<u16>,
    pub(super) q_proj_out: DeviceBuffer<u16>,
    pub(super) q: DeviceBuffer<u16>,
    pub(super) k: DeviceBuffer<u16>,
    pub(super) v: DeviceBuffer<u16>,
    pub(super) attn_out: DeviceBuffer<u16>,
    pub(super) attn_proj: DeviceBuffer<u16>,
    pub(super) attn_gate: DeviceBuffer<u16>,
    pub(super) mlp_out: DeviceBuffer<u16>,
    pub(super) mlp: MlpScratch,
    pub(super) gdn: Option<GdnScratch>,
    pub(super) logits: DeviceBuffer<f32>,
    pub(super) next_token_ids: DeviceBuffer<i32>,
}

pub(super) enum MlpScratch {
    Dense(DenseScratch),
    Moe(MoeScratch),
}

pub(super) struct DenseScratch {
    pub(super) gate: DeviceBuffer<u16>,
    pub(super) up: DeviceBuffer<u16>,
    pub(super) activated: DeviceBuffer<u16>,
}

pub(super) struct MoeScratch {
    pub(super) router_logits: DeviceBuffer<u16>,
    pub(super) topk_ids: DeviceBuffer<i32>,
    pub(super) topk_weights: DeviceBuffer<f32>,
    pub(super) workspace: DeviceBuffer<u8>,
    pub(super) shared: Option<SharedExpertScratch>,
}

pub(super) struct SharedExpertScratch {
    pub(super) gate: DeviceBuffer<u16>,
    pub(super) up: DeviceBuffer<u16>,
    pub(super) activated: DeviceBuffer<u16>,
    pub(super) out: DeviceBuffer<u16>,
    pub(super) gate_logits: DeviceBuffer<f32>,
}

pub(super) struct GdnScratch {
    pub(super) packed: DeviceBuffer<u16>,
    pub(super) conv_out: DeviceBuffer<u16>,
    pub(super) a: DeviceBuffer<u16>,
    pub(super) b: DeviceBuffer<u16>,
    pub(super) q: DeviceBuffer<u16>,
    pub(super) k: DeviceBuffer<u16>,
    pub(super) v: DeviceBuffer<u16>,
    pub(super) recurrent_out: DeviceBuffer<u16>,
    pub(super) gate: DeviceBuffer<u16>,
    pub(super) norm_out: DeviceBuffer<u16>,
    pub(super) seq_indptr: DeviceBuffer<i32>,
    pub(super) state_indices: DeviceBuffer<i32>,
    pub(super) state_out_indices: DeviceBuffer<i32>,
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
            .then(|| GdnScratch::new(&ctx, 1))
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
    fn new(ctx: &Rc<CudaCtx>, rows: u32) -> Result<Self, Status> {
        let row_count = rows as usize;
        let packed = checked_usize_product(&[rows, PACKED_QKV_CHANNELS])?;
        let heads = checked_usize_product(&[rows, NUM_VALUE_HEADS])?;
        let qk = checked_usize_product(&[rows, NUM_KEY_HEADS, KEY_HEAD_DIM])?;
        let output = checked_usize_product(&[rows, OUTPUT_WIDTH])?;
        Ok(Self {
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
            state_indices: DeviceBuffer::with_capacity(ctx.clone(), row_count)?,
            state_out_indices: DeviceBuffer::with_capacity(ctx.clone(), row_count)?,
        })
    }

    fn realloc(&mut self, rows: u32) -> Result<(), Status> {
        let row_count = rows as usize;
        let packed = checked_usize_product(&[rows, PACKED_QKV_CHANNELS])?;
        let heads = checked_usize_product(&[rows, NUM_VALUE_HEADS])?;
        let qk = checked_usize_product(&[rows, NUM_KEY_HEADS, KEY_HEAD_DIM])?;
        let output = checked_usize_product(&[rows, OUTPUT_WIDTH])?;
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
        self.state_indices.realloc(row_count)?;
        self.state_out_indices.realloc(row_count)?;
        Ok(())
    }
}

#[cfg(test)]
mod view_tests {
    use super::*;
    use crate::backend::{BF16, DMat, F32};

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
        assert_eq!(dense.activated.cap, 4 * config.intermediate_size() as usize);
        let residual = scratch.residual.erase();
        let logits = scratch.logits.erase();
        let next_token = scratch.next_token_ids.erase();

        scratch.reserve(1).unwrap();
        assert_eq!(scratch.residual.erase(), residual);
        assert_eq!(scratch.row_capacity, 4);
        scratch.reserve(8).unwrap();
        assert_eq!(scratch.row_capacity, 8);
        assert_eq!(scratch.residual.cap, 8 * config.hidden_size() as usize);
        assert_eq!(scratch.logits.erase(), logits);
        assert_eq!(scratch.next_token_ids.erase(), next_token);
        let MlpScratch::Dense(dense) = &scratch.mlp else {
            unreachable!();
        };
        assert_eq!(dense.activated.cap, 8 * config.intermediate_size() as usize);
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
            assert_eq!(moe.topk_ids.cap, 4 * shape.num_experts_per_tok as usize);
            assert_eq!(moe.workspace.erase(), workspace);
            assert_eq!(moe.workspace.cap, 64);
            if let Some(shared) = &moe.shared {
                assert_eq!(
                    shared.activated.cap,
                    4 * shape.shared_expert_intermediate_size as usize
                );
            }
            if let Some(gdn) = &scratch.gdn {
                assert_eq!(gdn.packed.cap, 4 * PACKED_QKV_CHANNELS as usize);
                assert_eq!(gdn.seq_indptr.cap, 2);
            }
        }
        ctx.synchronize().unwrap();
    }

    #[test]
    fn typed_views_check_capacity_offsets_and_shape_overflow() {
        let ctx = Rc::new(CudaCtx::default().unwrap());
        let buffer = DeviceBuffer::<u16>::with_capacity(ctx.clone(), 8).unwrap();
        let matrix: DMat<BF16> = buffer.matrix(2, 4).unwrap();
        // SAFETY: buffer owns the entire matrix and remains live for both calls.
        unsafe {
            assert_eq!(matrix.row(1).unwrap(), buffer.matrix_at(4, 1, 4).unwrap());
            assert!(matrix.row(2).is_err());
        }
        assert!(buffer.matrix(3, 3).is_err());
        assert!(buffer.matrix_at(5, 1, 4).is_err());
        assert!(buffer.matrix_at(usize::MAX, 1, 1).is_err());
        assert!(buffer.matrix(u32::MAX, u32::MAX).is_err());
        assert!(buffer.matrix(0, 4).is_err());
        assert!(buffer.vector(8).is_ok());
        assert!(buffer.vector(9).is_err());
        assert!(buffer.tensor3(2, 2, 2).is_ok());
        assert!(buffer.tensor3(2, 2, 3).is_err());
        assert!(buffer.heads(1, 2, 4).is_ok());
        assert!(buffer.heads(2, 2, 4).is_err());

        let buffer = DeviceBuffer::<f32>::with_capacity(ctx, 8).unwrap();
        let matrix: DMat<F32> = buffer.matrix(2, 4).unwrap();
        // SAFETY: buffer owns the entire matrix and remains live here.
        assert_eq!(
            unsafe { matrix.row(1) }.unwrap(),
            buffer.matrix_at(4, 1, 4).unwrap()
        );
    }

    #[test]
    fn workspace_view_checks_requested_bytes() {
        let ctx = Rc::new(CudaCtx::default().unwrap());
        let buffer = DeviceBuffer::<u8>::with_capacity(ctx, 32).unwrap();
        assert!(buffer.workspace(0).is_ok());
        assert!(buffer.workspace(16).is_ok());
        assert!(buffer.workspace(32).is_ok());
        assert!(buffer.workspace(33).is_err());
    }
}
