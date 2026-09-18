//! Linear projection dispatch and reusable activation scratch/plans.
#[cfg(test)]
mod tests;
use super::BatchExecution;
use crate::{
    Status,
    backend::{
        DMat, DVec, Operators, Workspace,
        qsfi::{Nvfp4Plan, scale_count},
    },
    dtype::{BF16, F32, Fp8E4M3, Nvfp4E2M1, U8},
    engine::Engine,
    memory::{CudaCtx, DeviceBuffer},
    model::{QwenConfig, QwenWeights, weights::*},
};
use std::{
    collections::{HashMap, HashSet, hash_map::Entry},
    rc::Rc,
};

pub(super) struct QuantizedScratch {
    prepared_rows: HashSet<u32>,
    plans: HashMap<[u32; 3], Nvfp4Plan>,
    fp8: DeviceBuffer<Fp8E4M3>,
    fp4: DeviceBuffer<Nvfp4E2M1>,
    scales: DeviceBuffer<Fp8E4M3>,
    workspace: DeviceBuffer<U8>,
    logits: DeviceBuffer<BF16>,
}
impl QuantizedScratch {
    pub(super) fn new(ctx: Rc<CudaCtx>, vocab: u32) -> Result<Self, Status> {
        Ok(Self {
            prepared_rows: HashSet::new(),
            plans: HashMap::new(),
            fp8: DeviceBuffer::with_capacity(ctx.clone(), 1)?,
            fp4: DeviceBuffer::with_capacity(ctx.clone(), 2)?,
            scales: DeviceBuffer::with_capacity(ctx.clone(), 1)?,
            workspace: DeviceBuffer::with_capacity(ctx.clone(), 1)?,
            logits: DeviceBuffer::with_capacity(ctx, vocab as usize)?,
        })
    }
    pub(super) fn prepare(
        &mut self,
        engine: &mut Engine,
        weights: &QwenWeights,
        rows: u32,
        config: &QwenConfig,
    ) -> Result<(), Status> {
        if self.prepared_rows.contains(&rows) {
            return Ok(());
        }
        let QwenWeights::DenseNvfp4(m) = weights else {
            return Err(Status::InternalError);
        };
        let mut k_max = config
            .hidden_size()
            .max(config.intermediate_size())
            .max(config.q_hidden_size()?)
            .max(crate::constants::gdn::OUTPUT_WIDTH);
        let mut shapes = vec![[1, m.lm_head.shape[0], m.lm_head.shape[1]]];
        for layer in &m.layers {
            let (_, mlp) = layer.post_attention_mlp();
            for p in [&mlp.gate_proj, &mlp.up_proj, &mlp.down_proj] {
                shapes.push([rows, p.shape[0], p.shape[1]]);
                k_max = k_max.max(p.shape[1]);
            }
        }
        self.reserve_shapes(engine, rows, k_max, shapes, config.nvfp4_tactic)?;
        self.prepared_rows.insert(rows);
        Ok(())
    }

    fn reserve_shapes(
        &mut self,
        engine: &mut Engine,
        rows: u32,
        k_max: u32,
        shapes: Vec<[u32; 3]>,
        tactic: crate::model::Nvfp4Tactic,
    ) -> Result<(), Status> {
        let elements = (rows as usize)
            .checked_mul(k_max as usize)
            .ok_or(Status::InvalidArgument)?;
        self.fp8.realloc(elements)?;
        self.fp4.realloc(elements)?;
        self.scales.realloc(scale_count(rows, k_max)? as usize)?;
        for shape in shapes {
            let plan = match self.plans.entry(shape) {
                Entry::Occupied(e) => e.into_mut(),
                Entry::Vacant(e) => {
                    e.insert(engine.operators().qsfi().create_nvfp4_plan(shape, tactic)?)
                }
            };
            self.workspace.realloc(plan.workspace_bytes.max(1))?;
        }
        Ok(())
    }
}

/// Borrowed device bindings only; allocation ownership stays in prepared weights.
#[derive(Clone, Copy)]
pub(super) enum LinearWeight {
    Bf16(DMat<BF16>),
    Fp8 {
        weight: DMat<Fp8E4M3>,
        input_scale: DVec<F32>,
        weight_scale: DVec<F32>,
    },
    Nvfp4 {
        weight: DMat<Nvfp4E2M1>,
        block_scales: DVec<Fp8E4M3>,
        quant_multiplier: DVec<F32>,
        alpha: DVec<F32>,
    },
}

pub(super) trait Projection {
    fn view(&self, n: u32, k: u32) -> Result<LinearWeight, Status>;
}
impl Projection for W<BF16> {
    fn view(&self, n: u32, k: u32) -> Result<LinearWeight, Status> {
        Ok(LinearWeight::Bf16(self.matrix(n, k)?))
    }
}
impl Projection for Fp8Block {
    fn view(&self, n: u32, k: u32) -> Result<LinearWeight, Status> {
        if self.shape != [n, k] {
            return Err(Status::InvalidArgument);
        }
        Ok(LinearWeight::Fp8 {
            weight: self.weight.matrix(n, k)?,
            input_scale: self.scales.input_scale()?,
            weight_scale: self.scales.weight_scale()?,
        })
    }
}
impl Projection for Nvfp4Block {
    fn view(&self, n: u32, k: u32) -> Result<LinearWeight, Status> {
        if self.shape != [n, k] {
            return Err(Status::InvalidArgument);
        }
        Ok(LinearWeight::Nvfp4 {
            weight: self.weight.matrix(n, k)?,
            block_scales: self.weight_scale.vector(scale_count(n, k)?)?,
            quant_multiplier: self.parameters.quant_multiplier()?,
            alpha: self.parameters.alpha()?,
        })
    }
}

impl Operators<'_> {
    /// Projects BF16 activations, including activation quantization for FP8/NVFP4.
    ///
    /// # Safety
    /// Views do not retain allocations. Keep all bindings alive and order accesses
    /// on this stream through completion, including scratch reuse.
    pub(super) unsafe fn linear(
        &mut self,
        input: DMat<BF16>,
        weight: LinearWeight,
        output: DMat<BF16>,
        scratch: Option<&QuantizedScratch>,
        workspace: Workspace,
    ) -> Result<(), Status> {
        let [m, k] = input.shape();
        let shape = match weight {
            LinearWeight::Bf16(w) => w.shape(),
            LinearWeight::Fp8 { weight, .. } => weight.shape(),
            LinearWeight::Nvfp4 { weight, .. } => weight.shape(),
        };
        let [n, weight_k] = shape;
        if k != weight_k || output.shape() != [m, n] {
            return Err(Status::InvalidArgument);
        }
        match weight {
            LinearWeight::Bf16(weight) => unsafe {
                self.qscb().linear(input, weight, output, workspace)
            },
            LinearWeight::Fp8 {
                weight,
                input_scale,
                weight_scale,
            } => {
                let scratch = scratch.ok_or(Status::InternalError)?;
                let x = scratch.fp8.matrix(m, k)?;
                unsafe {
                    self.qscb().quantize_fp8(input, x, input_scale)?;
                    self.qscb().linear_fp8(
                        x,
                        weight,
                        output,
                        [input_scale, weight_scale],
                        workspace,
                    )
                }
            }
            LinearWeight::Nvfp4 {
                weight,
                block_scales,
                quant_multiplier,
                alpha,
            } => {
                let scratch = scratch.ok_or(Status::InternalError)?;
                let plan = scratch.plans.get(&[m, n, k]).ok_or(Status::InternalError)?;
                let x = scratch.fp4.matrix(m, k)?;
                let x_scales = scratch.scales.vector(scale_count(m, k)?)?;
                unsafe {
                    self.qsfi()
                        .nvfp4_quantize(input, x, x_scales, quant_multiplier)?;
                    self.qsfi().nvfp4_execute(
                        plan,
                        x,
                        weight,
                        [x_scales, block_scales],
                        alpha,
                        output,
                        scratch.workspace.workspace(scratch.workspace.len())?,
                    )
                }
            }
        }
    }
}

impl BatchExecution<'_> {
    pub(super) unsafe fn project_logits(
        &mut self,
        ctx: &CudaCtx,
        input: DMat<BF16>,
        weight: LinearWeight,
        output: DMat<F32>,
    ) -> Result<(), Status> {
        match weight {
            LinearWeight::Bf16(weight) => unsafe {
                if let Some(kernel) = self.lm_head {
                    kernel.launch(ctx.stream, input, weight, output)
                } else {
                    self.engine.operators().qscb().linear(
                        input,
                        weight,
                        output,
                        self.linear_workspace,
                    )
                }
            },
            LinearWeight::Nvfp4 { .. } => {
                let scratch = self.quantized_scratch.ok_or(Status::InternalError)?;
                let logits = scratch.logits.matrix(1, output.shape()[1])?;
                unsafe {
                    self.engine.operators().linear(
                        input,
                        weight,
                        logits,
                        self.quantized_scratch,
                        self.linear_workspace,
                    )?;
                    self.engine
                        .operators()
                        .qscu()
                        .logits_bf16_to_f32(logits, output)
                }
            }
            // The supported checkpoint recipe has an NVFP4 or BF16 LM head.
            LinearWeight::Fp8 { .. } => Err(Status::Unsupported),
        }
    }
}
