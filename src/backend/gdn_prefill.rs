//! Qwen-only 64-token GDN prefill. Modules and execution are separate from scratch storage.
use crate::{
    Status,
    backend::{Bf16Heads, DMat, DVec},
    constants::{
        GdnRecurrentDType,
        gdn::{
            KEY_HEAD_DIM as K, NUM_KEY_HEADS as HG, NUM_VALUE_HEADS as H,
            PACKED_QKV_CHANNELS as PACKED, VALUE_HEAD_DIM as V,
        },
    },
    dtype::{BF16, F32, I32},
    ffi,
    memory::{CudaCtx, DeviceBuffer},
};
use std::rc::Rc;
mod prep {
    include!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/build/triton/gdn_prefill_prep.rs"
    ));
}
mod cumsum {
    include!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/build/triton/gdn_prefill_cumsum.rs"
    ));
}
mod kkt {
    include!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/build/triton/gdn_prefill_kkt.rs"
    ));
}
mod solve {
    include!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/build/triton/gdn_prefill_solve.rs"
    ));
}
mod wu {
    include!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/build/triton/gdn_prefill_wu.rs"
    ));
}
mod state {
    include!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/build/triton/gdn_prefill_state.rs"
    ));
}
mod output {
    include!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/build/triton/gdn_prefill_output.rs"
    ));
}

pub(crate) struct GdnPrefill {
    prep: prep::Kernel,
    cumsum: cumsum::Kernel,
    kkt: kkt::Kernel,
    solve: solve::Kernel,
    wu: wu::Kernel,
    state: state::Kernel,
    output: output::Kernel,
}

/// Reusable allocations only; no kernel modules or execution methods.
pub(crate) struct GdnPrefillScratch {
    g: DeviceBuffer<F32>,
    beta: DeviceBuffer<F32>,
    gc: DeviceBuffer<F32>,
    matrix: DeviceBuffer<F32>,
    inverse: DeviceBuffer<BF16>,
    w: DeviceBuffer<BF16>,
    u: DeviceBuffer<BF16>,
    histories: DeviceBuffer<BF16>,
    new_v: DeviceBuffer<BF16>,
    unused: DeviceBuffer<I32>,
    capacity: u32,
}

fn count(rows: u32, width: u32) -> Result<usize, Status> {
    let n = rows
        .checked_mul(width)
        .filter(|&n| n <= i32::MAX as u32)
        .ok_or(Status::InvalidArgument)?;
    Ok(n as usize)
}

impl GdnPrefillScratch {
    pub(crate) fn new(ctx: Rc<CudaCtx>, rows: u32) -> Result<Self, Status> {
        let mut result = Self {
            g: DeviceBuffer::with_capacity(ctx.clone(), 1)?,
            beta: DeviceBuffer::with_capacity(ctx.clone(), 1)?,
            gc: DeviceBuffer::with_capacity(ctx.clone(), 1)?,
            matrix: DeviceBuffer::with_capacity(ctx.clone(), 1)?,
            inverse: DeviceBuffer::with_capacity(ctx.clone(), 1)?,
            w: DeviceBuffer::with_capacity(ctx.clone(), 1)?,
            u: DeviceBuffer::with_capacity(ctx.clone(), 1)?,
            histories: DeviceBuffer::with_capacity(ctx.clone(), 1)?,
            new_v: DeviceBuffer::with_capacity(ctx.clone(), 1)?,
            unused: DeviceBuffer::with_capacity(ctx.clone(), 1)?,
            capacity: 0,
        };
        result.reserve(rows)?;
        Ok(result)
    }

    pub(crate) fn reserve(&mut self, rows: u32) -> Result<(), Status> {
        if rows == 0 {
            return Err(Status::InvalidArgument);
        }
        if rows <= self.capacity {
            return Ok(());
        }
        let heads = count(rows, H)?;
        let matrix = count(rows, H * 64)?;
        let vectors = count(rows, H * V)?;
        let histories = count(rows.div_ceil(64), H * V * K)?;
        self.g.realloc(heads)?;
        self.beta.realloc(heads)?;
        self.gc.realloc(heads)?;
        self.matrix.realloc(matrix)?;
        self.inverse.realloc(matrix)?;
        self.w.realloc(vectors)?;
        self.u.realloc(vectors)?;
        self.new_v.realloc(vectors)?;
        self.histories.realloc(histories)?;
        self.capacity = rows;
        Ok(())
    }
}

impl GdnPrefill {
    /// Load modules in the current context before capture or execution.
    /// The caller retains that context until all module uses have completed.
    pub(crate) unsafe fn load() -> Result<Self, Status> {
        Ok(Self {
            prep: unsafe { prep::Kernel::load() }.map_err(|_| Status::CudaError)?,
            cumsum: unsafe { cumsum::Kernel::load() }.map_err(|_| Status::CudaError)?,
            kkt: unsafe { kkt::Kernel::load() }.map_err(|_| Status::CudaError)?,
            solve: unsafe { solve::Kernel::load() }.map_err(|_| Status::CudaError)?,
            wu: unsafe { wu::Kernel::load() }.map_err(|_| Status::CudaError)?,
            state: unsafe { state::Kernel::load() }.map_err(|_| Status::CudaError)?,
            output: unsafe { output::Kernel::load() }.map_err(|_| Status::CudaError)?,
        })
    }

    /// All descriptors must refer to live, nonaliasing allocations in this context.
    /// Input/output state are distinct live/staged slots; commit belongs to the runner.
    pub(crate) unsafe fn run(
        &self,
        scratch: &mut GdnPrefillScratch,
        stream: ffi::CudaStream,
        conv: DMat<BF16>,
        a: DMat<BF16>,
        b: DMat<BF16>,
        alog: DVec<BF16>,
        dt: DVec<BF16>,
        q: Bf16Heads,
        k: Bf16Heads,
        v: Bf16Heads,
        initial: DMat<GdnRecurrentDType>,
        final_state: DMat<GdnRecurrentDType>,
        out: Bf16Heads,
    ) -> Result<(), Status> {
        let rows = conv.rows;
        if rows == 0
            || rows > scratch.capacity
            || conv.cols != PACKED
            || a.shape() != [rows, H]
            || b.shape() != [rows, H]
            || alog.len != H
            || dt.len != H
            || initial.shape() != [H * V, K]
            || final_state.shape() != [H * V, K]
            || initial.data == final_state.data
        {
            return Err(Status::InvalidArgument);
        }
        conv.require_contiguous()?;
        a.require_contiguous()?;
        b.require_contiguous()?;
        alog.require_contiguous()?;
        dt.require_contiguous()?;
        initial.require_contiguous()?;
        final_state.require_contiguous()?;
        for (tensor, heads, width) in [(q, HG, K), (k, HG, K), (v, H, V), (out, H, V)] {
            if tensor.tokens != rows
                || tensor.heads != heads
                || tensor.head_dim != width
                || tensor.token_stride != heads * width
                || tensor.head_stride != width
            {
                return Err(Status::InvalidArgument);
            }
        }
        // solve writes only the lower block triangle, just as upstream.
        scratch.inverse.zero()?;
        let t = rows as i32;
        let nt = rows.div_ceil(64);
        let unused = scratch.unused.as_ptr();
        let map = |_| Status::CudaError;
        unsafe {
            self.prep
                .launch(
                    stream,
                    [rows.div_ceil(16), HG + H, 1],
                    conv.data,
                    a.data,
                    b.data,
                    alog.data,
                    dt.data,
                    q.data,
                    k.data,
                    v.data,
                    scratch.g.as_ptr(),
                    scratch.beta.as_ptr(),
                    t,
                )
                .map_err(map)?;
            self.cumsum
                .launch(
                    stream,
                    [nt, H, 1],
                    scratch.g.as_ptr(),
                    scratch.gc.as_ptr(),
                    unused,
                    unused,
                    t,
                )
                .map_err(map)?;
            self.kkt
                .launch(
                    stream,
                    [nt, H, 1],
                    k.data,
                    scratch.beta.as_ptr(),
                    scratch.gc.as_ptr(),
                    scratch.matrix.as_ptr(),
                    unused,
                    unused,
                    t,
                )
                .map_err(map)?;
            self.solve
                .launch(
                    stream,
                    [nt, H, 1],
                    scratch.matrix.as_ptr(),
                    scratch.inverse.as_ptr(),
                    unused,
                    unused,
                    t,
                )
                .map_err(map)?;
            self.wu
                .launch(
                    stream,
                    [nt, H, 1],
                    k.data,
                    v.data,
                    scratch.beta.as_ptr(),
                    scratch.w.as_ptr(),
                    scratch.u.as_ptr(),
                    scratch.inverse.as_ptr(),
                    scratch.gc.as_ptr(),
                    unused,
                    unused,
                    t,
                )
                .map_err(map)?;
            self.state
                .launch(
                    stream,
                    [V.div_ceil(64), H, 1],
                    k.data,
                    scratch.u.as_ptr(),
                    scratch.w.as_ptr(),
                    scratch.new_v.as_ptr(),
                    scratch.gc.as_ptr(),
                    scratch.gc.as_ptr(),
                    scratch.histories.as_ptr(),
                    initial.data,
                    final_state.data,
                    unused,
                    unused,
                    t,
                )
                .map_err(map)?;
            self.output
                .launch(
                    stream,
                    [V.div_ceil(64), nt, H],
                    q.data,
                    k.data,
                    scratch.new_v.as_ptr(),
                    scratch.histories.as_ptr(),
                    scratch.gc.as_ptr(),
                    out.data,
                    unused,
                    unused,
                    1.0 / (K as f32).sqrt(),
                    t,
                )
                .map_err(map)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{dtype::DType, memory::HostBuffer};

    fn bf16(x: f32) -> u16 {
        let bits = x.to_bits();
        ((bits + 0x7fff + ((bits >> 16) & 1)) >> 16) as u16
    }
    fn buffer<D: DType>(ctx: &Rc<CudaCtx>, values: &[f32]) -> DeviceBuffer<D> {
        let mut host = HostBuffer::<D>::new(values.len()).unwrap();
        for (bytes, &x) in host.as_mut().chunks_exact_mut(D::BITS / 8).zip(values) {
            match D::BITS {
                16 => bytes.copy_from_slice(&bf16(x).to_ne_bytes()),
                32 => bytes.copy_from_slice(&x.to_ne_bytes()),
                _ => panic!("test requires float storage"),
            }
        }
        host.upload(ctx.clone()).unwrap()
    }
    fn read<D: DType>(ctx: &CudaCtx, input: &DeviceBuffer<D>, len: usize) -> Vec<f32> {
        let mut host = HostBuffer::<D>::new(len).unwrap();
        unsafe { input.download(&mut host).unwrap() };
        ctx.synchronize().unwrap();
        host.as_ref()
            .chunks_exact(D::BITS / 8)
            .map(|bytes| match D::BITS {
                16 => {
                    f32::from_bits(u32::from(u16::from_ne_bytes(bytes.try_into().unwrap())) << 16)
                }
                32 => f32::from_ne_bytes(bytes.try_into().unwrap()),
                _ => unreachable!(),
            })
            .collect()
    }

    #[test]
    fn chunk64_boundaries_seeded_state_and_reused_scratch() {
        let ctx = Rc::new(CudaCtx::default().unwrap());
        let kernel = unsafe { GdnPrefill::load().unwrap() };
        let mut scratch = GdnPrefillScratch::new(ctx.clone(), 1).unwrap();
        assert_eq!(scratch.reserve(0), Err(Status::InvalidArgument));
        assert_eq!(scratch.reserve(u32::MAX), Err(Status::InvalidArgument));
        let alog = buffer::<BF16>(&ctx, &vec![-100.; H as usize]);
        let dt = buffer::<BF16>(&ctx, &vec![0.; H as usize]);
        let state_len = (H * V * K) as usize;
        let initial = buffer::<GdnRecurrentDType>(&ctx, &vec![0.25; state_len]);
        let final_state = buffer::<GdnRecurrentDType>(&ctx, &vec![-7.; state_len]);
        for rows in [1, 63, 64, 65, 129, 4, 128] {
            scratch.reserve(rows).unwrap();
            let mut packed = vec![0.; (rows * PACKED) as usize];
            for token in packed.chunks_exact_mut(PACKED as usize) {
                for head in 0..HG {
                    token[(head * K) as usize] = 1.;
                    token[((HG + head) * K) as usize] = 1.;
                }
                token[(2 * HG * K) as usize..].fill(0.5);
            }
            let conv = buffer::<BF16>(&ctx, &packed);
            let a = buffer::<BF16>(&ctx, &vec![0.; (rows * H) as usize]);
            let q = buffer::<BF16>(&ctx, &vec![0.; (rows * HG * K) as usize]);
            let k = buffer::<BF16>(&ctx, &vec![0.; (rows * HG * K) as usize]);
            let v = buffer::<BF16>(&ctx, &vec![0.; (rows * H * V) as usize]);
            let out = buffer::<BF16>(&ctx, &vec![-9.; (rows * H * V) as usize]);
            let input_state = initial.matrix(H * V, K).unwrap();
            let launch = |scratch: &mut GdnPrefillScratch, output_state| unsafe {
                kernel.run(
                    scratch,
                    ctx.stream,
                    conv.matrix(rows, PACKED).unwrap(),
                    a.matrix(rows, H).unwrap(),
                    a.matrix(rows, H).unwrap(),
                    alog.vector(H).unwrap(),
                    dt.vector(H).unwrap(),
                    q.heads(rows, HG, K).unwrap(),
                    k.heads(rows, HG, K).unwrap(),
                    v.heads(rows, H, V).unwrap(),
                    input_state,
                    output_state,
                    out.heads(rows, H, V).unwrap(),
                )
            };
            assert_eq!(
                launch(&mut scratch, input_state),
                Err(Status::InvalidArgument)
            );
            launch(&mut scratch, final_state.matrix(H * V, K).unwrap()).unwrap();
            let output = read(&ctx, &out, (rows * H * V) as usize);
            for (t, row) in output.chunks_exact((H * V) as usize).enumerate() {
                // k=e0, beta=1/2, exp(g)=1: s_t = 1/2 - 1/4 * 2^-(t+1).
                let expected = bf16((0.5 - 0.25 * 0.5f32.powi(t as i32 + 1)) / (K as f32).sqrt());
                assert!(
                    row.iter().all(|&x| bf16(x) == expected),
                    "output at rows={rows}, t={t}"
                );
            }
            let state = read(&ctx, &final_state, state_len);
            for (i, &x) in state.iter().enumerate() {
                let expected = if i % K as usize == 0 {
                    0.5 - 0.25 * 0.5f32.powi(rows as i32)
                } else {
                    0.25
                };
                assert!(
                    (x - expected).abs() < 0.000002,
                    "state rows={rows}, i={i}: {x} vs {expected}"
                );
            }
            assert!(read(&ctx, &initial, state_len).iter().all(|&x| x == 0.25));
        }
    }
}
