//! GB10 small-batch W4A4 projections; existing 128x4 block-scale layout.
use crate::{
    Status,
    backend::{DMat, DVec},
    dtype::{BF16, F32, Fp8E4M3, Nvfp4E2M1},
    ffi,
};
mod up {
    include!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/build/cute/nvfp4_up.rs"
    ));
}
mod down {
    include!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/build/cute/nvfp4_down.rs"
    ));
}
mod head {
    include!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/build/cute/nvfp4_head.rs"
    ));
}

pub(crate) struct Nvfp4Linears {
    up: up::Kernel,
    down: down::Kernel,
    head: head::Kernel,
}
impl Nvfp4Linears {
    pub(crate) const MAX_ROWS: u32 = 16;
    pub(crate) const TACTIC: &str = "cute_32x64x512";
    pub(crate) const HIDDEN: u32 = up::constants::K as u32;
    pub(crate) const INTERMEDIATE: u32 = up::constants::N as u32;
    pub(crate) const VOCAB: u32 = head::constants::N as u32;
    const _SHAPES: () = {
        assert!(down::constants::N == up::constants::K);
        assert!(down::constants::K == up::constants::N);
        assert!(head::constants::K == up::constants::K);
    };
    pub(crate) fn supports_model(hidden: u32, intermediate: u32, vocab: u32) -> bool {
        [hidden, intermediate, vocab] == [Self::HIDDEN, Self::INTERMEDIATE, Self::VOCAB]
    }
    pub(crate) fn supports([m, n, k]: [u32; 3]) -> bool {
        (1..=Self::MAX_ROWS).contains(&m)
            && matches!(
                (n, k),
                (Self::INTERMEDIATE, Self::HIDDEN)
                    | (Self::HIDDEN, Self::INTERMEDIATE)
                    | (Self::VOCAB, Self::HIDDEN)
            )
    }
    /// Keep the current context alive until all launches and this owner complete.
    /// A second live owner of these exports panics.
    pub(crate) unsafe fn load() -> Result<Self, Status> {
        let () = Self::_SHAPES;
        unsafe {
            Ok(Self {
                up: up::Kernel::load().map_err(|_| Status::CudaError)?,
                down: down::Kernel::load().map_err(|_| Status::CudaError)?,
                head: head::Kernel::load().map_err(|_| Status::CudaError)?,
            })
        }
    }
    /// All views must remain live, nonaliasing, and ordered on the loading context's stream.
    pub(crate) unsafe fn launch(
        &self,
        stream: ffi::CudaStream,
        input: DMat<Nvfp4E2M1>,
        weight: DMat<Nvfp4E2M1>,
        scales: [DVec<Fp8E4M3>; 2],
        alpha: DVec<F32>,
        output: DMat<BF16>,
    ) -> Result<(), Status> {
        let [m, k] = input.shape();
        let [n, wk] = weight.shape();
        if !Self::supports([m, n, k])
            || wk != k
            || output.shape() != [m, n]
            || scales[0].len != m.div_ceil(128) * 128 * (k / 16)
            || scales[1].len != n.div_ceil(128) * 128 * (k / 16)
            || alpha.len != 1
        {
            return Err(Status::InvalidArgument);
        }
        input.require_contiguous()?;
        weight.require_contiguous()?;
        output.require_contiguous()?;
        for scale in scales {
            scale.require_contiguous()?;
        }
        alpha.require_contiguous()?;
        for address in [
            input.data.erase(),
            weight.data.erase(),
            scales[0].data.erase(),
            scales[1].data.erase(),
            output.data.erase(),
        ] {
            if !(address as usize).is_multiple_of(16) {
                return Err(Status::InvalidArgument);
            }
        }
        if !(alpha.data.erase() as usize).is_multiple_of(4) {
            return Err(Status::InvalidArgument);
        }
        let status = unsafe {
            match (n, k) {
                (Self::INTERMEDIATE, Self::HIDDEN) => self.up.launch(
                    input.data,
                    weight.data,
                    scales[0].data,
                    scales[1].data,
                    output.data,
                    alpha.data,
                    m as i32,
                    stream,
                ),
                (Self::HIDDEN, Self::INTERMEDIATE) => self.down.launch(
                    input.data,
                    weight.data,
                    scales[0].data,
                    scales[1].data,
                    output.data,
                    alpha.data,
                    m as i32,
                    stream,
                ),
                (Self::VOCAB, Self::HIDDEN) => self.head.launch(
                    input.data,
                    weight.data,
                    scales[0].data,
                    scales[1].data,
                    output.data,
                    alpha.data,
                    m as i32,
                    stream,
                ),
                _ => unreachable!(),
            }
        };
        status.map_err(|_| Status::CudaError)
    }
}
