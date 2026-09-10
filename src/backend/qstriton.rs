use super::{BF16, DMat, F32};
use crate::{Status, ffi};

mod lm_head {
    include!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/build/triton/lm_head.rs"
    ));
}

/// The Nix-selected LM-head specialization, owned by the runner's CUDA context.
pub(crate) struct LmHead(lm_head::Kernel);

impl LmHead {
    pub(crate) fn supports(hidden: u32, vocab: u32) -> bool {
        i64::from(hidden) == lm_head::constants::K && vocab == lm_head::GRID[0]
    }

    /// The runner's context must remain current through launch and destruction.
    pub(crate) unsafe fn load() -> Result<Self, Status> {
        unsafe { lm_head::Kernel::load() }
            .map(Self)
            .map_err(|_| Status::CudaError)
    }

    /// Retain the module and nonaliasing tensor allocations until work completes.
    pub(crate) unsafe fn launch(
        &self,
        stream: ffi::CudaStream,
        input: DMat<BF16>,
        weight: DMat<BF16>,
        output: DMat<F32>,
    ) -> Result<(), Status> {
        if input.rows != 1
            || output.rows != 1
            || input.cols != weight.cols
            || output.cols != weight.rows
            || !Self::supports(weight.cols, weight.rows)
        {
            return Err(Status::InvalidArgument);
        }
        input.require_contiguous()?;
        weight.require_contiguous()?;
        output.require_contiguous()?;
        unsafe {
            self.0.launch(
                stream,
                input.tensor().data.cast(),
                weight.tensor().data.cast(),
                output.tensor().data.cast(),
            )
        }
        .map_err(|_| Status::CudaError)
    }
}
