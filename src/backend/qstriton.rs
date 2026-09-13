use super::{BF16, DMat, F32};
use crate::{Status, ffi};

mod lm_head {
    include!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/build/triton/lm_head.rs"
    ));
}

mod gdn_qkv {
    include!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/build/triton/gdn_qkv.rs"
    ));
}

macro_rules! qstriton_gemv {
    ($name:ident,$k:ident,$input:ty, $weight:ty, $output:ty) => {
        pub(crate) struct $name($k::Kernel);

        impl $name {
            pub(crate) fn supports(hidden: u32, vocab: u32) -> bool {
                i64::from(hidden) == $k::constants::K && vocab == $k::GRID[0]
            }

            pub(crate) unsafe fn load() -> Result<Self, Status> {
                unsafe { $k::Kernel::load() }
                    .map(Self)
                    .map_err(|_| Status::CudaError)
            }

            /// Retain the module and nonaliasing tensor allocations until work completes.
            pub(crate) unsafe fn launch(
                &self,
                stream: ffi::CudaStream,
                input: DMat<$input>,
                weight: DMat<$weight>,
                output: DMat<$output>,
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
    };
}

qstriton_gemv!(GdnQkv, gdn_qkv, BF16, BF16, BF16);
qstriton_gemv!(LmHead, lm_head, BF16, BF16, F32);
