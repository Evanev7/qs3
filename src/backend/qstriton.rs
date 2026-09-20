use crate::{
    Status,
    backend::DMat,
    dtype::{BF16, F32},
    ffi,
};

pub(crate) mod sampling {
    pub(crate) mod prepare {
        include!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/build/triton/sampling_prepare.rs"
        ));
    }
    pub(crate) mod filter {
        include!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/build/triton/sampling_filter.rs"
        ));
    }
    pub(crate) mod gumbel {
        include!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/build/triton/sampling_gumbel.rs"
        ));
    }
    pub(crate) mod reduce {
        include!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/build/triton/sampling_reduce.rs"
        ));
    }
}

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

mod fp8_reduce {
    include!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/build/triton/fp8_reduce.rs"
    ));
}

pub(crate) struct Fp8Reduce(fp8_reduce::Kernel);

impl Fp8Reduce {
    pub(crate) const N: u32 = fp8_reduce::constants::N as u32;

    pub(crate) unsafe fn load() -> Result<Self, Status> {
        unsafe { fp8_reduce::Kernel::load() }
            .map(Self)
            .map_err(|_| Status::CudaError)
    }

    /// Retain the module and nonaliasing tensor allocations until work completes.
    pub(crate) unsafe fn launch(
        &self,
        stream: ffi::CudaStream,
        partials: DMat<F32>,
        output: DMat<BF16>,
    ) -> Result<(), Status> {
        if partials.shape() != [2, Self::N] || output.shape() != [1, Self::N] {
            return Err(Status::InvalidArgument);
        }
        partials.require_contiguous()?;
        output.require_contiguous()?;
        if !(partials.data.erase() as usize).is_multiple_of(4)
            || !(output.data.erase() as usize).is_multiple_of(2)
        {
            return Err(Status::InvalidArgument);
        }
        unsafe { self.0.launch(stream, partials.data, output.data) }.map_err(|_| Status::CudaError)
    }
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
                unsafe { self.0.launch(stream, input.data, weight.data, output.data) }
                    .map_err(|_| Status::CudaError)
            }
        }
    };
}

qstriton_gemv!(GdnQkv, gdn_qkv, BF16, BF16, BF16);
qstriton_gemv!(LmHead, lm_head, BF16, BF16, F32);
