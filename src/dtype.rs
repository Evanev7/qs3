#![allow(dead_code)]
use crate::Status;
use crate::ffi::{
    DTYPE_BF16, DTYPE_F16, DTYPE_F32, DTYPE_FP8_E4M3, DTYPE_FP8_E5M2, DTYPE_I8, DTYPE_I32,
    DTYPE_MXFP4_E2M1, DTYPE_MXFP8_E4M3, DTYPE_NVFP4_E2M1, DTYPE_U8, DTYPE_U32, DTypeRaw,
};
pub unsafe trait DType: Copy + 'static {
    const RAW: DTypeRaw;
    const BITS: usize;
    const ALIGN: usize = (Self::BITS + 7) / 8;
    #[inline(always)]
    fn size_of(elements: usize) -> Result<usize, Status> {
        elements
            .checked_mul(Self::BITS)
            .filter(|bits| bits.is_multiple_of(8))
            .map(|bits| bits / 8)
            .ok_or(Status::InvalidArgument)
    }
    #[inline(always)]
    fn len_of(bytes: usize) -> Result<usize, Status> {
        bytes
            .checked_mul(8)
            .filter(|bits| bits.is_multiple_of(Self::BITS))
            .map(|bits| bits / Self::BITS)
            .ok_or(Status::InvalidArgument)
    }
}

macro_rules! impl_dtype {
    ($t:ident, $e:expr, $bits:literal) => {
        #[derive(Clone, Copy, Debug, PartialEq, Eq)]
        pub struct $t;
        unsafe impl DType for $t {
            const RAW: DTypeRaw = $e;
            const BITS: usize = $bits;
        }
    };
}
impl_dtype!(F32, DTYPE_F32, 32);
impl_dtype!(F16, DTYPE_F16, 16);
impl_dtype!(BF16, DTYPE_BF16, 16);
impl_dtype!(Fp8E4M3, DTYPE_FP8_E4M3, 8);
impl_dtype!(Mxfp8E4M3, DTYPE_MXFP8_E4M3, 8);
impl_dtype!(Fp8E5M2, DTYPE_FP8_E5M2, 8);
impl_dtype!(Nvfp4E2M1, DTYPE_NVFP4_E2M1, 4);
impl_dtype!(Mxfp4E2M1, DTYPE_MXFP4_E2M1, 4);
impl_dtype!(I32, DTYPE_I32, 32);
impl_dtype!(U32, DTYPE_U32, 32);
impl_dtype!(I8, DTYPE_I8, 8);
impl_dtype!(U8, DTYPE_U8, 8);

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DynDType {
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

impl DynDType {
    pub fn bits(self) -> usize {
        match self {
            DynDType::F32 | DynDType::I32 | DynDType::U32 => 32,
            DynDType::F16 | DynDType::BF16 => 16,
            DynDType::FP8E4M3
            | DynDType::FP8E5M2
            | DynDType::MXFP8E4M3
            | DynDType::I8
            | DynDType::U8 => 8,
            DynDType::NVFP4E2M1 | DynDType::MXFP4E2M1 => 4,
        }
    }

    pub fn storage_bytes_for(&self, elements: usize) -> Result<usize, Status> {
        elements
            .checked_mul(self.bits())
            .filter(|bits| bits.is_multiple_of(8))
            .map(|bits| bits / 8)
            .ok_or(Status::InvalidArgument)
    }
    pub fn storage_bytes_for_shape(&self, shape: &[u32]) -> Result<usize, Status> {
        let elements = shape.iter().try_fold(1usize, |acc, dim| {
            acc.checked_mul(*dim as usize)
                .ok_or(Status::InvalidArgument)
        })?;
        self.storage_bytes_for(elements)
    }

    pub(crate) fn is_runtime_supported(self) -> bool {
        matches!(self, DynDType::F16 | DynDType::BF16)
    }

    pub(crate) fn to_raw(self) -> crate::ffi::DTypeRaw {
        match self {
            DynDType::F32 => crate::ffi::DTYPE_F32,
            DynDType::F16 => crate::ffi::DTYPE_F16,
            DynDType::BF16 => crate::ffi::DTYPE_BF16,
            DynDType::FP8E4M3 => crate::ffi::DTYPE_FP8_E4M3,
            DynDType::FP8E5M2 => crate::ffi::DTYPE_FP8_E5M2,
            DynDType::NVFP4E2M1 => crate::ffi::DTYPE_NVFP4_E2M1,
            DynDType::MXFP4E2M1 => crate::ffi::DTYPE_MXFP4_E2M1,
            DynDType::MXFP8E4M3 => crate::ffi::DTYPE_MXFP8_E4M3,
            DynDType::I32 => crate::ffi::DTYPE_I32,
            DynDType::U32 => crate::ffi::DTYPE_U32,
            DynDType::I8 => crate::ffi::DTYPE_I8,
            DynDType::U8 => crate::ffi::DTYPE_U8,
        }
    }
}
