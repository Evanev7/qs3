#![allow(dead_code)]
use crate::ffi::{
    DTYPE_BF16, DTYPE_F16, DTYPE_F32, DTYPE_FP8_E4M3, DTYPE_FP8_E5M2, DTYPE_I8, DTYPE_I32,
    DTYPE_MXFP4_E2M1, DTYPE_MXFP8_E4M3, DTYPE_NVFP4_E2M1, DTYPE_U8, DTYPE_U32, DTypeRaw,
};
pub trait DType: Copy {
    const RAW: DTypeRaw;
}

macro_rules! impl_dtype {
    ($t:ident, $e:expr) => {
        #[derive(Clone, Copy, Debug, PartialEq, Eq)]
        pub struct $t;
        impl DType for $t {
            const RAW: DTypeRaw = $e;
        }
    };
}
impl_dtype!(F32, DTYPE_F32);
impl_dtype!(F16, DTYPE_F16);
impl_dtype!(BF16, DTYPE_BF16);
impl_dtype!(Fp8E4M3, DTYPE_FP8_E4M3);
impl_dtype!(Mxfp8E4M3, DTYPE_MXFP8_E4M3);
impl_dtype!(Fp8E5M2, DTYPE_FP8_E5M2);
impl_dtype!(Nvfp4E2M1, DTYPE_NVFP4_E2M1);
impl_dtype!(Mxfp4E2M1, DTYPE_MXFP4_E2M1);
impl_dtype!(I32, DTYPE_I32);
impl_dtype!(U32, DTYPE_U32);
impl_dtype!(I8, DTYPE_I8);
impl_dtype!(U8, DTYPE_U8);

/// Floating storage accepted by mixed BF16/FP32 qwen kernels.
pub(crate) trait FloatDType: DType {}
impl FloatDType for BF16 {}
impl FloatDType for F32 {}
