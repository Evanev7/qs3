#![allow(dead_code)]

use crate::Status;

use super::{self as ffi, sys};

use std::{
    mem::MaybeUninit,
    ptr::{self, NonNull},
};

pub(crate) type ContextDesc = sys::qscb_context_desc;
pub(crate) type Bf16GemmDesc = sys::qscb_bf16_gemm_desc;

pub(crate) struct Context {
    raw: NonNull<sys::qscb_context>,
}

impl Context {
    pub(crate) fn new(device_ordinal: i32, stream: ffi::CudaStream) -> Result<Self, Status> {
        let desc = ContextDesc {
            device_ordinal,
            stream,
        };
        Self::from_desc(&desc)
    }

    pub(crate) fn from_desc(desc: &ContextDesc) -> Result<Self, Status> {
        let mut raw = ptr::null_mut();
        ffi::result_from_raw(unsafe { sys::qscb_context_create(desc, &mut raw) })?;
        let raw = NonNull::new(raw).ok_or(Status::InternalError)?;
        Ok(Self { raw })
    }

    pub(crate) unsafe fn gemm_bf16(&mut self, desc: &Bf16GemmDesc) -> Result<(), Status> {
        ffi::result_from_raw(unsafe { sys::qscb_gemm_bf16(self.raw.as_ptr(), desc) })
    }

    pub(crate) fn last_error(&self) -> Result<ffi::ErrorInfo, Status> {
        let mut out = MaybeUninit::uninit();
        ffi::result_from_raw(unsafe {
            sys::qscb_context_get_last_error(self.raw.as_ptr(), out.as_mut_ptr())
        })?;
        Ok(unsafe { out.assume_init() })
    }

    pub(crate) fn clear_last_error(&mut self) {
        unsafe {
            sys::qscb_context_clear_last_error(self.raw.as_ptr());
        }
    }
}

impl Drop for Context {
    fn drop(&mut self) {
        unsafe {
            sys::qscb_context_destroy(self.raw.as_ptr());
        }
    }
}
