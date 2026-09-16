use crate::dtype::DType;
use std::{ffi::c_void, fmt, marker::PhantomData, ptr::NonNull};

/// A non-owning non-null device address with a known GPU dtype.
///
/// The brand distinguishes device addresses from host pointers. It does not
/// establish allocation validity, initialization, lifetime, device identity, or
/// stream ordering. Creating a pointer only constructs metadata;
/// callers of unsafe device operations must establish those properties.
/// This type deliberately provides no host dereference or ownership operations.
#[repr(transparent)]
pub struct DevicePtr<D: DType>(NonNull<u8>, PhantomData<D>);

impl<T: DType> DevicePtr<T> {
    pub fn new(raw: *mut u8) -> Option<Self> {
        NonNull::new(raw).map(|ptr| Self(ptr, PhantomData))
    }
    pub const fn as_raw(self) -> *mut u8 {
        self.0.as_ptr()
    }

    /// Returns a raw, type-erased address for an FFI call or descriptor.
    pub const fn erase(self) -> *mut c_void {
        self.0.as_ptr().cast()
    }
}

// Pointer metadata is copyable/comparable regardless of the pointee's traits.
impl<T: DType> Copy for DevicePtr<T> {}
impl<T: DType> Clone for DevicePtr<T> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<T: DType> PartialEq for DevicePtr<T> {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}
impl<T: DType> Eq for DevicePtr<T> {}
impl<T: DType> fmt::Debug for DevicePtr<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Pointer::fmt(&self.0, f)
    }
}
