use crate::{Status, ffi};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct DVec<const DTYPE: ffi::DTypeRaw> {
    pub(super) data: ffi::DevicePtr,
    pub(super) len: u32,
    stride: u32,
}

impl<const DT: ffi::DTypeRaw> DVec<DT> {
    pub(crate) fn contiguous(data: ffi::DevicePtr, len: u32) -> Result<Self, Status> {
        Self::new(data, len, 1)
    }

    pub(crate) fn new(data: ffi::DevicePtr, len: u32, stride: u32) -> Result<Self, Status> {
        validate_ptr(data)?;
        validate_nonzero(&[len, stride])?;
        Ok(Self { data, len, stride })
    }

    pub(super) fn tensor(self) -> ffi::Tensor1 {
        ffi::Tensor1 {
            data: self.data,
            dtype: DT,
            shape: [self.len.into()],
            stride: [self.stride.into()],
        }
    }

    pub(super) fn is_contiguous(self) -> bool {
        self.stride == 1
    }

    pub(super) fn require_contiguous(&self) -> Result<(), Status> {
        if !self.is_contiguous() {
            Err(Status::InvalidArgument)
        } else {
            Ok(())
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct DMat<const DTYPE: ffi::DTypeRaw> {
    data: ffi::DevicePtr,
    pub(super) rows: u32,
    pub(super) cols: u32,
    row_stride: u32,
}

impl<const DT: ffi::DTypeRaw> DMat<DT> {
    pub(crate) fn contiguous(data: ffi::DevicePtr, rows: u32, cols: u32) -> Result<Self, Status> {
        Self::new(data, rows, cols, cols)
    }

    pub(crate) fn new(
        data: ffi::DevicePtr,
        rows: u32,
        cols: u32,
        row_stride: u32,
    ) -> Result<Self, Status> {
        validate_ptr(data)?;
        validate_nonzero(&[rows, cols, row_stride])?;
        if row_stride < cols {
            return Err(Status::InvalidArgument);
        }
        Ok(Self {
            data,
            rows,
            cols,
            row_stride,
        })
    }

    pub(super) fn tensor(self) -> ffi::Tensor2 {
        ffi::Tensor2 {
            data: self.data,
            dtype: DT,
            shape: [self.rows.into(), self.cols.into()],
            stride: [self.row_stride.into(), 1],
        }
    }

    pub(super) fn same_shape<const T: ffi::DTypeRaw>(self, other: DMat<T>) -> bool {
        self.rows == other.rows && self.cols == other.cols
    }

    pub(super) fn is_contiguous(self) -> bool {
        self.row_stride == self.cols
    }

    pub(super) fn require_contiguous(&self) -> Result<(), Status> {
        if !self.is_contiguous() {
            Err(Status::InvalidArgument)
        } else {
            Ok(())
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct DTensor3<const DTYPE: ffi::DTypeRaw> {
    data: ffi::DevicePtr,
    dim0: u32,
    dim1: u32,
    dim2: u32,
    stride0: u32,
    stride1: u32,
    stride2: u32,
}

impl<const DT: ffi::DTypeRaw> DTensor3<DT> {
    pub(crate) fn contiguous(
        data: ffi::DevicePtr,
        dim0: u32,
        dim1: u32,
        dim2: u32,
    ) -> Result<Self, Status> {
        let stride1 = dim2;
        let stride0 = dim1.checked_mul(dim2).ok_or(Status::InvalidArgument)?;
        Self::new(data, dim0, dim1, dim2, stride0, stride1, 1)
    }

    pub(crate) fn new(
        data: ffi::DevicePtr,
        dim0: u32,
        dim1: u32,
        dim2: u32,
        stride0: u32,
        stride1: u32,
        stride2: u32,
    ) -> Result<Self, Status> {
        validate_ptr(data)?;
        validate_nonzero(&[dim0, dim1, dim2, stride0, stride1, stride2])?;
        if stride2 != 1
            || stride1 < dim2
            || stride0 < dim1.checked_mul(stride1).ok_or(Status::InvalidArgument)?
        {
            return Err(Status::InvalidArgument);
        }
        Ok(Self {
            data,
            dim0,
            dim1,
            dim2,
            stride0,
            stride1,
            stride2,
        })
    }

    pub(super) fn tensor(self) -> ffi::Tensor3 {
        ffi::Tensor3 {
            data: self.data,
            dtype: DT,
            shape: [self.dim0.into(), self.dim1.into(), self.dim2.into()],
            stride: [
                self.stride0.into(),
                self.stride1.into(),
                self.stride2.into(),
            ],
        }
    }

    fn is_contiguous(self) -> bool {
        self.stride2 == 1
            && self.stride1 == self.dim2
            && self.dim1.checked_mul(self.dim2) == Some(self.stride0)
    }

    pub(super) fn require_contiguous(&self) -> Result<(), Status> {
        if !self.is_contiguous() {
            Err(Status::InvalidArgument)
        } else {
            Ok(())
        }
    }
}

fn validate_ptr(data: ffi::DevicePtr) -> Result<(), Status> {
    if data.is_null() {
        return Err(Status::InvalidArgument);
    }
    Ok(())
}

fn validate_nonzero(values: &[u32]) -> Result<(), Status> {
    if values.contains(&0) {
        return Err(Status::InvalidArgument);
    }
    Ok(())
}
