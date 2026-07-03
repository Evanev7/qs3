use std::{collections::HashSet, hash::Hash};

use crate::Status;

pub(crate) struct OOM;
impl From<OOM> for Status {
    fn from(_value: OOM) -> Self {
        Status::OutOfMemory
    }
}

pub(crate) trait SafeVec {
    fn safe_reserve(&mut self, additional: usize) -> Result<(), OOM>;
    fn safe_reserve_exact(&mut self, additional: usize) -> Result<(), OOM>;
    fn safe_new(capacity: usize) -> Result<Self, OOM>
    where
        Self: Sized;
}
impl<T> SafeVec for Vec<T> {
    fn safe_reserve(&mut self, additional: usize) -> Result<(), OOM> {
        self.try_reserve(additional).map_err(|_| OOM)
    }
    fn safe_reserve_exact(&mut self, additional: usize) -> Result<(), OOM> {
        self.try_reserve_exact(additional).map_err(|_| OOM)
    }
    fn safe_new(capacity: usize) -> Result<Self, OOM> {
        let mut s = Self::default();
        s.safe_reserve_exact(capacity)?;
        Ok(s)
    }
}

pub(crate) fn try_clone_slice<T: Copy>(slice: &[T]) -> Result<Vec<T>, Status> {
    let mut out = Vec::safe_new(slice.len())?;
    out.extend_from_slice(slice);
    Ok(out)
}

pub(crate) trait SafeHashSet {
    fn safe_reserve(&mut self, additional: usize) -> Result<(), OOM>;
    fn safe_new(capacity: usize) -> Result<Self, OOM>
    where
        Self: Sized;
}
impl<T: Eq + Hash> SafeHashSet for HashSet<T> {
    fn safe_reserve(&mut self, additional: usize) -> Result<(), OOM> {
        self.try_reserve(additional).map_err(|_| OOM)
    }
    fn safe_new(capacity: usize) -> Result<Self, OOM> {
        let mut s = Self::default();
        s.safe_reserve(capacity)?;
        Ok(s)
    }
}
