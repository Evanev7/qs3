// Standalone generated-launcher integration test, run by run_cuda_test.sh.
use std::{ffi::c_void, ptr};

mod full {
    include!(concat!(env!("QS3_TRITON_OUTPUT"), "/full/lm_head.rs"));
}
mod tail {
    include!(concat!(env!("QS3_TRITON_OUTPUT"), "/tail/lm_head.rs"));
}

#[link(name = "cudart")]
unsafe extern "C" {
    fn cudaMalloc(ptr: *mut *mut c_void, size: usize) -> i32;
    fn cudaFree(ptr: *mut c_void) -> i32;
    fn cudaMemcpy(dst: *mut c_void, src: *const c_void, bytes: usize, kind: i32) -> i32;
    fn cudaMemset(ptr: *mut c_void, value: i32, size: usize) -> i32;
}

struct Buffer(*mut c_void);
impl Buffer {
    fn new(bytes: usize) -> Self {
        let mut p = ptr::null_mut();
        assert_eq!(unsafe { cudaMalloc(&mut p, bytes) }, 0);
        Self(p)
    }
    fn upload<T>(values: &[T]) -> Self {
        let bytes = std::mem::size_of_val(values);
        let buffer = Self::new(bytes);
        assert_eq!(
            unsafe { cudaMemcpy(buffer.0, values.as_ptr().cast(), bytes, 1) },
            0
        );
        buffer
    }
}
impl Drop for Buffer {
    fn drop(&mut self) {
        assert_eq!(unsafe { cudaFree(self.0) }, 0);
    }
}

fn bf16(x: f32) -> u16 {
    // Fixture values are exactly representable. Outputs use round-to-nearest-even.
    let bits = x.to_bits();
    ((bits + 0x7fff + ((bits >> 16) & 1)) >> 16) as u16
}

fn exercise<T: Copy + Default>(
    k: usize,
    n: usize,
    launch: impl FnOnce(*mut u16, *mut u16, *mut T),
    decode: impl Fn(T) -> f32,
    round_reference: impl Fn(f32) -> f32,
) {
    let mut x: Vec<u16> = (0..k)
        .map(|col| bf16((col % 7) as f32 / 8.0 - 0.375))
        .collect();
    x[0] = bf16(1.0); // Break periodic cancellation at the full model width.
    let base: Vec<f32> = (0..k).map(|col| (col % 13) as f32 / 16.0 - 0.375).collect();
    let dot: f64 = x
        .iter()
        .zip(&base)
        .map(|(&x, &w)| f32::from_bits((x as u32) << 16) as f64 * w as f64)
        .sum();
    assert_ne!(dot, 0.0);
    let mut weights = Vec::with_capacity(n * k);
    for row in 0..n {
        let factor = (row % 17) as f32 - 8.0;
        weights.extend(base.iter().map(|&w| bf16(w * factor)));
    }
    let x = Buffer::upload(&x);
    let w = Buffer::upload(&weights);
    drop(weights);
    // Sentinel slots check output bounds, including the last grid row.
    let y = Buffer::new((n + 2) * std::mem::size_of::<T>());
    assert_eq!(
        unsafe { cudaMemset(y.0, 0xa5, (n + 2) * std::mem::size_of::<T>()) },
        0
    );
    launch(x.0.cast(), w.0.cast(), unsafe { y.0.cast::<T>().add(1) });
    let mut result = vec![T::default(); n + 2];
    assert_eq!(
        unsafe {
            cudaMemcpy(
                result.as_mut_ptr().cast(),
                y.0,
                std::mem::size_of_val(&*result),
                2,
            )
        },
        0
    );
    let raw = unsafe {
        std::slice::from_raw_parts(
            result.as_ptr().cast::<u8>(),
            std::mem::size_of_val(&*result),
        )
    };
    let width = std::mem::size_of::<T>();
    assert!(raw[..width].iter().all(|&b| b == 0xa5));
    assert!(raw[raw.len() - width..].iter().all(|&b| b == 0xa5));
    for row in 0..n {
        let expected = round_reference((dot * ((row % 17) as f64 - 8.0)) as f32);
        assert_eq!(decode(result[row + 1]), expected, "row {row}");
    }
}

#[test]
fn full_lm_head_f32() {
    let k = env!("QS3_TRITON_K").parse().unwrap();
    // A CUDA runtime allocation establishes the primary context before load.
    let _context = Buffer::new(1);
    let kernel = unsafe { full::Kernel::load().unwrap() };
    exercise(
        k,
        full::GRID[0] as usize,
        |x, w, y| unsafe { kernel.launch(ptr::null_mut(), x, w, y).unwrap() },
        |x| x,
        |x| x,
    );
}

#[test]
fn masked_tail_bf16() {
    let _context = Buffer::new(1);
    let kernel = unsafe { tail::Kernel::load().unwrap() };
    exercise(
        93,
        tail::GRID[0] as usize,
        |x, w, y| unsafe { kernel.launch(ptr::null_mut(), x, w, y).unwrap() },
        |x| f32::from_bits((x as u32) << 16),
        |x| f32::from_bits((bf16(x) as u32) << 16),
    );
}
