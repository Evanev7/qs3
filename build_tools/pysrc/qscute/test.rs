// Generated-launcher integration test, run by qscute/test.py.
use dtype::{DType, F32, Fp8E4M3};
use ffi::DevicePtr;
use qs3::{dtype, ffi};
use std::{ffi::c_void, ptr};

mod fp8_kernel {
    include!(concat!(env!("QS3_CUTE_OUTPUT"), "/fp8_decode_test.rs"));
}

#[link(name = "cudart")]
unsafe extern "C" {
    fn cudaMalloc(ptr: *mut *mut c_void, size: usize) -> i32;
    fn cudaFree(ptr: *mut c_void) -> i32;
    fn cudaMemcpy(dst: *mut c_void, src: *const c_void, bytes: usize, kind: i32) -> i32;
    fn cudaMemset(ptr: *mut c_void, value: i32, size: usize) -> i32;
    fn cudaStreamCreateWithFlags(stream: *mut *mut c_void, flags: u32) -> i32;
    fn cudaStreamSynchronize(stream: *mut c_void) -> i32;
    fn cudaStreamDestroy(stream: *mut c_void) -> i32;
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
    fn data<D: DType>(&self) -> DevicePtr<D> {
        // Leave a 16-byte guard while preserving the promised pointer alignment.
        DevicePtr::new(unsafe { self.0.cast::<u8>().add(16).cast() }).unwrap()
    }
}
impl Drop for Buffer {
    fn drop(&mut self) {
        assert_eq!(unsafe { cudaFree(self.0) }, 0);
    }
}

struct Stream(*mut c_void);
impl Stream {
    fn new() -> Self {
        let mut stream = ptr::null_mut();
        assert_eq!(unsafe { cudaStreamCreateWithFlags(&mut stream, 1) }, 0);
        Self(stream)
    }
}
impl Drop for Stream {
    fn drop(&mut self) {
        assert_eq!(unsafe { cudaStreamDestroy(self.0) }, 0);
    }
}

fn fp8(code: u8) -> f32 {
    let exponent = (code >> 3) & 15;
    let mantissa = code & 7;
    let value = if exponent == 0 {
        f32::from(mantissa) / 512.0
    } else {
        (1.0 + f32::from(mantissa) / 8.0) * 2f32.powi(i32::from(exponent) - 7)
    };
    if code & 128 == 0 { value } else { -value }
}

fn exercise(stream: &Stream, kernel: &fp8_kernel::Kernel) {
    let n = fp8_kernel::constants::N as usize;
    let k = fp8_kernel::constants::K as usize;
    let mut x = vec![0u8; k + 32];
    let mut w = vec![0u8; n * k + 32];
    for j in 0..k {
        x[16 + j] = 0x30 + (j % 16) as u8;
    }
    for row in 0..n {
        for j in 0..k {
            w[16 + row * k + j] = 0x28 + ((row * 7 + j / 32) % 32) as u8;
        }
    }
    let dx = Buffer::upload(&x);
    let dw = Buffer::upload(&w);
    let bytes = 2 * n * 4 + 32;
    let first = Buffer::new(bytes);
    let second = Buffer::new(bytes);
    for [xs, ws] in [[0.125f32, 0.5f32], [0.37, 0.019]] {
        let sx = Buffer::upload(&[0., 0., 0., 0., xs]);
        let sw = Buffer::upload(&[0., 0., 0., 0., ws]);
        for buffer in [&first, &second] {
            assert_eq!(unsafe { cudaMemset(buffer.0, 0xa5, bytes) }, 0);
        }
        assert_eq!(unsafe { cudaStreamSynchronize(ptr::null_mut()) }, 0);
        unsafe {
            kernel
                .launch(
                    dx.data::<Fp8E4M3>(),
                    dw.data::<Fp8E4M3>(),
                    first.data::<F32>(),
                    sx.data::<F32>(),
                    sw.data::<F32>(),
                    stream.0,
                )
                .unwrap();
            // Depend on the first launch's FP32 result as the next launch's input scale.
            kernel
                .launch(
                    dx.data::<Fp8E4M3>(),
                    dw.data::<Fp8E4M3>(),
                    second.data::<F32>(),
                    first.data::<F32>(),
                    sw.data::<F32>(),
                    stream.0,
                )
                .unwrap();
        }
        assert_eq!(unsafe { cudaStreamSynchronize(stream.0) }, 0);
        let sum = |row: usize, split: usize| -> f32 {
            (split * k / 2..(split + 1) * k / 2)
                .map(|j| fp8(x[16 + j]) * fp8(w[16 + row * k + j]))
                .sum()
        };
        let first_scale = sum(0, 0) * (xs * ws);
        for (buffer, alpha) in [(&first, xs * ws), (&second, first_scale * ws)] {
            let mut result = vec![0u8; bytes];
            assert_eq!(
                unsafe { cudaMemcpy(result.as_mut_ptr().cast(), buffer.0, bytes, 2) },
                0
            );
            assert!(result[..16].iter().all(|&b| b == 0xa5));
            assert!(result[bytes - 16..].iter().all(|&b| b == 0xa5));
            for split in 0..2 {
                for row in 0..n {
                    let offset = 16 + (split * n + row) * 4;
                    let actual = f32::from_ne_bytes(result[offset..offset + 4].try_into().unwrap());
                    assert_eq!(
                        actual,
                        sum(row, split) * alpha,
                        "row={row} split={split} alpha={alpha}"
                    );
                }
            }
        }
    }
}

fn assert_duplicate_load_panics() {
    let panic = std::panic::catch_unwind(|| unsafe { fp8_kernel::Kernel::load() })
        .err()
        .expect("duplicate load must panic");
    let message = panic
        .downcast_ref::<&str>()
        .copied()
        .or_else(|| panic.downcast_ref::<String>().map(String::as_str));
    assert_eq!(
        message,
        Some("CuTe specialization qscute_fp8_decode_test is already loaded")
    );
}

#[test]
fn typed_launches_stream_ordering_and_reload() {
    let _context = Buffer::new(1);
    let stream = Stream::new();
    for _ in 0..2 {
        let kernel = unsafe { fp8_kernel::Kernel::load().unwrap() };
        assert_duplicate_load_panics();
        std::thread::spawn(assert_duplicate_load_panics)
            .join()
            .unwrap();
        exercise(&stream, &kernel);
    }
}
