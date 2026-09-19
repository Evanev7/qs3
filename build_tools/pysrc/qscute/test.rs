// Generated-launcher integration test, run by qscute/test.py.
use dtype::{BF16, DType, F32};
use ffi::DevicePtr;
use qs3::{dtype, ffi};
use std::{ffi::c_void, ptr};

mod f32_kernel {
    include!(concat!(env!("QS3_CUTE_OUTPUT"), "/saxpy_f32.rs"));
}
mod bf16_kernel {
    include!(concat!(env!("QS3_CUTE_OUTPUT"), "/saxpy_bf16.rs"));
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

fn bf16(x: f32) -> u16 {
    let bits = x.to_bits();
    ((bits + 0x7fff + ((bits >> 16) & 1)) >> 16) as u16
}

fn exercise<D: DType, T: Copy + Default + std::fmt::Debug + PartialEq>(
    stream: &Stream,
    launch: impl Fn(DevicePtr<D>, DevicePtr<D>, DevicePtr<D>, i32, f32, *mut c_void),
    encode: impl Fn(f32) -> T,
    decode: impl Fn(T) -> f32,
) {
    for n in [1, 64, 129, 257] {
        let guard = 16 / std::mem::size_of::<T>();
        let mut x = vec![T::default(); n + 2 * guard];
        let mut y = x.clone();
        for i in 0..n {
            x[guard + i] = encode((i % 17) as f32 / 16.0 - 0.5);
            y[guard + i] = encode((i % 13) as f32 / 8.0 - 0.75);
        }
        let dx = Buffer::upload(&x);
        let dy = Buffer::upload(&y);
        let bytes = std::mem::size_of_val(&*x);
        let intermediate = Buffer::new(bytes);
        let output = Buffer::new(bytes);
        for buffer in [&intermediate, &output] {
            assert_eq!(unsafe { cudaMemset(buffer.0, 0xa5, bytes) }, 0);
        }
        // Finish fixture initialization before using the nonblocking stream.
        assert_eq!(unsafe { cudaStreamSynchronize(ptr::null_mut()) }, 0);
        launch(
            dx.data(),
            dy.data(),
            intermediate.data(),
            n as i32,
            1.5,
            stream.0,
        );
        launch(
            intermediate.data(),
            dy.data(),
            output.data(),
            n as i32,
            -0.5,
            stream.0,
        );
        assert_eq!(unsafe { cudaStreamSynchronize(stream.0) }, 0);
        for (buffer, second) in [(&intermediate, false), (&output, true)] {
            let mut result = vec![T::default(); x.len()];
            assert_eq!(
                unsafe { cudaMemcpy(result.as_mut_ptr().cast(), buffer.0, bytes, 2) },
                0
            );
            let raw = unsafe { std::slice::from_raw_parts(result.as_ptr().cast::<u8>(), bytes) };
            assert!(raw[..16].iter().all(|&b| b == 0xa5));
            assert!(raw[bytes - 16..].iter().all(|&b| b == 0xa5));
            for i in guard..guard + n {
                let first = encode(1.5 * decode(x[i]) + decode(y[i]));
                let expected = if second {
                    encode(-0.5 * decode(first) + decode(y[i]))
                } else {
                    first
                };
                assert_eq!(
                    result[i],
                    expected,
                    "n={n}, index={}, second={second}",
                    i - guard
                );
            }
        }
    }
}

fn assert_duplicate_load_panics() {
    let panic = std::panic::catch_unwind(|| unsafe { f32_kernel::Kernel::load() })
        .err()
        .expect("duplicate load must panic");
    let message = panic
        .downcast_ref::<&str>()
        .copied()
        .or_else(|| panic.downcast_ref::<String>().map(String::as_str));
    assert_eq!(
        message,
        Some("CuTe specialization qscute_saxpy_f32 is already loaded"),
    );
}

#[test]
fn typed_launches_stream_ordering_and_reload() {
    let _context = Buffer::new(1);
    let stream = Stream::new();
    for _ in 0..2 {
        let f32_kernel = unsafe { f32_kernel::Kernel::load().unwrap() };
        let bf16_kernel = unsafe { bf16_kernel::Kernel::load().unwrap() };
        assert_duplicate_load_panics();
        std::thread::spawn(assert_duplicate_load_panics)
            .join()
            .unwrap();
        exercise::<F32, f32>(
            &stream,
            |x, y, output, n, alpha, stream| unsafe {
                f32_kernel.launch(x, y, output, n, alpha, stream).unwrap();
            },
            |x| x,
            |x| x,
        );
        exercise::<BF16, u16>(
            &stream,
            |x, y, output, n, alpha, stream| unsafe {
                bf16_kernel.launch(x, y, output, n, alpha, stream).unwrap();
            },
            bf16,
            |x| f32::from_bits((x as u32) << 16),
        );
    }
}
