use super::*;
use crate::{
    backend::{Qscb, qstriton::Fp8Reduce},
    dtype::{BF16, DType, U8},
    memory::{CudaCtx, DeviceBuffer, HostBuffer},
};
use std::rc::Rc;

fn device<D: DType>(ctx: &Rc<CudaCtx>, bytes: &[u8]) -> DeviceBuffer<D> {
    let mut host = HostBuffer::<D>::new(D::len_of(bytes.len()).unwrap()).unwrap();
    host.as_mut().copy_from_slice(bytes);
    host.upload(ctx.clone()).unwrap()
}
fn download<D: DType>(ctx: &CudaCtx, device: &DeviceBuffer<D>) -> Vec<u8> {
    let mut host = HostBuffer::<D>::new(device.len()).unwrap();
    unsafe {
        device.download(&mut host).unwrap();
    }
    ctx.synchronize().unwrap();
    host.as_ref().to_vec()
}
fn fp8(code: u8) -> f32 {
    let e = (code >> 3) & 15;
    let m = code & 7;
    let v = if e == 0 {
        f32::from(m) / 512.0
    } else {
        (1.0 + f32::from(m) / 8.0) * 2f32.powi(i32::from(e) - 7)
    };
    if code & 128 == 0 { v } else { -v }
}
fn duplicate_load_panics() {
    let panic = std::panic::catch_unwind(|| unsafe { fp8_decode::Kernel::load() })
        .err()
        .expect("duplicate load must panic");
    let message = panic
        .downcast_ref::<&str>()
        .copied()
        .or_else(|| panic.downcast_ref::<String>().map(String::as_str));
    assert_eq!(
        message,
        Some("CuTe specialization qscute_fp8_decode is already loaded")
    );
}

#[test]
fn fp8_decode_scales_split_partials_guards_stream_ordering_and_reload() {
    let _kernel_owner = TEST_LOCK.lock().unwrap();
    let ctx = Rc::new(CudaCtx::new(0).unwrap());
    let mut qscb = Qscb::new(&ctx).unwrap();
    let (n, k) = (Fp8Decode::N, Fp8Decode::K);
    let mut rng = 91u64;
    let weights: Vec<u8> = (0..n * k)
        .map(|_| {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            // Varied signs and exponents, with products exactly representable in FP32.
            24 + (rng % 32) as u8 | if rng & 256 != 0 { 128 } else { 0 }
        })
        .collect();
    let weight = device::<Fp8E4M3>(&ctx, &weights);
    let workspace = DeviceBuffer::<U8>::with_capacity(ctx.clone(), 32 << 20).unwrap();
    let reference = DeviceBuffer::<BF16>::with_capacity(ctx.clone(), n as usize).unwrap();
    for reload in 0..2 {
        let kernel = unsafe { Fp8Decode::load().unwrap() };
        let reduce = unsafe { Fp8Reduce::load().unwrap() };
        duplicate_load_panics();
        std::thread::spawn(duplicate_load_panics).join().unwrap();
        for [xs, ws] in [[0.125f32, 0.5f32], [0.37, 0.019], [0.0625, 0.125]] {
            let values: Vec<u8> = (0..k)
                .flat_map(|i| {
                    let value = ((i * 37 + reload * 11) % 97) as f32 / 32.0 - 1.5;
                    ((value.to_bits() >> 16) as u16).to_ne_bytes()
                })
                .collect();
            let input = device::<BF16>(&ctx, &values);
            let x = DeviceBuffer::<Fp8E4M3>::with_capacity(ctx.clone(), k as usize).unwrap();
            let input_scale = device::<F32>(&ctx, &xs.to_ne_bytes());
            let weight_scale = device::<F32>(&ctx, &ws.to_ne_bytes());
            let partials = device::<F32>(&ctx, &vec![0xa5; 2 * n as usize * 4 + 32]);
            let output = device::<BF16>(&ctx, &vec![0xa5; n as usize * 2 + 32]);
            let p = ffi::DevicePtr::new(unsafe { partials.as_raw().add(16) }).unwrap();
            let out = ffi::DevicePtr::new(unsafe { output.as_raw().add(16) }).unwrap();
            let scales = [
                input_scale.vector(1).unwrap(),
                weight_scale.vector(1).unwrap(),
            ];
            unsafe {
                // No synchronization between quantization, CuTe, reduction and cuBLASLt.
                qscb.quantize_fp8(
                    input.matrix(1, k).unwrap(),
                    x.matrix(1, k).unwrap(),
                    scales[0],
                )
                .unwrap();
                kernel
                    .launch(
                        ctx.stream,
                        x.matrix(1, k).unwrap(),
                        weight.matrix(n, k).unwrap(),
                        scales,
                        DMat::contiguous(p, 2, n).unwrap(),
                    )
                    .unwrap();
                reduce
                    .launch(
                        ctx.stream,
                        DMat::contiguous(p, 2, n).unwrap(),
                        DMat::contiguous(out, 1, n).unwrap(),
                    )
                    .unwrap();
                qscb.linear_fp8(
                    x.matrix(1, k).unwrap(),
                    weight.matrix(n, k).unwrap(),
                    reference.matrix(1, n).unwrap(),
                    scales,
                    workspace.workspace(workspace.len()).unwrap(),
                )
                .unwrap();
            }
            let actual = download(&ctx, &output);
            let partial = download(&ctx, &partials);
            for bytes in [&actual, &partial] {
                assert!(bytes[..16].iter().all(|b| *b == 0xa5));
                assert!(bytes[bytes.len() - 16..].iter().all(|b| *b == 0xa5));
            }
            let expected = download(&ctx, &reference);
            let actual: Vec<f32> = actual[16..16 + n as usize * 2]
                .chunks_exact(2)
                .map(|b| f32::from_bits(u32::from(u16::from_ne_bytes(b.try_into().unwrap())) << 16))
                .collect();
            let expected: Vec<f32> = expected
                .chunks_exact(2)
                .map(|b| f32::from_bits(u32::from(u16::from_ne_bytes(b.try_into().unwrap())) << 16))
                .collect();
            let error: f64 = actual
                .iter()
                .zip(&expected)
                .map(|(a, b)| f64::from(a - b).powi(2))
                .sum();
            let norm: f64 = expected.iter().map(|v| f64::from(*v).powi(2)).sum();
            assert!(
                (error / norm).sqrt() < 0.001,
                "relative L2={} xs={xs} ws={ws}",
                (error / norm).sqrt()
            );
            assert!(actual.iter().all(|v| v.is_finite()));
            // Independently check both split-K outputs, including the last column.
            let quantized = download(&ctx, &x);
            for column in [0, 1, 63, 64, 127, 128, n - 1] {
                for split in 0..2 {
                    let sum: f32 = (split * k / 2..(split + 1) * k / 2)
                        .map(|j| {
                            fp8(quantized[j as usize]) * fp8(weights[(column * k + j) as usize])
                        })
                        .sum();
                    let offset = 16 + ((split * n + column) * 4) as usize;
                    let value = f32::from_ne_bytes(partial[offset..offset + 4].try_into().unwrap());
                    assert_eq!(value, sum * (xs * ws), "split={split} column={column}");
                }
            }
            // Descriptors reject incompatible shape/alignment before a device launch.
            unsafe {
                assert_eq!(
                    kernel.launch(
                        ctx.stream,
                        x.matrix(1, k).unwrap(),
                        weight.matrix(n, k).unwrap(),
                        scales,
                        DMat::contiguous(p, 1, n).unwrap(),
                    ),
                    Err(Status::InvalidArgument)
                );
                let bad = ffi::DevicePtr::new(x.as_raw().add(1)).unwrap();
                assert_eq!(
                    kernel.launch(
                        ctx.stream,
                        DMat::contiguous(bad, 1, k).unwrap(),
                        weight.matrix(n, k).unwrap(),
                        scales,
                        DMat::contiguous(p, 2, n).unwrap(),
                    ),
                    Err(Status::InvalidArgument)
                );
                assert_eq!(
                    reduce.launch(
                        ctx.stream,
                        DMat::contiguous(p, 1, n).unwrap(),
                        DMat::contiguous(out, 1, n).unwrap(),
                    ),
                    Err(Status::InvalidArgument)
                );
                assert_eq!(
                    reduce.launch(
                        ctx.stream,
                        DMat::contiguous(p, 2, n).unwrap(),
                        DMat::contiguous(out, 1, n - 1).unwrap(),
                    ),
                    Err(Status::InvalidArgument)
                );
            }
        }
    }
}
