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

fn nvfp4_scale_offset(row: usize, group: usize, k: usize) -> usize {
    ((row / 128 * (k / 64) + group / 4) * 32 + row % 32) * 16 + (row % 128) / 32 * 4 + group % 4
}
fn nvfp4_fixture(rows: usize, k: usize, seed: u32) -> (Vec<u8>, Vec<u8>) {
    let hash = |mut x: u32| {
        x ^= x >> 16;
        x = x.wrapping_mul(0x7feb352d);
        x ^= x >> 15;
        x = x.wrapping_mul(0x846ca68b);
        x ^ (x >> 16)
    };
    let values = (0..rows * k / 2)
        .map(|i| hash(i as u32 ^ seed) as u8)
        .collect();
    let mut scales = vec![0; rows.div_ceil(128) * 128 * k / 16];
    for row in 0..rows {
        for block in 0..k / 16 {
            scales[nvfp4_scale_offset(row, block, k)] =
                0x20 + (hash((row * k / 16 + block) as u32 ^ seed) % 40) as u8;
        }
    }
    (values, scales)
}
fn nvfp4_value(q: &[u8], sf: &[u8], row: usize, col: usize, k: usize) -> f64 {
    let i = row * k + col;
    let code = (q[i / 2] >> (4 * (i % 2))) & 15;
    let value = [0., 0.5, 1., 1.5, 2., 3., 4., 6.][usize::from(code & 7)];
    (if code & 8 == 0 { value } else { -value })
        * f64::from(fp8(sf[nvfp4_scale_offset(row, col / 16, k)]))
}
#[test]
fn nvfp4_small_batches_scales_guards_and_binding_validation() {
    use crate::{
        backend::{DMat, Qsfi, qsfi::Nvfp4Tactic},
        dtype::Nvfp4E2M1,
        ffi::{DevicePtr, cuda},
    };
    let _owner = TEST_LOCK.lock().unwrap();
    let ctx = Rc::new(CudaCtx::new(0).unwrap());
    let kernel = unsafe { Nvfp4Linears::load().unwrap() };
    let mut baseline = Qsfi::new(&ctx).unwrap();
    for (n, k) in [
        (Nvfp4Linears::INTERMEDIATE, Nvfp4Linears::HIDDEN),
        (Nvfp4Linears::HIDDEN, Nvfp4Linears::INTERMEDIATE),
        (Nvfp4Linears::VOCAB, Nvfp4Linears::HIDDEN),
    ] {
        let (w, sf) = nvfp4_fixture(n as usize, k as usize, 17);
        let weight = device::<Nvfp4E2M1>(&ctx, &w);
        let weight_scale = device::<Fp8E4M3>(&ctx, &sf);
        for m in [1u32, 2, 16] {
            let (x, xsf) = nvfp4_fixture(m as usize, k as usize, 43);
            let input = device::<Nvfp4E2M1>(&ctx, &x);
            let input_scale = device::<Fp8E4M3>(&ctx, &xsf);
            let count = (m * n) as usize;
            let output = DeviceBuffer::<U8>::with_capacity(ctx.clone(), count * 2 + 32).unwrap();
            let reference = DeviceBuffer::<BF16>::with_capacity(ctx.clone(), count).unwrap();
            let plan = baseline
                .create_nvfp4_plan([m, n, k], Nvfp4Tactic::Tile128x32Dp)
                .unwrap();
            let workspace =
                DeviceBuffer::<U8>::with_capacity(ctx.clone(), plan.workspace_bytes.max(256))
                    .unwrap();
            let scales = [
                input_scale.vector(input_scale.len() as u32).unwrap(),
                weight_scale.vector(weight_scale.len() as u32).unwrap(),
            ];
            let result = DMat::contiguous(
                DevicePtr::<BF16>::new(unsafe { output.as_raw().add(16) }).unwrap(),
                m,
                n,
            )
            .unwrap();
            for alpha in [1f32, 0.137] {
                let alpha_device = device::<F32>(&ctx, &alpha.to_le_bytes());
                unsafe {
                    assert_eq!(
                        cuda::cudaMemsetAsync(
                            output.as_raw().cast(),
                            0xa5,
                            count * 2 + 32,
                            ctx.stream
                        ),
                        0
                    );
                    let misaligned = DMat::contiguous(
                        DevicePtr::<Nvfp4E2M1>::new(input.as_raw().add(1)).unwrap(),
                        m,
                        k,
                    )
                    .unwrap();
                    assert_eq!(
                        kernel.launch(
                            ctx.stream,
                            misaligned,
                            weight.matrix(n, k).unwrap(),
                            scales,
                            alpha_device.vector(1).unwrap(),
                            result
                        ),
                        Err(Status::InvalidArgument)
                    );
                    let short_scales = [input_scale.vector(scales[0].len - 1).unwrap(), scales[1]];
                    assert_eq!(
                        kernel.launch(
                            ctx.stream,
                            input.matrix(m, k).unwrap(),
                            weight.matrix(n, k).unwrap(),
                            short_scales,
                            alpha_device.vector(1).unwrap(),
                            result
                        ),
                        Err(Status::InvalidArgument)
                    );
                    kernel
                        .launch(
                            ctx.stream,
                            input.matrix(m, k).unwrap(),
                            weight.matrix(n, k).unwrap(),
                            scales,
                            alpha_device.vector(1).unwrap(),
                            result,
                        )
                        .unwrap();
                    baseline
                        .nvfp4_execute(
                            &plan,
                            input.matrix(m, k).unwrap(),
                            weight.matrix(n, k).unwrap(),
                            scales,
                            alpha_device.vector(1).unwrap(),
                            reference.matrix(m, n).unwrap(),
                            workspace.workspace(workspace.len()).unwrap(),
                        )
                        .unwrap();
                }
                let actual = download(&ctx, &output);
                let expected = download(&ctx, &reference);
                assert!(
                    actual[..16]
                        .iter()
                        .chain(&actual[actual.len() - 16..])
                        .all(|&b| b == 0xa5)
                );
                assert_eq!(
                    &actual[16..16 + count * 2],
                    expected.as_slice(),
                    "M{m} N{n} K{k} alpha={alpha}"
                );
                for row in [0, m as usize - 1] {
                    for col in [0, 63, 64, n as usize - 1] {
                        let exact = (0..k as usize)
                            .map(|j| {
                                nvfp4_value(&x, &xsf, row, j, k as usize)
                                    * nvfp4_value(&w, &sf, col, j, k as usize)
                            })
                            .sum::<f64>()
                            * f64::from(alpha);
                        let offset = 2 * (row * n as usize + col);
                        let got = f32::from_bits(
                            u32::from(u16::from_le_bytes(
                                expected[offset..offset + 2].try_into().unwrap(),
                            )) << 16,
                        );
                        assert!((f64::from(got) - exact).abs() <= 0.004 * exact.abs() + 0.02);
                    }
                }
            }
        }
    }
}
