// Included in a disposable qs3 source overlay: production Qsfi baseline and memory.
use crate::{
    backend::{
        Qsfi,
        qsfi::{Nvfp4Tactic, scale_count},
    },
    dtype::{BF16, DType, F32, Fp8E4M3, Nvfp4E2M1, U8},
    ffi::{self, DevicePtr, cuda},
    memory::{CudaCtx, DeviceBuffer, HostBuffer},
};
use std::{ffi::c_void, fs::File, io::Write, rc::Rc, time::Instant};

unsafe extern "C" {
    fn cudaDeviceGetAttribute(value: *mut i32, attribute: i32, device: i32) -> i32;
    fn cudaEventElapsedTime(ms: *mut f32, a: *mut c_void, b: *mut c_void) -> i32;
}
struct Events(*mut c_void, *mut c_void);
impl Events {
    fn new() -> Self {
        let mut a = std::ptr::null_mut();
        let mut b = std::ptr::null_mut();
        unsafe {
            assert_eq!(cuda::cudaEventCreateWithFlags(&mut a, 0), 0);
            assert_eq!(cuda::cudaEventCreateWithFlags(&mut b, 0), 0);
        }
        Self(a, b)
    }
    fn sample(&self, ctx: &CudaCtx, mut run: impl FnMut()) -> (f64, f64) {
        unsafe {
            assert_eq!(cuda::cudaEventRecord(self.0, ctx.stream), 0);
        }
        let start = Instant::now();
        run();
        let enqueue = start.elapsed().as_secs_f64() * 1e6;
        unsafe {
            assert_eq!(cuda::cudaEventRecord(self.1, ctx.stream), 0);
            assert_eq!(cuda::cudaEventSynchronize(self.1), 0);
        }
        let mut ms = 0f32;
        unsafe {
            assert_eq!(cudaEventElapsedTime(&mut ms, self.0, self.1), 0);
        }
        (f64::from(ms) * 1000.0, enqueue)
    }
}
impl Drop for Events {
    fn drop(&mut self) {
        unsafe {
            cuda::cudaEventDestroy(self.0);
            cuda::cudaEventDestroy(self.1);
        }
    }
}
fn device<D: DType>(ctx: &Rc<CudaCtx>, bytes: &[u8]) -> DeviceBuffer<D> {
    let mut h = HostBuffer::<D>::new(D::len_of(bytes.len()).unwrap()).unwrap();
    h.as_mut().copy_from_slice(bytes);
    h.upload(ctx.clone()).unwrap()
}
fn download<D: DType>(ctx: &CudaCtx, buffer: &DeviceBuffer<D>) -> Vec<u8> {
    let mut h = HostBuffer::<D>::new(buffer.len()).unwrap();
    unsafe {
        buffer.download(&mut h).unwrap();
    }
    ctx.synchronize().unwrap();
    h.as_ref().to_vec()
}
fn sf_offset(row: usize, group: usize, k: usize) -> usize {
    ((row / 128 * (k / 64) + group / 4) * 32 + row % 32) * 16 + (row % 128) / 32 * 4 + group % 4
}
fn hash(mut x: u32) -> u32 {
    x ^= x >> 16;
    x = x.wrapping_mul(0x7feb352d);
    x ^= x >> 15;
    x = x.wrapping_mul(0x846ca68b);
    x ^ (x >> 16)
}
fn fixture(rows: usize, k: usize, seed: u32) -> (Vec<u8>, Vec<u8>) {
    let q = (0..rows * k / 2)
        .map(|i| hash((i as u32) ^ seed) as u8)
        .collect();
    let mut sf = vec![0u8; scale_count(rows as u32, k as u32).unwrap() as usize];
    for row in 0..rows {
        for group in 0..k / 16 {
            sf[sf_offset(row, group, k)] =
                0x20 + (hash((row * k / 16 + group) as u32 ^ seed) % 40) as u8;
        }
    }
    (q, sf)
}
fn fp8(x: u8) -> f64 {
    let e = (x >> 3) & 15;
    let m = x & 7;
    if e == 0 {
        f64::from(m) / 512.0
    } else {
        (1. + f64::from(m) / 8.) * 2f64.powi(i32::from(e) - 7)
    }
}
fn value(q: &[u8], sf: &[u8], row: usize, col: usize, k: usize) -> f64 {
    let i = row * k + col;
    let code = (q[i / 2] >> (4 * (i % 2))) & 15;
    let v = [0., 0.5, 1., 1.5, 2., 3., 4., 6.][usize::from(code & 7)];
    (if code & 8 == 0 { v } else { -v }) * fp8(sf[sf_offset(row, col / 16, k)])
}
fn bf16(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(2)
        .map(|b| f32::from_bits(u32::from(u16::from_le_bytes(b.try_into().unwrap())) << 16))
        .collect()
}
fn metrics(a: &[f32], b: &[f32]) -> (usize, f64, f64) {
    assert_eq!(a.len(), b.len());
    assert!(a.iter().all(|x| x.is_finite()));
    let mut changed = 0;
    let mut err = 0.;
    let mut norm = 0.;
    let mut max = 0f64;
    for (&a, &b) in a.iter().zip(b) {
        let d = f64::from(a) - f64::from(b);
        changed += usize::from(a != b);
        err += d * d;
        norm += f64::from(b).powi(2);
        max = max.max(d.abs());
    }
    (changed, (err / norm.max(1e-30)).sqrt(), max)
}
struct Args {
    a: *mut u8,
    b: *mut u8,
    sfa: *mut u8,
    sfb: *mut u8,
    out: *mut u8,
    partials: *mut u8,
    alpha: *mut u8,
    m: i32,
    stream: ffi::CudaStream,
}
struct Candidate {
    name: &'static str,
    n: u32,
    k: u32,
    splits: u32,
    run: Box<dyn Fn(&Args) -> Result<(), i32>>,
}
// The builder appends generated module includes and candidate constructors.

#[test]
#[ignore]
fn nvfp4_decode_candidates() {
    let ctx = Rc::new(CudaCtx::new(0).unwrap());
    // CUDA device attributes: multiprocessor count, compute capability major/minor.
    for (attribute, expected) in [(16, 48), (75, 12), (76, 1)] {
        let mut value = 0;
        unsafe {
            assert_eq!(cudaDeviceGetAttribute(&mut value, attribute, 0), 0);
        }
        assert_eq!(
            value, expected,
            "probe is specialized for GB10 SM121 with 48 SMs"
        );
    }
    let mut baseline = Qsfi::new(&ctx).unwrap();
    let candidates = load_candidates();
    let reducer = unsafe { reduction::Kernel::load().unwrap() };
    let events = Events::new();
    let mut eviction = DeviceBuffer::<U8>::with_capacity(ctx.clone(), 128 << 20).unwrap();
    let out_dir = std::path::PathBuf::from(std::env::var_os("NVFP4_OUTPUT").unwrap());
    let mut log = File::create(out_dir.join("results.jsonl")).unwrap();
    let seed: u32 = std::env::var("NVFP4_SEED")
        .unwrap_or("17".into())
        .parse()
        .unwrap();
    let shapes: Vec<_> = candidates
        .iter()
        .map(|c| (c.n, c.k))
        .collect::<std::collections::BTreeSet<_>>()
        .into_iter()
        .collect();
    for (n, k) in shapes {
        let (wq, ws) = fixture(n as usize, k as usize, seed);
        let weight = device::<Nvfp4E2M1>(&ctx, &wq);
        let weight_sf = device::<Fp8E4M3>(&ctx, &ws);
        for m in [1u32, 2, 4, 8, 16] {
            let (xq, xs) = fixture(m as usize, k as usize, seed + 3);
            let input = device::<Nvfp4E2M1>(&ctx, &xq);
            let input_sf = device::<Fp8E4M3>(&ctx, &xs);
            let count = (m * n) as usize;
            let output = DeviceBuffer::<U8>::with_capacity(ctx.clone(), count * 2 + 32).unwrap();
            let partials = DeviceBuffer::<U8>::with_capacity(ctx.clone(), count * 8 + 32).unwrap();
            let reference = DeviceBuffer::<BF16>::with_capacity(ctx.clone(), count).unwrap();
            let plan = baseline
                .create_nvfp4_plan([m, n, k], Nvfp4Tactic::Tile128x32Dp)
                .unwrap();
            let workspace =
                DeviceBuffer::<U8>::with_capacity(ctx.clone(), plan.workspace_bytes.max(256))
                    .unwrap();
            for alpha in [1f32, 0.137] {
                let scale = device::<F32>(&ctx, &alpha.to_le_bytes());
                let args = Args {
                    a: input.as_raw(),
                    b: weight.as_raw(),
                    sfa: input_sf.as_raw(),
                    sfb: weight_sf.as_raw(),
                    out: unsafe { output.as_raw().add(16) },
                    partials: unsafe { partials.as_raw().add(16) },
                    alpha: scale.as_raw(),
                    m: m as i32,
                    stream: ctx.stream,
                };
                let mut reference_run = || unsafe {
                    baseline
                        .nvfp4_execute(
                            &plan,
                            input.matrix(m, k).unwrap(),
                            weight.matrix(n, k).unwrap(),
                            [
                                input_sf.vector(input_sf.len() as u32).unwrap(),
                                weight_sf.vector(weight_sf.len() as u32).unwrap(),
                            ],
                            scale.vector(1).unwrap(),
                            reference.matrix(m, n).unwrap(),
                            workspace.workspace(workspace.len()).unwrap(),
                        )
                        .unwrap()
                };
                reference_run();
                let expected = bf16(&download(&ctx, &reference));
                // Independent FP64 dot products check corners and tile boundaries.
                for row in [0, m as usize - 1] {
                    for col in [0, 1, 63, 64, 127, 128, n as usize - 1] {
                        let exact = (0..k as usize)
                            .map(|j| {
                                value(&xq, &xs, row, j, k as usize)
                                    * value(&wq, &ws, col, j, k as usize)
                            })
                            .sum::<f64>()
                            * f64::from(alpha);
                        let got = f64::from(expected[row * n as usize + col]);
                        assert!(
                            (got - exact).abs() <= exact.abs() * 0.004 + 0.02,
                            "baseline vs FP64 row={row} col={col} expected={exact} actual={got}"
                        );
                    }
                }
                for candidate in candidates.iter().filter(|c| c.n == n && c.k == k) {
                    let run = || unsafe {
                        (candidate.run)(&args).unwrap();
                        if candidate.splits == 2 {
                            reducer
                                .launch(
                                    ctx.stream,
                                    [((count + 1023) / 1024) as u32, 1, 1],
                                    DevicePtr::<F32>::new(args.partials).unwrap(),
                                    DevicePtr::<BF16>::new(args.out).unwrap(),
                                    count as i32,
                                )
                                .unwrap();
                        }
                    };
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
                        assert_eq!(
                            cuda::cudaMemsetAsync(
                                partials.as_raw().cast(),
                                0xa5,
                                count * 8 + 32,
                                ctx.stream
                            ),
                            0
                        );
                    }
                    run();
                    let bytes = download(&ctx, &output);
                    let pbytes = download(&ctx, &partials);
                    for bytes in [&bytes, &pbytes] {
                        assert!(
                            bytes[..16]
                                .iter()
                                .chain(&bytes[bytes.len() - 16..])
                                .all(|&x| x == 0xa5)
                        );
                    }
                    let actual = bf16(&bytes[16..16 + count * 2]);
                    let (changed, rel, max) = metrics(&actual, &expected);
                    assert!(
                        rel < 0.005,
                        "{} M{m} N{n} K{k} relative_l2={rel} max={max}",
                        candidate.name
                    );
                    writeln!(log,"{{\"kind\":\"correctness\",\"candidate\":\"{}\",\"m\":{m},\"n\":{n},\"k\":{k},\"alpha\":{alpha},\"changed\":{changed},\"relative_l2\":{rel},\"max_abs\":{max}}}",candidate.name).unwrap();
                    log.flush().unwrap();
                    if alpha != 1.0 {
                        continue;
                    }
                    for _ in 0..5 {
                        reference_run();
                        run();
                    }
                    ctx.synchronize().unwrap();
                    for evicted in [false, true] {
                        let mut timings = [Vec::new(), Vec::new()];
                        let mut enqueues = [Vec::new(), Vec::new()];
                        for repeat in 0..30 {
                            for provider in if repeat % 2 == 0 { [0, 1] } else { [1, 0] } {
                                if evicted {
                                    eviction.zero().unwrap();
                                }
                                let (elapsed, host) = events.sample(&ctx, || {
                                    if provider == 0 {
                                        reference_run();
                                    } else {
                                        run();
                                    }
                                });
                                timings[provider].push(elapsed);
                                enqueues[provider].push(host);
                            }
                        }
                        writeln!(log,"{{\"kind\":\"timing\",\"candidate\":\"{}\",\"m\":{m},\"n\":{n},\"k\":{k},\"evicted\":{evicted},\"baseline_us\":{:?},\"candidate_us\":{:?},\"baseline_enqueue_us\":{:?},\"candidate_enqueue_us\":{:?}}}",candidate.name,timings[0],timings[1],enqueues[0],enqueues[1]).unwrap();
                        log.flush().unwrap();
                    }
                    // Recheck after timing with poisoned output and no inter-launch sync.
                    unsafe {
                        assert_eq!(
                            cuda::cudaMemsetAsync(
                                output.as_raw().cast(),
                                0xff,
                                count * 2 + 32,
                                ctx.stream
                            ),
                            0
                        );
                    }
                    run();
                    let post = download(&ctx, &output);
                    let (_, rel, _) = metrics(&bf16(&post[16..16 + count * 2]), &expected);
                    assert!(rel < 0.005);
                    println!(
                        "qualified {} M{m} N{n} K{k}, relative L2={rel}",
                        candidate.name
                    );
                }
            }
        }
    }
    ctx.synchronize().unwrap();
}
