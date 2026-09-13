use super::*;
use crate::ffi::cuda;
use std::ptr;

unsafe extern "C" {
    fn cudaGetDeviceCount(count: *mut i32) -> i32;
    fn cudaStreamCreateWithFlags(stream: *mut ffi::CudaStream, flags: u32) -> i32;
    fn cudaStreamDestroy(stream: ffi::CudaStream) -> i32;
    fn cudaStreamBeginCapture(stream: ffi::CudaStream, mode: i32) -> i32;
    fn cudaStreamEndCapture(stream: ffi::CudaStream, graph: *mut *mut std::ffi::c_void) -> i32;
    fn cudaGraphInstantiateWithFlags(
        exec: *mut *mut std::ffi::c_void,
        graph: *mut std::ffi::c_void,
        flags: u64,
    ) -> i32;
    fn cudaGraphLaunch(exec: *mut std::ffi::c_void, stream: ffi::CudaStream) -> i32;
    fn cudaGraphExecDestroy(exec: *mut std::ffi::c_void) -> i32;
    fn cudaGraphDestroy(graph: *mut std::ffi::c_void) -> i32;
}

fn device_available() -> bool {
    let mut count = 0;
    if unsafe { cudaGetDeviceCount(&mut count) } != 0 || count == 0 {
        eprintln!("SKIP: no CUDA device available");
        return false;
    }
    assert_eq!(unsafe { cuda::cudaSetDevice(0) }, 0);
    true
}

fn params(temperature: f32, top_k: u32, top_p: f32) -> SamplingParams {
    SamplingParams {
        temperature,
        top_k,
        top_p,
        seed: 0xfedc_ba98_7654_3210,
    }
}

// Sort-based reference, independent of the upstream pivot search. Strict ties
// use token ID; the nucleus includes the first token reaching/exceeding p.
fn reference(logits: &[f32], params: SamplingParams) -> Vec<(usize, f64)> {
    let mut ranked: Vec<_> = logits
        .iter()
        .enumerate()
        .filter(|(_, x)| x.is_finite())
        .map(|(i, x)| (i, f64::from(*x) / f64::from(params.temperature)))
        .collect();
    ranked.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
    if params.top_k != 0 {
        ranked.truncate(params.top_k as usize);
    }
    let max = ranked[0].1;
    for (_, x) in &mut ranked {
        *x = (*x - max).exp();
    }
    let total: f64 = ranked.iter().map(|x| x.1).sum();
    let mut cumulative = 0.0;
    let count = ranked
        .iter()
        .position(|x| {
            cumulative += x.1 / total;
            cumulative >= f64::from(params.top_p)
        })
        .map_or(ranked.len(), |i| i + 1);
    ranked.truncate(count);
    let total: f64 = ranked.iter().map(|x| x.1).sum();
    for (_, x) in &mut ranked {
        *x /= total;
    }
    ranked
}

#[test]
fn sampling_parameter_boundaries() {
    assert_eq!(SamplingParams::default().validate(64), Ok(()));
    for p in [
        params(-1.0, 0, 1.0),
        params(f32::NAN, 0, 1.0),
        params(f32::INFINITY, 0, 1.0),
        params(1.0, 65, 1.0),
        params(1.0, 0, 0.0),
        params(1.0, 0, f32::NAN),
        params(1.0, 0, 1.01),
    ] {
        assert_eq!(p.validate(64), Err(Status::InvalidArgument));
    }
    assert_eq!(params(1.0, 64, 1.0).validate(64), Ok(()));
}

#[test]
fn triton_sampling_filter_matches_sorted_reference() {
    if !device_available() {
        return;
    }
    let stream = ptr::null_mut();
    let vocab = kernels::prepare::constants::VOCAB as usize;
    let mut logits = vec![-40.0; vocab];
    // Winners span tiles and the padded vocabulary tail. The padded winner is
    // retained here; ModelRunner is responsible for rejecting it before decode.
    let ids = [0, 1023, 1024, 8191, 8192, vocab - 1];
    for (&id, value) in ids.iter().zip([4.0, 3.0, 2.0, 1.0, 0.0, -1.0]) {
        logits[id] = value;
    }
    let mut input = DeviceBuffer::from_slice(0, stream, &logits).unwrap();
    let positions = DeviceBuffer::from_slice(0, stream, &[53]).unwrap();
    let output = DeviceBuffer::from_slice(0, stream, &[0_i32]).unwrap();
    let mut sampler = Sampler::new(0, stream, vocab as u32, params(1.0, 0, 1.0)).unwrap();
    for p in [
        params(1.0, 1, 1.0),
        params(0.7, 3, 1.0),
        params(1.0, 0, 0.8),
        params(1.3, 5, 0.9),
        params(1.0, 3, 0.8),
        params(1.0, 0, 0.01),
    ] {
        sampler.params = p;
        sampler
            .launch(stream, &input, &positions, 0, &output)
            .unwrap();
        let mut filtered = vec![0.0; vocab];
        sampler.processed.download(stream, &mut filtered).unwrap();
        let kept: Vec<_> = filtered
            .iter()
            .enumerate()
            .filter(|(_, x)| x.is_finite())
            .map(|(i, _)| i)
            .collect();
        let mut expected: Vec<_> = reference(&logits, p).iter().map(|x| x.0).collect();
        expected.sort_unstable();
        assert_eq!(kept, expected, "{p:?}");
        for id in kept {
            assert!((filtered[id] - logits[id] / p.temperature).abs() < 1e-5);
        }
    }
    // Exact boundary ties cross block/tile boundaries. Top-k must keep only k.
    logits.fill(f32::NEG_INFINITY);
    for id in ids {
        logits[id] = 0.0;
    }
    input.upload(stream, &logits).unwrap();
    for p in [
        params(1.0, 3, 1.0),
        params(1.0, 0, 0.5),
        params(1.0, 4, 0.5),
    ] {
        sampler.params = p;
        sampler
            .launch(stream, &input, &positions, 0, &output)
            .unwrap();
        let mut filtered = vec![0.0; vocab];
        sampler.processed.download(stream, &mut filtered).unwrap();
        let kept: Vec<_> = filtered
            .iter()
            .enumerate()
            .filter(|(_, x)| x.is_finite())
            .map(|(i, _)| i)
            .collect();
        let mut expected: Vec<_> = reference(&logits, p).iter().map(|x| x.0).collect();
        expected.sort_unstable();
        assert_eq!(kept, expected, "ties: {p:?}");
    }
    // Mixed duplicates, including k larger than the number of finite logits.
    logits.fill(f32::NEG_INFINITY);
    for i in 0..127 {
        logits[i * 1931] = ((i * 71 % 127) / 3) as f32 * 0.125 - 2.0;
    }
    input.upload(stream, &logits).unwrap();
    for k in [0, 1, 7, 40, 200] {
        for p in [0.1, 0.7, 0.95, 1.0] {
            let settings = params(0.8, k, p);
            sampler.params = settings;
            sampler
                .launch(stream, &input, &positions, 0, &output)
                .unwrap();
            let mut filtered = vec![0.0; vocab];
            sampler.processed.download(stream, &mut filtered).unwrap();
            let kept: Vec<_> = filtered
                .iter()
                .enumerate()
                .filter(|(_, x)| x.is_finite())
                .map(|(i, _)| i)
                .collect();
            let mut expected: Vec<_> = reference(&logits, settings).iter().map(|x| x.0).collect();
            expected.sort_unstable();
            assert_eq!(kept, expected, "repeated logits: {settings:?}");
        }
    }
}

#[test]
fn triton_sampling_distribution_and_position_reproducibility() {
    if !device_available() {
        return;
    }
    let stream = ptr::null_mut();
    let mut logits = vec![f32::NEG_INFINITY; 2051];
    for (id, probability) in [(0, 0.5_f32), (1024, 0.3), (2050, 0.15), (9, 0.05)] {
        logits[id] = probability.ln();
    }
    let input = DeviceBuffer::from_slice(0, stream, &logits).unwrap();
    let draws = 4096;
    let positions =
        DeviceBuffer::from_slice(0, stream, &(0..draws as i32).collect::<Vec<_>>()).unwrap();
    let output = DeviceBuffer::from_slice(0, stream, &[0_i32]).unwrap();
    let mut sampler = Sampler::new(0, stream, logits.len() as u32, params(1.0, 0, 1.0)).unwrap();
    for p in [
        params(1.0, 0, 1.0),
        params(0.7, 0, 1.0),
        params(1.0, 3, 0.9),
    ] {
        sampler.params = p;
        let mut counts = vec![0; logits.len()];
        let mut first = Vec::new();
        for pos in 0..draws {
            sampler
                .launch(stream, &input, &positions, pos, &output)
                .unwrap();
            let mut token = [0_i32];
            output.download(stream, &mut token).unwrap();
            counts[token[0] as usize] += 1;
            if pos < 32 {
                first.push(token[0]);
            }
        }
        let expected = reference(&logits, p);
        for (id, &count) in counts.iter().enumerate() {
            let probability = expected.iter().find(|x| x.0 == id).map_or(0.0, |x| x.1);
            if probability == 0.0 {
                assert_eq!(count, 0, "masked token {id}");
            } else {
                let error = (f64::from(count) / f64::from(draws) - probability).abs();
                assert!(error < 0.035, "{p:?}, token {id}: error {error}");
            }
        }
        for pos in (0..32).rev() {
            sampler
                .launch(stream, &input, &positions, pos, &output)
                .unwrap();
            let mut token = [0_i32];
            output.download(stream, &mut token).unwrap();
            assert_eq!(token[0], first[pos as usize]);
        }
    }
}

#[test]
fn triton_sampling_invalid_logits_fail_and_negative_infinity_masks() {
    if !device_available() {
        return;
    }
    let stream = ptr::null_mut();
    let positions = DeviceBuffer::from_slice(0, stream, &[42]).unwrap();
    let output = DeviceBuffer::from_slice(0, stream, &[0_i32]).unwrap();
    let mut sampler = Sampler::new(0, stream, 17, params(1.0, 0, 1.0)).unwrap();
    for bad in [f32::NEG_INFINITY, f32::NAN, f32::INFINITY, f32::MAX] {
        sampler.params = params(0.5, 3, 0.9);
        let mut logits = [f32::NEG_INFINITY; 17];
        logits[16] = bad;
        let input = DeviceBuffer::from_slice(0, stream, &logits).unwrap();
        sampler
            .launch(stream, &input, &positions, 0, &output)
            .unwrap();
        let mut token = [0_i32];
        output.download(stream, &mut token).unwrap();
        assert_eq!(token[0], -1);
    }
    let mut logits = [f32::NEG_INFINITY; 17];
    logits[16] = -100.0;
    let input = DeviceBuffer::from_slice(0, stream, &logits).unwrap();
    sampler
        .launch(stream, &input, &positions, 0, &output)
        .unwrap();
    let mut token = [0_i32];
    output.download(stream, &mut token).unwrap();
    assert_eq!(token[0], 16);
    assert_eq!(
        sampler.launch(stream, &input, &positions, 1, &output),
        Err(Status::InvalidArgument)
    );
}

#[test]
fn triton_sampling_graph_replay_reads_updated_device_position() {
    if !device_available() {
        return;
    }
    struct Capture {
        stream: ffi::CudaStream,
        graph: *mut std::ffi::c_void,
        exec: *mut std::ffi::c_void,
    }
    impl Drop for Capture {
        fn drop(&mut self) {
            unsafe {
                cuda::cudaStreamSynchronize(self.stream);
                if !self.exec.is_null() {
                    cudaGraphExecDestroy(self.exec);
                }
                if !self.graph.is_null() {
                    cudaGraphDestroy(self.graph);
                }
                cudaStreamDestroy(self.stream);
            }
        }
    }
    let mut capture = Capture {
        stream: ptr::null_mut(),
        graph: ptr::null_mut(),
        exec: ptr::null_mut(),
    };
    assert_eq!(
        unsafe { cudaStreamCreateWithFlags(&mut capture.stream, 1) },
        0
    );
    let stream = capture.stream;
    let input = DeviceBuffer::from_slice(0, stream, &[0.0_f32, -0.3, -0.6, -3.0]).unwrap();
    let mut positions = DeviceBuffer::from_slice(0, stream, &[0_i32]).unwrap();
    let output = DeviceBuffer::from_slice(0, stream, &[0_i32]).unwrap();
    let mut sampler = Sampler::new(0, stream, 4, params(0.8, 3, 0.95)).unwrap();
    let mut expected = Vec::new();
    for pos in 0..32 {
        positions.upload(stream, &[pos]).unwrap();
        sampler
            .launch(stream, &input, &positions, 0, &output)
            .unwrap();
        let mut token = [0];
        output.download(stream, &mut token).unwrap();
        expected.push(token[0]);
    }
    assert!(expected.iter().any(|x| *x != expected[0]));
    assert_eq!(unsafe { cudaStreamBeginCapture(stream, 1) }, 0);
    sampler
        .launch(stream, &input, &positions, 0, &output)
        .unwrap();
    assert_eq!(
        unsafe { cudaStreamEndCapture(stream, &mut capture.graph) },
        0
    );
    assert_eq!(
        unsafe { cudaGraphInstantiateWithFlags(&mut capture.exec, capture.graph, 0) },
        0
    );
    for pos in (0..32).rev() {
        positions.upload(stream, &[pos as i32]).unwrap();
        assert_eq!(unsafe { cudaGraphLaunch(capture.exec, stream) }, 0);
        let mut token = [0];
        output.download(stream, &mut token).unwrap();
        assert_eq!(token[0], expected[pos]);
    }
    // Destroy graphs before their module/allocation owners.
    drop(capture);
}
