// Diagnostic-only same-input comparison; CUTLASS output continues through the model.
pub(crate) struct Probe {
    pub(crate) enabled: bool,
    pub(crate) compare_enabled: bool,
    candidates: Vec<Candidate>,
    output: DeviceBuffer<BF16>,
    ctx: Rc<CudaCtx>,
    log: std::cell::RefCell<(File, usize)>,
}
impl Probe {
    pub(crate) fn new(ctx: Rc<CudaCtx>) -> Self {
        let root = std::path::PathBuf::from(std::env::var_os("NVFP4_OUTPUT").unwrap());
        Self {
            enabled: std::env::var("NVFP4_BENCH_PROVIDER").as_deref() == Ok("cute"),
            compare_enabled: std::env::var_os("NVFP4_BENCH_PROVIDER").is_none(),
            candidates: load_candidates(),
            output: DeviceBuffer::with_capacity(ctx.clone(), 16 * 248320).unwrap(),
            ctx,
            log: std::cell::RefCell::new((
                File::create(root.join("real-results.jsonl")).unwrap(),
                0,
            )),
        }
    }
    pub(crate) unsafe fn launch(
        &self,
        input: crate::backend::DMat<Nvfp4E2M1>,
        weight: crate::backend::DMat<Nvfp4E2M1>,
        scales: [crate::backend::DVec<Fp8E4M3>; 2],
        alpha: crate::backend::DVec<F32>,
        expected: crate::backend::DMat<BF16>,
    ) {
        let [m, k] = input.shape();
        let [n, _] = weight.shape();
        assert!(m <= 16);
        let candidate = self
            .candidates
            .iter()
            .find(|c| c.n == n && c.k == k)
            .unwrap();
        assert_eq!(candidate.splits, 1);
        let args = Args {
            a: input.data.as_raw(),
            b: weight.data.as_raw(),
            sfa: scales[0].data.as_raw(),
            sfb: scales[1].data.as_raw(),
            out: expected.data.as_raw(),
            partials: std::ptr::null_mut(),
            alpha: alpha.data.as_raw(),
            m: m as i32,
            stream: self.ctx.stream,
        };
        (candidate.run)(&args).unwrap();
    }
    pub(crate) unsafe fn compare(
        &self,
        input: crate::backend::DMat<Nvfp4E2M1>,
        weight: crate::backend::DMat<Nvfp4E2M1>,
        scales: [crate::backend::DVec<Fp8E4M3>; 2],
        alpha: crate::backend::DVec<F32>,
        expected: crate::backend::DMat<BF16>,
    ) {
        let [m, k] = input.shape();
        let [n, _] = weight.shape();
        let candidate = self
            .candidates
            .iter()
            .find(|c| c.n == n && c.k == k)
            .unwrap();
        unsafe {
            self.launch(
                input,
                weight,
                scales,
                alpha,
                self.output.matrix(m, n).unwrap(),
            );
        }
        let count = (m * n) as usize;
        let mut a = HostBuffer::<BF16>::new(count).unwrap();
        let mut b = HostBuffer::<BF16>::new(count).unwrap();
        unsafe {
            self.output.download_range(0, &mut a).unwrap();
            self.ctx
                .download(expected.data.as_raw(), b.as_mut())
                .unwrap();
        }
        self.ctx.synchronize().unwrap();
        let (changed, rel, max) = metrics(&bf16(a.as_ref()), &bf16(b.as_ref()));
        let mut log = self.log.borrow_mut();
        let index = log.1;
        writeln!(log.0,"{{\"index\":{index},\"candidate\":\"{}\",\"m\":{m},\"n\":{n},\"k\":{k},\"changed\":{changed},\"relative_l2\":{rel},\"max_abs\":{max}}}",candidate.name).unwrap();
        log.0.flush().unwrap();
        log.1 += 1;
        assert!(
            rel < 0.005,
            "real NVFP4 {} index={index} relative_l2={rel} max={max}",
            candidate.name
        );
    }
}

#[test]
#[ignore]
fn nvfp4_model_benchmark() {
    let result = crate::loader::benchmark::run_core_benchmark();
    let root = std::path::PathBuf::from(std::env::var_os("NVFP4_OUTPUT").unwrap());
    std::fs::write(
        root.join("model-benchmark.json"),
        result.stringify().unwrap(),
    )
    .unwrap();
}
