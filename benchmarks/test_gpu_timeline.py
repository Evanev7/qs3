"""CPU-only checks: python3 -m unittest discover -s benchmarks -p 'test_gpu_timeline.py'."""

import sqlite3
import tempfile
from pathlib import Path
import unittest

from render_history import classify_gaps, extract_directory, extract_phase, kernel_kind, summarize_phase, variant_label


class TimelineTests(unittest.TestCase):
    def test_preserves_overlapping_streams_launch_shapes_and_nanosecond_precision(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "decode.sqlite"
            with sqlite3.connect(path) as db:
                db.executescript("""
                    CREATE TABLE StringIds(id INTEGER,value TEXT);
                    INSERT INTO StringIds VALUES(1,'kernel <float>');
                    CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL(
                        start INTEGER,end INTEGER,deviceId INTEGER,contextId INTEGER,
                        streamId INTEGER,demangledName INTEGER,gridX INTEGER,gridY INTEGER,
                        gridZ INTEGER,blockX INTEGER,blockY INTEGER,blockZ INTEGER,
                        registersPerThread INTEGER,staticSharedMemory INTEGER,dynamicSharedMemory INTEGER);
                    CREATE TABLE ENUM_CUDA_MEMCPY_OPER(id INTEGER,label TEXT);
                    INSERT INTO ENUM_CUDA_MEMCPY_OPER VALUES(1,'HtoD');
                    CREATE TABLE CUPTI_ACTIVITY_KIND_MEMCPY(
                        start INTEGER,end INTEGER,deviceId INTEGER,contextId INTEGER,
                        streamId INTEGER,copyKind INTEGER,bytes INTEGER);
                """)
                start = 2**54
                for a, b, stream in [(10, 60, 7), (30, 80, 8), (90, 100, 7)]:
                    db.execute("INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                               (start+a, start+b, 0, 1, stream, 1, 4, 1, 1, 128, 1, 1, 64, 0, 1024))
                db.execute("INSERT INTO CUPTI_ACTIVITY_KIND_MEMCPY VALUES(?,?,?,?,?,?,?)",
                           (start, start+20, 0, 1, 9, 1, 4096))
            before = path.read_bytes()
            trace = extract_phase(path)
            self.assertEqual(trace["origin_ns"], start)
            self.assertEqual(trace["events"], [
                [0, 20, 1, 0, 1, 9], [10, 50, 0, 0, 1, 7],
                [30, 50, 0, 0, 1, 8], [90, 10, 0, 0, 1, 7]])
            self.assertEqual(trace["definitions"][0]["grid"], [4, 1, 1])
            self.assertEqual(trace["definitions"][0]["shared_bytes"], 1024)
            self.assertEqual(trace["definitions"][1]["bytes"], 4096)
            self.assertEqual(trace["gaps"], [[80, 10, 5]])
            summary = summarize_phase(trace)
            self.assertEqual(summary["span_ns"], 100)
            self.assertEqual(summary["busy_ns"], 90)
            self.assertEqual(summary["work_ns"], 130)
            self.assertEqual(summary["overlap_ns"], 40)
            self.assertEqual(summary["gap_ns"], 10)
            self.assertEqual(set(extract_directory(Path(root))), {"decode"})
            self.assertEqual(path.read_bytes(), before)
            with sqlite3.connect(path) as db:
                db.executescript("""
                    CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME(start INTEGER,end INTEGER,nameId INTEGER);
                    INSERT INTO StringIds VALUES(2,'cudaMemcpyAsync_v3020');
                """)
                db.execute("INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES(?,?,2)", (start+80, start+90))
            with_api = extract_phase(path)
            self.assertTrue(with_api["gap_api_available"])
            self.assertEqual(with_api["gaps"], [[80, 10, 2]])
            self.assertEqual(summarize_phase(with_api)["groups"]["No GPU work"]["Transfer API active"][0]["ns"], 10)
            with sqlite3.connect(path) as db:
                db.execute("UPDATE CUPTI_ACTIVITY_KIND_KERNEL SET end=start-1")
            with self.assertRaisesRegex(ValueError, "ends before"):
                extract_phase(path)

    def test_missing_capture_is_an_error(self):
        with tempfile.TemporaryDirectory() as root:
            with self.assertRaisesRegex(ValueError, "no prefill/decode"):
                extract_directory(Path(root))

    def test_gap_evidence_is_exclusive_clipped_and_not_a_causal_claim(self):
        events = [[0, 10], [30, 10], [50, 10]]
        gaps, count = classify_gaps(events, [
            (5, 14, 0), (12, 22, 3), (18, 20, 1), (21, 26, 4), (42, 45, 0),
        ])
        self.assertEqual(count, 2)
        self.assertEqual(gaps, [
            [10, 4, 0], [14, 4, 3], [18, 2, 1], [20, 2, 3], [22, 4, 4],
            [26, 4, 5], [40, 2, 5], [42, 3, 0], [45, 5, 5],
        ])
        self.assertEqual(sum(g[1] for g in gaps), 30)
        # A nested shorter kernel must not make its enclosing GPU work look idle.
        self.assertEqual(classify_gaps([[0, 50], [5, 10], [60, 10]], []), ([[50, 10, 5]], 1))
        with self.assertRaisesRegex(ValueError, "API interval ends before"):
            classify_gaps(events, [(20, 19, 0)])

    def test_operation_families_do_not_confuse_quantization_with_gemm(self):
        def classify(name):
            return kernel_kind(dict(name=name, kind="kernel"))
        self.assertEqual(classify("fp8_quantize_kernel"), ("Quantization", "FP8 quantization"))
        self.assertEqual(classify("nvjet_sm121_qqtst_mma_64x128x128"), ("Matrix multiplication", "FP8 GEMM"))
        self.assertEqual(classify("GemmUniversal<float_e2m1_t>"), ("Matrix multiplication", "NVFP4 GEMM"))
        self.assertEqual(classify("qwen36_gdn_rmsnorm_gated_kernel"), ("GDN", "Gated normalization"))
        self.assertEqual(classify("embedding_gather_bf16_kernel"), ("Embedding / sampling", "Embedding lookup"))

        # CuTe exports retain the source hash and Python class in their Nsight names.
        for name, kind in [
            ("kernel_cutlass_kernel__qscute_source_7b4bc2997e5df419475449fc54ba498f9634feb9cf4d15d14a436444da7b0eb7Fp8Decode_object_at__CopyAtom_ThrID10_TVLayoutSrc1409601_TVLayoutDst1409601_Valuetypef_0", "FP8 GEMM"),
            ("kernel_cutlass_kernel__qscute_source_561a8ce51f0e04f87de3e56b3b06066119eabfda6309ac63469fb60ccc9db680DenseGemmKernel_object_at__CopyAtom_ThrID10_TVLayoutSrc1819201_TVLayoutDst1819201_Valu_0", "NVFP4 GEMM"),
        ]:
            with self.subTest(kind=kind):
                self.assertEqual(classify(name), ("Matrix multiplication", kind))
                definition = dict(name=name, kind="kernel", grid=[1, 1, 48])
                self.assertEqual(variant_label(definition, kind), f"{kind} · qscute · grid 1 × 1 × 48")


if __name__ == "__main__":
    unittest.main()
