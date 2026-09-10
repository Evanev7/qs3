"""Summarize the completed sweep; never treat isolated kernels as model TPS."""

import csv
import json
import statistics
import sys
from pathlib import Path

root = Path(sys.argv[1])
groups = {}
for layer in (0, 18, 38):
    for mode in ("device", "managed_upload", "managed_cpu"):
        stem = root / f"layer{layer}_{mode}"
        rows = list(csv.DictReader(stem.with_suffix(".csv").open()))
        assert len(rows) == 36, (stem, len(rows))
        log = stem.with_suffix(".log").read_text()
        assert log.count("validated ") == 6, stem
        assert "mismatch" not in log, stem
        seen = set()
        for row in rows:
            assert (row["n"], row["k"], row["dtype"]) == ("8192", "2048", "bf16")
            key = (
                mode,
                row["kernel"],
                int(row["rows"]),
                int(row["warps"]),
                row["cache"],
            )
            sample = (key, int(row["repeat"]))
            assert sample not in seen, (stem, sample)
            seen.add(sample)
            control = float(row["cublas_p50_us"])
            candidate = float(row["triton_p50_us"])
            assert control > 0 and candidate > 0
            groups.setdefault(key, []).append((control, candidate))

summary = []
for (mode, kernel, rows, warps, cache), pairs in sorted(groups.items()):
    assert len(pairs) == 9
    savings = [a - b for a, b in pairs]
    summary.append(
        dict(
            mode=mode,
            kernel=kernel,
            rows=rows,
            warps=warps,
            cache=cache,
            paired_measurements=len(pairs),
            cublas_median_us=statistics.median(a for a, _ in pairs),
            triton_median_us=statistics.median(b for _, b in pairs),
            saving_median_us=statistics.median(savings),
            saving_min_us=min(savings),
            saving_max_us=max(savings),
            median_latency_reduction_percent=statistics.median(
                100 * (a - b) / a for a, b in pairs
            ),
        )
    )
(root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
print(
    "| Allocation | Candidate rows/warps | cuBLASLt µs | Triton µs | Median saving µs | Paired saving range µs |"
)
print("| --- | --- | ---: | ---: | ---: | ---: |")
for row in summary:
    if row["cache"] == "evicted":
        print(
            f"| {row['mode']} | {row['kernel']} {row['rows']}/{row['warps']} | "
            f"{row['cublas_median_us']:.3f} | {row['triton_median_us']:.3f} | "
            f"{row['saving_median_us']:.3f} | {row['saving_min_us']:.3f}–{row['saving_max_us']:.3f} |"
        )
