import json
import sqlite3
from pathlib import Path

source = Path(".prototypes/vllm-runs/2026-09-09T050426Z-vllm-bf16-decode")
output = Path("benchmarks/2026-09-09T050426Z-vllm-bf16-decode")
conn = sqlite3.connect(f"file:{source / 'decode.sqlite'}?mode=ro", uri=True)


def merge(spans):
    merged = []
    for start, end in sorted(spans):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
        else:
            merged.append((start, end))
    return merged


def duration(spans):
    return sum(end - start for start, end in spans)


def intersection(a, b):
    i = j = total = 0
    while i < len(a) and j < len(b):
        total += max(0, min(a[i][1], b[j][1]) - max(a[i][0], b[j][0]))
        if a[i][1] < b[j][1]:
            i += 1
        else:
            j += 1
    return total


kernels = list(conn.execute("SELECT start,end FROM CUPTI_ACTIVITY_KIND_KERNEL"))
gpu = list(kernels)
for table in ["CUPTI_ACTIVITY_KIND_MEMCPY", "CUPTI_ACTIVITY_KIND_MEMSET"]:
    gpu += list(conn.execute(f"SELECT start,end FROM {table}"))
gpu = merge(gpu)
pins = merge(
    list(
        conn.execute(
            "SELECT r.start,r.end FROM CUPTI_ACTIVITY_KIND_RUNTIME r JOIN StringIds s ON s.id=r.nameId WHERE s.value LIKE 'cudaHostAlloc_%' OR s.value LIKE 'cudaFreeHost_%'"
        )
    )
)
start, end = min(a for a, b in kernels), max(b for a, b in kernels)
summary = {
    "kernel_count": len(kernels),
    "kernel_duration_sum_ms": duration(kernels) / 1e6,
    "kernel_span_ms": (end - start) / 1e6,
    "gpu_work_union_within_kernel_span_ms": intersection(gpu, [(start, end)]) / 1e6,
    "no_gpu_work_within_kernel_span_ms": (
        (end - start) - intersection(gpu, [(start, end)])
    )
    / 1e6,
    "pinned_host_api_union_ms": duration(pins) / 1e6,
    "pinned_host_api_overlap_gpu_work_ms": intersection(pins, gpu) / 1e6,
    "grouped_moe_grid_counts": list(
        conn.execute(
            "SELECT k.gridX,k.gridY,k.gridZ,COUNT(*) FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON s.id=k.demangledName WHERE s.value LIKE '%GemmGrouped%' GROUP BY k.gridX,k.gridY,k.gridZ"
        )
    ),
}
(output / "timing_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary, indent=2))
