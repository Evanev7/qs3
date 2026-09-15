"""Summarize the saved comparisons; no model execution or raw rows required."""

import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REFERENCE = ROOT.parent / "2026-09-14-vllm-references/capture/data/qwen3.8-27b"


def summarize(name):
    case = ROOT / "capture" / name
    comparison = json.loads((case / "comparison.json").read_text())
    qs3 = json.loads((case / "qs3/scores.json").read_text())
    vllm = json.loads((REFERENCE / name / "vllm/scores.json").read_text())
    assert qs3["input"] == vllm["input"]
    assert qs3["addressable_token_count"] == 248077
    assert len(comparison["captures"]) == comparison["steps"]
    differing = [
        row for row in comparison["captures"]
        if row["qs3_argmax"] != row["vllm_argmax"]
    ]
    ties = [row["step"] for row in differing if row["qs3_winner_vllm_gap"] == 0]
    return {
        "rows": comparison["steps"],
        "argmax_agreement": comparison["argmax_agreement_count"],
        "difference_steps": comparison["argmax_difference_steps"],
        "differences_with_qs3_winner_tied_for_vllm_max": ties,
        "differences_outside_vllm_max_tie": [
            row["step"] for row in differing if row["qs3_winner_vllm_gap"] > 0
        ],
        "mean_centered_rmse": comparison["mean_centered_rmse"],
        "mean_kl_vllm_to_qs3": statistics.fmean(
            row["kl_vllm_to_qs3"] for row in comparison["captures"]
        ),
        "max_kl_vllm_to_qs3": comparison["max_kl_vllm_to_qs3"],
        "largest_distribution_differences": sorted(
            comparison["captures"], key=lambda row: row["kl_vllm_to_qs3"], reverse=True
        )[:3],
        "qs3_addressable_forced_mean_nll": statistics.fmean(
            row["addressable_forced_nll"] for row in qs3["records"][:-1]
        ),
        "vllm_addressable_forced_mean_nll": statistics.fmean(
            row["addressable_logsumexp"] - row["forced_logit"]
            for row in vllm["records"][:-1]
        ),
        "differing_rows": differing,
    }


if __name__ == "__main__":
    result = {
        "model": "qwen3.8-27b",
        "model_revision": "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0",
        "protocol": "BF16 forced-prefix replay; FP32 GDN state; no timing claim",
        "qs3_lm_head_output": "float32",
        "vllm_lm_head_output": "bfloat16 (stored as float32)",
        "distribution_metrics": "full padded vocabulary; NLL also reported over addressable IDs",
        "workloads": {
            name: summarize(name) for name in ("4-4", "102-32", "500-200", "4000-800")
        },
    }
    (ROOT / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    for name, case in result["workloads"].items():
        print(name, json.dumps({
            key: value for key, value in case.items()
            if key not in ("differing_rows", "largest_distribution_differences")
        }))
