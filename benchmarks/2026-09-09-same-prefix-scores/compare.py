"""Compare recorded full-vocabulary scores at identical forced prefixes."""

import argparse
import array
import hashlib
import json
import math
import statistics
import sys
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("run", type=Path)
a = p.parse_args()
q = json.loads((a.run / "qs3/scores.json").read_text())
v = json.loads((a.run / "vllm/scores.json").read_text())
assert q["input"] == v["input"]
qr, vr = q["records"], v["records"]
assert len(qr) == len(vr) == len(q["input"]["forced_ids"]) + 1
assert [r["step"] for r in qr] == [r["step"] for r in vr] == list(range(len(qr)))


def read(path):
    data = path.read_bytes()
    values = array.array("f")
    values.frombytes(data)
    if sys.byteorder != "little":
        values.byteswap()
    assert len(values) == 248320 and all(map(math.isfinite, values))
    return values, hashlib.sha256(data).hexdigest()


def logsumexp(x):
    maximum = max(x)
    return maximum + math.log(sum(math.exp(t - maximum) for t in x))


captures = []
for x, y in zip(qr, vr):
    assert x.get("forced_id") == y.get("forced_id")
    if "file" not in x:
        assert "file" not in y
        continue
    xx, hq = read(a.run / "qs3" / x["file"])
    yy, hv = read(a.run / "vllm" / y["file"])
    diff = [float(i) - float(j) for i, j in zip(xx, yy)]
    mean = statistics.fmean(diff)
    lq, lv = logsumexp(xx), logsumexp(yy)
    kl = sum(math.exp(j - lv) * ((j - lv) - (i - lq)) for i, j in zip(xx, yy))
    choices = sorted(set(x["top_ids"][:5] + y["top_ids"][:5]))
    captures.append(
        {
            "step": x["step"],
            "qs3_sha256": hq,
            "vllm_sha256": hv,
            "mean_delta": mean,
            "mean_abs_delta": statistics.fmean(map(abs, diff)),
            "max_abs_delta": max(map(abs, diff)),
            "rmse": math.sqrt(statistics.fmean(t * t for t in diff)),
            "centered_rmse": math.sqrt(statistics.fmean((t - mean) ** 2 for t in diff)),
            "kl_vllm_to_qs3": kl,
            "qs3_argmax": x["argmax"],
            "vllm_argmax": y["argmax"],
            "qs3_top2_margin": x["top_values"][0] - x["top_values"][1],
            "vllm_top2_margin": y["top_values"][0] - y["top_values"][1],
            "candidate_scores": [
                {"id": t, "qs3": xx[t], "vllm": yy[t]} for t in choices
            ],
        }
    )
agree = [x["argmax"] == y["argmax"] for x, y in zip(qr, vr)]
forced = q["input"]["forced_ids"]
result = {
    "protocol": q["protocol"],
    "steps": len(qr),
    "argmax_agreement_count": sum(agree),
    "first_argmax_difference": next((i for i, x in enumerate(agree) if not x), None),
    "argmax_difference_steps": [i for i, x in enumerate(agree) if not x],
    "qs3_forced_mean_nll": statistics.fmean(x["forced_nll"] for x in qr[:-1]),
    "vllm_forced_mean_nll": statistics.fmean(x["forced_nll"] for x in vr[:-1]),
    "vllm_matches_original_greedy_count": sum(
        r["argmax"] == t for r, t in zip(vr, forced)
    ),
    "captures": captures,
}
(a.run / "comparison.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps({k: v for k, v in result.items() if k != "captures"}, indent=2))
