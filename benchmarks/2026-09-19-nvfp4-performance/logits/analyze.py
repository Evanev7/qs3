"""Separate identical-prefix comparisons from post-divergence logit rows."""

import array
import hashlib
import json
import math
from pathlib import Path
import sys

VOCAB = 248077


def read_row(path):
    values = array.array("f")
    values.frombytes(path.read_bytes())
    assert len(values) == 248320
    return values[:VOCAB]


def metrics(a, b):
    delta = [x - y for x, y in zip(a, b)]
    mean = sum(delta) / VOCAB
    amax, bmax = max(a), max(b)
    az = sum(math.exp(x - amax) for x in a)
    bz = sum(math.exp(x - bmax) for x in b)
    la, lb = math.log(az) + amax, math.log(bz) + bmax
    return {
        "max_abs": max(map(abs, delta)),
        "rmse": math.sqrt(sum(x*x for x in delta)/VOCAB),
        "mean_delta": mean,
        "centered_rmse": math.sqrt(sum((x-mean)**2 for x in delta)/VOCAB),
        "kl_control_candidate": sum(math.exp(y-lb)*((y-lb)-(x-la)) for x,y in zip(a,b)),
        "argmax": max(range(VOCAB), key=a.__getitem__),
        "control_argmax": max(range(VOCAB), key=b.__getitem__),
    }


def main():
    output = Path(sys.argv[1])
    comparisons = []
    for case in sorted(p for p in output.iterdir() if (p / "result.json").is_file()):
        context, tactic, workspace = case.name.split("-")
        context = int(context)
        control = output / f"{context}-32dp-64m"
        # Also isolate the tactic effect at the smaller workspace.
        controls = [control]
        if tactic == "64dp" and workspace == "16m":
            controls.append(output / f"{context}-32dp-16m")
        for control in controls:
            tokens = json.loads((case / "result.json").read_text())["measurement"]["decode"]["generated_token_ids"]
            ref = json.loads((control / "result.json").read_text())["measurement"]["decode"]["generated_token_ids"]
            assert len(tokens) == len(ref) == 68
            prefix = next((i for i,(a,b) in enumerate(zip(tokens,ref)) if a != b), len(tokens))
            rows = []
            for path in sorted(case.glob("*.f32"), key=lambda p:int(p.stem)):
                offset = int(path.stem) - context
                raw, refraw = path.read_bytes(), (control / path.name).read_bytes()
                row = {"prefix_length": int(path.stem), "same_prefix": offset <= prefix, "exact": raw == refraw, "sha256": hashlib.sha256(raw).hexdigest(), "control_sha256": hashlib.sha256(refraw).hexdigest()}
                if offset in (0,prefix):
                    row["metrics"] = metrics(read_row(path), read_row(control / path.name))
                rows.append(row)
            assert len(rows) == 69
            comparisons.append({"case": case.name, "control": control.name, "common_greedy_prefix": prefix, "same_prefix_rows": sum(r["same_prefix"] for r in rows), "exact_same_prefix_rows": sum(r["same_prefix"] and r["exact"] for r in rows), "rows": rows})
    (output / "comparisons.json").write_text(json.dumps(comparisons, indent=2)+"\n")
    for c in comparisons:
        print(json.dumps({k:v for k,v in c.items() if k != "rows"}), flush=True)


if __name__ == "__main__":
    main()
