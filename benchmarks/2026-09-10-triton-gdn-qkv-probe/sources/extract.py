"""Extract three pinned real GDN QKV tensors for the native-only probe."""

import hashlib
import json
import os
import struct
from pathlib import Path

snapshot = Path(
    os.environ.get(
        "QS3_QWEN36_MODEL_DIR",
        str(
            Path.home()
            / ".cache/huggingface/hub/models--Qwen--Qwen3.6-35B-A3B/snapshots/995ad96eacd98c81ed38be0c5b274b04031597b0"
        ),
    )
)
output = Path(".prototypes/gdn_qkv_aot/out")
index = json.loads((snapshot / "model.safetensors.index.json").read_text())
records = []
for layer in (0, 18, 38):
    name = f"model.language_model.layers.{layer}.linear_attn.in_proj_qkv.weight"
    shard = snapshot / index["weight_map"][name]
    with shard.open("rb") as source:
        size = struct.unpack("<Q", source.read(8))[0]
        header = json.loads(source.read(size))
        tensor = header[name]
        assert tensor["shape"] == [8192, 2048] and tensor["dtype"] == "BF16", tensor
        begin, end = tensor["data_offsets"]
        assert 0 <= begin < end <= shard.stat().st_size - 8 - size
        assert end - begin == 8192 * 2048 * 2
        source.seek(8 + size + begin)
        data = source.read(end - begin)
        assert len(data) == end - begin
    (output / f"layer{layer}.bf16").write_bytes(data)
    records.append(
        dict(name=name, shard=shard.name, sha256=hashlib.sha256(data).hexdigest())
    )
(output / "weights.json").write_text(
    json.dumps(dict(snapshot=str(snapshot), tensors=records), indent=2) + "\n"
)
