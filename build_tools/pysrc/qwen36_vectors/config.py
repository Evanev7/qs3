"""Explicit numerical inputs exported by nixsrc/vectors.nix."""

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

from qsutil.config import parse


@dataclass(frozen=True)
class VectorConfig:
    hidden_size: int
    q_heads: int
    kv_heads: int
    head_dim: int
    rotary_dim: int
    q_hidden: int
    kv_hidden: int
    q_proj_out: int
    group_size: int
    rms_eps: float
    rope_theta: float
    key_heads: int
    value_heads: int
    key_dim: int
    value_dim: int
    conv_width: int
    qkv_dim: int
    gdn_output_dim: int
    has_experts: bool

    @classmethod
    def read(cls, path: str | Path) -> VectorConfig:
        return parse(Path(path).read_text(), cls)

    def json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True, indent=2) + "\n"

    @property
    def key(self) -> str:
        return hashlib.sha256(self.json().encode()).hexdigest()[:16]

    @property
    def attention_scale(self) -> float:
        return 1.0 / math.sqrt(self.head_dim)

    @property
    def recurrent_scale(self) -> float:
        return 1.0 / math.sqrt(self.key_dim)

    @property
    def q_dim(self) -> int:
        return self.key_heads * self.key_dim

    @property
    def k_dim(self) -> int:
        return self.q_dim

    @property
    def v_dim(self) -> int:
        return self.gdn_output_dim

    @property
    def z_dim(self) -> int:
        return self.gdn_output_dim

    @property
    def qkvz_dim(self) -> int:
        return self.qkv_dim + self.gdn_output_dim

    @property
    def ba_dim(self) -> int:
        return 2 * self.value_heads
