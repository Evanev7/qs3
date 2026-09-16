"""Configuration changes must reach the reference data and its oracle selection."""

import subprocess
from dataclasses import replace
from pathlib import Path

import pytest
from qsutil.config import parse
from qwen36_vectors.attention import generate_tensors
from qwen36_vectors.config import VectorConfig
from qwen36_vectors.gdn import TOTAL_TOKENS, _build_projected_inputs
from qwen36_vectors.oracle import OracleError, check_oracles, write_oracles


@pytest.fixture(scope="module")
def vector_config() -> VectorConfig:
    export = Path(__file__).resolve().parents[1] / "nixsrc/vectors.nix"
    return parse(
        subprocess.check_output(
            ["nix", "eval", "--offline", "--raw", "--file", str(export)], text=True
        ),
        VectorConfig,
    )


def test_attention_generation_uses_each_explicit_config(vector_config: VectorConfig):
    # Two configs in one process catch stale module globals and imported constants.
    narrow = replace(
        vector_config,
        q_heads=2,
        kv_heads=1,
        kv_hidden=vector_config.head_dim,
        group_size=2,
        q_hidden=2 * vector_config.head_dim,
        q_proj_out=4 * vector_config.head_dim,
    )
    first = generate_tensors(narrow)
    second = generate_tensors(vector_config)
    again = generate_tensors(narrow)
    assert first == again
    a = {tensor.name: tensor for tensor in first}
    b = {tensor.name: tensor for tensor in second}
    assert a["packed_q_gate"].shape != b["packed_q_gate"].shape
    assert len(a["packed_q_gate"].data) * vector_config.q_heads == (
        len(b["packed_q_gate"].data) * narrow.q_heads
    )


def test_oracles_reject_wrong_config_and_changed_bytes(
    vector_config: VectorConfig, tmp_path: Path
):
    vectors = tmp_path / "vectors"
    group = vectors / "full_attention_primitives"
    group.mkdir(parents=True)
    (vectors / "config.json").write_text(vector_config.json())
    data = group / "input.bf16"
    data.write_bytes(b"\x80\x3f")
    oracles = tmp_path / "oracles"
    groups = (group.name,)
    write_oracles(vectors, oracles, groups=groups)
    check_oracles(vectors, oracles, groups=groups)

    changed = replace(vector_config, rope_theta=vector_config.rope_theta * 2)
    (vectors / "config.json").write_text(changed.json())
    with pytest.raises(OracleError, match="no committed vector oracles"):
        check_oracles(vectors, oracles, groups=groups)

    (vectors / "config.json").write_text(vector_config.json())
    data.write_bytes(b"\x00\x00")
    with pytest.raises(OracleError, match="sha256/size mismatch"):
        check_oracles(vectors, oracles, groups=groups)


@pytest.mark.parametrize("repeat", [2, 3])
def test_gdn_debug_layout_retains_all_value_heads(
    vector_config: VectorConfig, repeat: int
):
    config = replace(
        vector_config,
        value_heads=repeat * vector_config.key_heads,
        gdn_output_dim=repeat * vector_config.key_heads * vector_config.value_dim,
        qkv_dim=(2 * vector_config.key_dim + repeat * vector_config.value_dim)
        * vector_config.key_heads,
    )
    qkvz, ba, _, _, _, _, interleaved_qkvz, interleaved_ba = _build_projected_inputs(
        config
    )
    assert len(qkvz) == len(interleaved_qkvz) == TOTAL_TOKENS * config.qkvz_dim
    assert len(ba) == len(interleaved_ba) == TOTAL_TOKENS * config.ba_dim
    # The debug format must only permute heads within each token.
    for flat, interleaved, width in (
        (qkvz, interleaved_qkvz, config.qkvz_dim),
        (ba, interleaved_ba, config.ba_dim),
    ):
        for token in range(TOTAL_TOKENS):
            span = slice(token * width, (token + 1) * width)
            assert sorted(flat[span]) == sorted(interleaved[span])
