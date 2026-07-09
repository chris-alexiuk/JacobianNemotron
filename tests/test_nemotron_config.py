import math

import pytest

from nemotron_jlens.config import (
    ACCEPTANCE_SCHEMA,
    ARTIFACT_STORAGE_DTYPE,
    NEMOTRON,
    PAPER_SCALE_CORPUS_MANIFEST_SHA256,
    PAPER_SCALE_N_PROMPTS,
    PILOT_CORPUS_MANIFEST_SHA256,
    PILOT_N_PROMPTS,
    REPRODUCTION_FIT_SETTINGS,
    REQUIRED_TRANSFORMERS_VERSION,
    RUNTIME_DISTRIBUTIONS,
)
from nemotron_jlens.pipeline import parse_source_layers, resolve_layer
from nemotron_jlens.provenance import adaptation_source_sha256


def test_nemotron_resource_estimates_match_pinned_architecture():
    assert NEMOTRON.n_layers == 52
    assert NEMOTRON.source_layer_count == 51
    assert NEMOTRON.d_model == 2688
    assert NEMOTRON.vocab_size == 131072

    matrix_elements = 51 * 2688 * 2688
    assert math.isclose(
        NEMOTRON.artifact_size_gib(bytes_per_element=2),
        matrix_elements * 2 / 1024**3,
    )
    assert math.isclose(NEMOTRON.fit_host_ram_gib(), matrix_elements * 8 / 1024**3)
    assert NEMOTRON.backward_passes_per_prompt(8) == 336
    assert NEMOTRON.backward_passes_per_prompt(2688) == 1
    assert NEMOTRON.backward_passes_per_prompt(2689) == 1
    with pytest.raises(ValueError, match="positive"):
        NEMOTRON.backward_passes_per_prompt(0)


def test_nemotron_serialized_spec_pins_layout_and_revision():
    serialized = NEMOTRON.to_dict()
    assert serialized["revision"] == ("cbd3fa9f933d55ef16a84236559f4ee2a0526848")
    assert serialized["layout"] == {
        "path": "backbone",
        "layers": "layers",
        "norm": "norm_f",
        "embed": "embeddings",
        "lm_head": "lm_head",
    }


def test_reproduction_profiles_pin_manifests_settings_and_fp32_storage():
    assert ARTIFACT_STORAGE_DTYPE == "float32"
    assert ACCEPTANCE_SCHEMA == "nemotron-jlens-acceptance/v1"
    assert PILOT_N_PROMPTS == 100
    assert PAPER_SCALE_N_PROMPTS == 1000
    assert PILOT_CORPUS_MANIFEST_SHA256 == (
        "df38859ae72daab950b93ddd912106138c2ea1b51fc69f6adcc3673c918ccb3d"
    )
    assert PAPER_SCALE_CORPUS_MANIFEST_SHA256 == (
        "c75fc7ee5d92335f0620a08c6f87d210fc3b3fe4f3d6bcfae7d0cd864a88d63e"
    )
    assert REPRODUCTION_FIT_SETTINGS == {
        "dim_batch": 8,
        "max_seq_len": 128,
        "skip_first": 16,
        "dtype": "bfloat16",
        "compile_blocks": False,
        "disable_mamba_kernels": False,
        "mamba_backend": "fused-or-auto",
        "patched_mamba_layers": 0,
    }
    assert REQUIRED_TRANSFORMERS_VERSION == "4.57.3"
    assert RUNTIME_DISTRIBUTIONS == (
        "transformers",
        "mamba-ssm",
        "causal-conv1d",
        "accelerate",
        "datasets",
        "huggingface-hub",
    )


def test_adaptation_source_fingerprint_is_stable_sha256():
    first = adaptation_source_sha256()
    second = adaptation_source_sha256()
    assert first == second
    assert len(first) == 64
    assert set(first) <= set("0123456789abcdef")


@pytest.mark.parametrize(
    ("index", "expected"),
    [(0, 0), (51, 51), (-1, 51), (-52, 0)],
)
def test_resolve_layer(index, expected):
    assert resolve_layer(index, 52) == expected


@pytest.mark.parametrize("index", [-53, 52, 99])
def test_resolve_layer_rejects_out_of_range(index):
    with pytest.raises(ValueError, match="out of range"):
        resolve_layer(index, 52)


def test_parse_source_layers_supports_all_lists_ranges_and_negatives():
    assert parse_source_layers("all", n_layers=52, target_layer=51) == list(range(51))
    assert parse_source_layers("0, 8, 16, -2, 8", n_layers=52, target_layer=51) == [
        0,
        8,
        16,
        50,
    ]
    assert parse_source_layers("0:51:10", n_layers=52, target_layer=51) == [
        0,
        10,
        20,
        30,
        40,
        50,
    ]
    assert parse_source_layers("0:-1:25", n_layers=52, target_layer=51) == [
        0,
        25,
        50,
    ]


@pytest.mark.parametrize("spec", ["", ",", "51", "0:52", "0:1:2:3"])
def test_parse_source_layers_rejects_empty_invalid_or_target_layers(spec):
    with pytest.raises(ValueError):
        parse_source_layers(spec, n_layers=52, target_layer=51)
