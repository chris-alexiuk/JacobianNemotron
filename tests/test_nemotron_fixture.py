# Copyright 2026 Anthropic PBC
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from unittest.mock import patch

import pytest
import torch

from jlens import JacobianLens
from jlens.fitting import fit
from nemotron_jlens.cli import _run, build_parser
from nemotron_jlens.config import (
    NEMOTRON,
    REQUIRED_TRANSFORMERS_VERSION,
    UPSTREAM_JLENS_COMMIT,
)
from nemotron_jlens.corpus import sha256_text
from nemotron_jlens.fixture import (
    JACOBIAN_LENS,
    LOGIT_LENS,
    _top_readout,
    _tracked_readout,
    _write_fixture_atomic,
    build_fixture,
)

from .tiny import TinyDecoder

LENS_SHA = sha256_text("tiny fitted lens artifact")
CORPUS_SHA = sha256_text("tiny fitted corpus")
SOURCE_SHA = sha256_text("tiny test adaptation source")
TEST_RUNTIME = {
    "python": "3.12.0",
    "torch": "2.9.0+cu130",
    "cuda_runtime": "13.0",
    "packages": {
        "transformers": REQUIRED_TRANSFORMERS_VERSION,
        "mamba-ssm": "2.2.5",
        "causal-conv1d": "1.5.3",
        "accelerate": "1.12.0",
        "datasets": "3.1.0",
        "huggingface-hub": "0.36.0",
    },
}


@pytest.fixture(scope="module")
def fitted_tiny() -> tuple[TinyDecoder, JacobianLens]:
    """Fit a real tiny-model Jacobian; no Nemotron values are synthesized."""
    model = TinyDecoder(n_layers=4, d_model=8, vocab_size=32, seed=7)
    lens = fit(
        model,
        ["abcdefghij " * 5],
        source_layers=[0, 1, 2],
        dim_batch=4,
        max_seq_len=64,
    )
    return model, lens


@torch.no_grad()
def _activations(
    model: TinyDecoder, prompt: str
) -> tuple[torch.Tensor, dict[int, torch.Tensor]]:
    input_ids = model.encode(prompt, max_length=512)
    hidden = model.embed_tokens(input_ids)
    activations: dict[int, torch.Tensor] = {}
    for layer, block in enumerate(model.layers):
        hidden = block(hidden)
        activations[layer] = hidden[0].float()
    return input_ids, activations


def _build_tiny_fixture(
    model: TinyDecoder,
    lens: JacobianLens,
    prompt: str,
    **kwargs,
) -> dict:
    """Exercise fixture numerics while keeping the production identity guard intact."""
    lens_provenance = {
        "sha256": LENS_SHA,
        "n_prompts": lens.n_prompts,
        "source_layers": lens.source_layers,
        "target_layer": model.n_layers - 1,
        "prompt_indices": list(range(lens.n_prompts)),
        "corpus_manifest_sha256": CORPUS_SHA,
        "dataset": {"id": "tiny/data", "revision": "tiny-revision"},
        "storage_dtype": "float32",
        "acceptance": {
            "schema": "nemotron-jlens-acceptance/v1",
            "tier": "smoke",
            "status": "non-final",
            "is_final": False,
            "exportable": True,
            "checks": {
                "pinned_model": False,
                "pinned_dataset": False,
                "known_manifest": False,
                "complete_prompt_set": False,
                "full_layer_coverage": False,
                "fixed_fit_settings": False,
                "pinned_transformers": True,
                "fp32_storage": True,
            },
            "reasons": ["Tiny-model numerical fixture."],
        },
        "upstream_jlens_commit": UPSTREAM_JLENS_COMMIT,
        "adaptation_source_sha256": SOURCE_SHA,
        "runtime": TEST_RUNTIME,
        "fit": {
            "mamba_backend": "test",
            "patched_mamba_layers": 0,
        },
    }
    lens_metadata = {"mamba_backend": "test", "patched_mamba_layers": 0}
    with patch(
        "nemotron_jlens.fixture._validated_recording_provenance",
        return_value=lens_provenance,
    ):
        return build_fixture(
            model,
            lens,
            prompt,
            model_name=NEMOTRON.model_id,
            model_revision=NEMOTRON.revision,
            lens_metadata=lens_metadata,
            **kwargs,
        )


def test_build_fixture_has_exact_probabilities_ranks_and_schema(
    fitted_tiny: tuple[TinyDecoder, JacobianLens],
):
    model, lens = fitted_tiny
    prompt = "fixture probabilities"
    input_ids, activations = _activations(model, prompt)
    full_len = input_ids.shape[1]
    prompt_len = full_len - 3
    tracked = {"token-g": 7, "token-k": 11}

    fixture = _build_tiny_fixture(
        model,
        lens,
        prompt,
        prompt_len=prompt_len,
        completion="xyz",
        top_n=5,
        tracked_tokens=tracked,
        mask_display=False,
        fixture_metadata={"test_provenance": "TinyDecoder seed=7"},
    )

    # The result is directly serializable and follows the demo's recorded schema.
    json.dumps(fixture, allow_nan=False)
    assert fixture["_fixture"]["schema"] == "nemotron-jlens-fixture/v1"
    assert fixture["_fixture"]["mode"] == "recorded"
    assert fixture["_fixture"]["acceptance"]["tier"] == "smoke"
    assert fixture["_fixture"]["acceptance"]["status"] == "non-final"
    assert fixture["_fixture"]["acceptance"]["is_final"] is False
    assert fixture["_fixture"]["test_provenance"] == "TinyDecoder seed=7"
    assert fixture["_fixture"]["model_revision"] == NEMOTRON.revision
    assert (
        fixture["_fixture"]["model_revision"]
        == fixture["_fixture"]["tokenizer"]["revision"]
        == fixture["_fixture"]["provenance"]["model"]["revision"]
        == fixture["_fixture"]["provenance"]["tokenizer"]["revision"]
    )
    assert (
        fixture["_fixture"]["lens_sha256"]
        == fixture["_fixture"]["lens"]["sha256"]
        == fixture["_fixture"]["provenance"]["lens"]["sha256"]
        == LENS_SHA
    )
    assert (
        fixture["_fixture"]["prompt_sha256"]
        == fixture["_fixture"]["provenance"]["prompt"]["sha256"]
        == sha256_text(prompt)
    )
    assert fixture["_fixture"]["lens"]["runtime"] == TEST_RUNTIME
    assert fixture["_fixture"]["provenance"]["lens"]["runtime"] == TEST_RUNTIME
    assert fixture["_fixture"]["provenance"]["runtime"]["packages"] == (
        TEST_RUNTIME["packages"]
    )
    assert fixture["meta"] == {
        "kind": "meta",
        "model": NEMOTRON.model_id,
        "types": [JACOBIAN_LENS, LOGIT_LENS],
        "layers_by_type": {
            JACOBIAN_LENS: [0, 1, 2, 3],
            LOGIT_LENS: [0, 1, 2, 3],
        },
        "top_n": 5,
        "prompt_len": prompt_len,
        "num_completion_tokens": 3,
        "temperature": 0,
        "prepend_bos": True,
        "reuse_len": 0,
        "window_start": 0,
    }
    assert fixture["done"] == {
        "kind": "done",
        "seq_len": full_len,
        "prompt_len": prompt_len,
        "vocab_size": 32,
        "completion": "xyz",
    }
    assert len(fixture["tokens"]) == full_len

    layers = fixture["meta"]["layers_by_type"][JACOBIAN_LENS]
    for row in fixture["tokens"]:
        position = row["position"]
        assert row["id"] == int(input_ids[0, position])
        assert row["token"] == model.tokenizer.decode([row["id"]])
        assert row["is_generated"] == (position >= prompt_len)
        assert [result["type"] for result in row["results"]] == [
            JACOBIAN_LENS,
            LOGIT_LENS,
        ]

        for result in row["results"]:
            assert len(result["top_tokens"]) == len(layers)
            assert len(result["top_probs"]) == len(layers)
            assert [item["token"] for item in result["tracked"]] == list(tracked)
            for layer_index, layer in enumerate(layers):
                residual = activations[layer][position]
                if result["type"] == JACOBIAN_LENS and layer in lens.jacobians:
                    residual = lens.transport(residual, layer)
                logits = model.unembed(residual).float()
                probabilities = logits.softmax(dim=-1)
                expected_ids = logits.topk(5).indices.tolist()
                expected_tokens = [
                    model.tokenizer.decode([token_id]) for token_id in expected_ids
                ]

                assert result["top_tokens"][layer_index] == expected_tokens
                assert result["top_probs"][layer_index] == pytest.approx(
                    probabilities[expected_ids].tolist(), abs=1e-7
                )
                assert 0 < sum(result["top_probs"][layer_index]) <= 1

                for tracked_row in result["tracked"]:
                    token_id = tracked_row["id"]
                    expected_rank = int((logits > logits[token_id]).sum()) + 1
                    assert tracked_row["ranks"][layer_index] == expected_rank
                    assert tracked_row["probs"][layer_index] == pytest.approx(
                        probabilities[token_id].item(), abs=1e-7
                    )

    # The identity final row is shared by both readouts.
    for row in fixture["tokens"]:
        jacobian, logit = row["results"]
        assert jacobian["top_tokens"][-1] == logit["top_tokens"][-1]
        assert jacobian["top_probs"][-1] == logit["top_probs"][-1]


def test_build_fixture_windows_absolute_positions_and_layers(
    fitted_tiny: tuple[TinyDecoder, JacobianLens],
):
    model, lens = fitted_tiny
    prompt = "window this transcript"
    input_ids = model.encode(prompt, max_length=512)
    full_len = input_ids.shape[1]
    prompt_len = full_len - 2

    fixture = _build_tiny_fixture(
        model,
        lens,
        prompt,
        prompt_len=prompt_len,
        top_n=3,
        layer_stride=2,
        last_n_tokens=5,
        mask_display=False,
    )

    start = full_len - 5
    assert fixture["meta"]["window_start"] == start
    assert fixture["meta"]["layers_by_type"] == {
        JACOBIAN_LENS: [0, 2, 3],
        LOGIT_LENS: [0, 2, 3],
    }
    assert [token["position"] for token in fixture["tokens"]] == list(
        range(start, full_len)
    )
    assert [token["is_generated"] for token in fixture["tokens"]] == [
        position >= prompt_len for position in range(start, full_len)
    ]
    assert fixture["_fixture"]["transcript"] == [
        {
            "label": "Prompt",
            "start": start,
            "end": prompt_len - 1,
            "generated": False,
        },
        {
            "label": "Teacher-forced response",
            "start": prompt_len,
            "end": full_len - 1,
            "generated": True,
        },
    ]
    assert fixture["_fixture"]["ui"]["selected_positions"] == list(
        range(prompt_len, full_len)
    )
    assert fixture["done"]["seq_len"] == full_len

    with pytest.raises(ValueError, match="last_n_tokens must be positive"):
        _build_tiny_fixture(
            model,
            lens,
            prompt,
            last_n_tokens=0,
            mask_display=False,
        )


def test_build_fixture_rejects_non_nemotron_recordings(
    fitted_tiny: tuple[TinyDecoder, JacobianLens],
):
    model, lens = fitted_tiny
    with pytest.raises(ValueError, match="pinned Nemotron identity"):
        build_fixture(
            model,
            lens,
            "guarded fixture",
            model_name="tiny-decoder",
            model_revision="tiny-revision",
            lens_metadata={},
            mask_display=False,
        )

    with pytest.raises(ValueError, match="model shape"):
        build_fixture(
            model,
            lens,
            "guarded fixture",
            model_name=NEMOTRON.model_id,
            model_revision=NEMOTRON.revision,
            lens_metadata={},
            mask_display=False,
        )


def test_readouts_reject_non_finite_logits(fitted_tiny):
    model, _ = fitted_tiny
    logits = torch.zeros((2, 32))
    logits[0, 4] = torch.nan
    with pytest.raises(ValueError, match="non-finite"):
        _top_readout(
            logits,
            tokenizer=model.tokenizer,
            top_n=3,
            mask_display=False,
        )
    with pytest.raises(ValueError, match="non-finite"):
        _tracked_readout(logits, {"token": 4})


def test_fixture_writer_is_strict_and_atomic(tmp_path):
    output = tmp_path / "fixture.json"
    _write_fixture_atomic(output, {"probability": 0.25})
    assert json.loads(output.read_text(encoding="utf-8")) == {"probability": 0.25}
    assert list(tmp_path.glob(".fixture.json.*.tmp")) == []

    with pytest.raises(ValueError, match="Out of range float values"):
        _write_fixture_atomic(output, {"probability": float("nan")})
    assert json.loads(output.read_text(encoding="utf-8")) == {"probability": 0.25}
    assert list(tmp_path.glob(".fixture.json.*.tmp")) == []

def test_export_fixture_cli_forwards_advertised_runtime_flags():
    args = build_parser().parse_args(
        [
            "export-fixture",
            "lens.pt",
            "fixture.json",
            "--dtype",
            "float16",
            "--device-map",
            "cuda",
            "--compile-blocks",
            "--disable-mamba-kernels",
        ]
    )
    with patch(
        "nemotron_jlens.cli.export_fixture", return_value={"ok": True}
    ) as export:
        assert _run(args) == {"ok": True}
    assert export.call_args.kwargs["compile_blocks"] is True
    assert export.call_args.kwargs["disable_mamba_kernels"] is True
    assert export.call_args.kwargs["dtype"] == "float16"
    assert export.call_args.kwargs["device_map"] == "cuda"
