from types import SimpleNamespace

import pytest

from nemotron_jlens.preflight import check_native_logit_parity, run_vjp_preflight

from .tiny import TinyDecoder


class _NativeModel:
    def __init__(self, adapted, *, logit_offset=0.0):
        self.adapted = adapted
        self.logit_offset = logit_offset

    def __call__(self, *, input_ids, use_cache):
        assert use_cache is False
        hidden = self.adapted.forward(input_ids).last_hidden_state
        logits = self.adapted.unembed(hidden) + self.logit_offset
        return SimpleNamespace(logits=logits)


def test_vjp_preflight_runs_full_tiny_stack_with_frozen_parameters():
    model = TinyDecoder(n_layers=4, d_model=8)
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    result = run_vjp_preflight(
        model,
        "the quick brown fox",
        source_layer=0,
        target_layer=-1,
        max_seq_len=32,
    )
    assert result["ok"] is True
    assert result["source_layer"] == 0
    assert result["target_layer"] == 3
    assert result["d_model"] == 8
    assert result["seq_len"] > 2
    assert result["gradient_norm"] > 0
    assert result["gradient_abs_max"] > 0
    assert result["source_device"] == "cpu"
    assert result["target_device"] == "cpu"


@pytest.mark.parametrize(
    ("source_layer", "target_layer"),
    [(0, 0), (3, 2), (-5, -1), (0, 4)],
)
def test_vjp_preflight_rejects_invalid_layer_pairs(source_layer, target_layer):
    with pytest.raises(ValueError, match="source_layer < target_layer"):
        run_vjp_preflight(
            TinyDecoder(n_layers=4),
            "enough tokens",
            source_layer=source_layer,
            target_layer=target_layer,
        )


def test_vjp_preflight_rejects_single_token_prompt():
    with pytest.raises(ValueError, match="fewer than two tokens"):
        run_vjp_preflight(TinyDecoder(), "")


def test_native_logit_parity_accepts_matching_adapter_and_rejects_drift():
    model = TinyDecoder(n_layers=4, d_model=8)
    matching = check_native_logit_parity(
        _NativeModel(model), model, "matching logits", atol=1e-6
    )
    assert matching["ok"] is True
    assert matching["max_abs_error"] <= 1e-6
    assert matching["mean_abs_error"] <= 1e-7
    assert matching["cosine_similarity"] == pytest.approx(1.0, abs=1e-6)

    with pytest.raises(RuntimeError, match="does not reproduce native logits"):
        check_native_logit_parity(
            _NativeModel(model, logit_offset=0.25),
            model,
            "different logits",
            atol=0.05,
        )
