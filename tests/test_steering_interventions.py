"""CPU tests for steering math, validation, direction caching, and hooks."""

from __future__ import annotations

import threading
from collections import namedtuple
from typing import Any

import pytest
import torch
from torch import nn

from nemotron_steering.constants import (
    MAX_INTERVENTION_LAYERS,
    MAX_READOUT_LAYERS,
    VOCAB_SIZE,
)
from nemotron_steering.errors import InferenceCancelled, ValidationError
from nemotron_steering.interventions import (
    DirectionCache,
    ForwardContext,
    HookSession,
    ablate,
    apply_intervention,
    jacobian_direction,
    position_mask,
    steer,
    swap,
    unit_normalize,
)
from nemotron_steering.validation import (
    InterventionSpec,
    token_pieces,
    validate_intervention,
    validate_layers,
    validate_readout_layers,
    validate_single_token_text,
    validate_strength,
)


def _spec(
    *,
    mode: str = "steer",
    lens_type: str = "jacobian",
    layers: tuple[int, ...] = (0,),
    source_token_ids: tuple[int, ...] = (7,),
    target_token_id: int | None = None,
    strength: float = 0.5,
    apply_to_generated: bool = False,
) -> InterventionSpec:
    return InterventionSpec(
        mode=mode,
        lens_type=lens_type,
        layers=layers,
        source_token_ids=source_token_ids,
        target_token_id=target_token_id,
        strength=strength,
        apply_to_generated=apply_to_generated,
    )


def test_jacobian_direction_uses_w_times_j_for_asymmetric_matrix() -> None:
    unembedding = torch.tensor([1.0, 2.0])
    jacobian = torch.tensor([[1.0, 2.0], [3.0, 7.0]])

    actual = jacobian_direction(unembedding, jacobian)
    expected = unit_normalize(torch.tensor([7.0, 16.0]))
    reversed_orientation = unit_normalize(jacobian @ unembedding)

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert not torch.allclose(actual, reversed_orientation)
    assert actual.device.type == "cpu"
    assert actual.dtype == torch.float32


@pytest.mark.parametrize(
    ("unembedding", "jacobian"),
    [
        (torch.ones(1, 2), torch.eye(2)),
        (torch.ones(2), torch.ones(2)),
        (torch.ones(2), torch.eye(3)),
    ],
)
def test_jacobian_direction_rejects_invalid_shapes(
    unembedding: torch.Tensor, jacobian: torch.Tensor
) -> None:
    with pytest.raises(ValidationError, match="direction inputs|Jacobian shape"):
        jacobian_direction(unembedding, jacobian)


def test_zero_strength_steer_is_exact_identity() -> None:
    hidden = torch.tensor([[[3.0, 4.0], [5.0, 12.0]]])
    result = steer(hidden, torch.tensor([1.0, 0.0]), 0.0)

    assert result is hidden
    assert torch.equal(result, hidden)


def test_steer_scales_by_each_positions_exact_l2_norm() -> None:
    hidden = torch.tensor([[[3.0, 4.0], [0.0, 5.0]]])
    result = steer(hidden, torch.tensor([1.0, 0.0]), 0.5)

    expected = torch.tensor([[[5.5, 4.0], [2.5, 5.0]]])
    torch.testing.assert_close(result, expected, rtol=0, atol=0)


def test_steer_changes_only_masked_positions_without_mutating_input() -> None:
    hidden = torch.tensor([[[3.0, 4.0], [0.0, 5.0]]])
    original = hidden.clone()
    mask = torch.tensor([[False, True]])

    result = steer(hidden, torch.tensor([1.0, 0.0]), 0.5, mask=mask)

    expected = torch.tensor([[[3.0, 4.0], [2.5, 5.0]]])
    torch.testing.assert_close(result, expected, rtol=0, atol=0)
    assert torch.equal(hidden, original)


def test_ablation_removes_source_projection_per_position() -> None:
    hidden = torch.tensor([[[7.0, -1.0, 2.0], [-3.0, 8.0, 5.0]]], dtype=torch.float32)
    direction = torch.tensor([3.0, 4.0, 0.0])

    result = ablate(hidden, direction)
    normalized = unit_normalize(direction)
    remaining_projection = (result * normalized).sum(dim=-1)

    torch.testing.assert_close(
        remaining_projection, torch.zeros_like(remaining_projection), atol=1e-6, rtol=0
    )


def test_swap_matches_exact_neuronpedia_one_way_equation() -> None:
    hidden = torch.tensor([[[3.0, 4.0], [-2.0, 5.0]]])
    source = torch.tensor([2.0, 0.0])
    target = torch.tensor([1.0, 1.0])
    source_hat = unit_normalize(source)
    target_hat = unit_normalize(target)
    coefficient = (hidden * source_hat).sum(dim=-1, keepdim=True)
    expected = hidden - coefficient * source_hat + coefficient * target_hat

    result = swap(hidden, source, target)

    torch.testing.assert_close(result, expected, atol=1e-6, rtol=0)


def test_apply_intervention_swap_ignores_strength() -> None:
    hidden = torch.tensor([[[3.0, 4.0]]])
    source = torch.tensor([1.0, 0.0])
    target = torch.tensor([0.0, -1.0])
    spec = _spec(mode="swap", target_token_id=8, strength=2.0)

    result = apply_intervention(hidden, spec, source, target_direction=target)

    torch.testing.assert_close(result, torch.tensor([[[0.0, 1.0]]]), rtol=0, atol=0)


def test_apply_intervention_swap_requires_target_direction() -> None:
    with pytest.raises(ValidationError, match="target direction"):
        apply_intervention(
            torch.ones(1, 1, 2),
            _spec(mode="swap", target_token_id=8),
            torch.tensor([1.0, 0.0]),
        )


def test_zero_directions_are_stable_no_ops() -> None:
    hidden = torch.tensor([[[3.0, 4.0]]])
    zero = torch.zeros(2)
    nonzero = torch.tensor([1.0, 0.0])

    assert torch.equal(unit_normalize(zero), zero)
    assert torch.equal(jacobian_direction(nonzero, torch.zeros(2, 2)), zero)
    assert steer(hidden, zero, 1.0) is hidden
    assert ablate(hidden, zero) is hidden
    assert swap(hidden, zero, nonzero) is hidden
    assert swap(hidden, nonzero, zero) is hidden


@pytest.mark.parametrize(
    "bad", [torch.tensor([float("nan")]), torch.tensor([float("inf")])]
)
def test_unit_normalize_rejects_non_finite_directions(bad: torch.Tensor) -> None:
    with pytest.raises(ValidationError, match="NaN or Inf"):
        unit_normalize(bad)


def test_position_mask_excludes_exact_bos_and_generated_positions() -> None:
    input_ids = torch.tensor([[1, 7, 8, 9, 1], [6, 1, 8, 9, 10]], dtype=torch.long)

    prompt_only = position_mask(
        input_ids,
        original_prompt_length=3,
        bos_token_id=1,
        apply_to_generated=False,
    )
    include_generated = position_mask(
        input_ids,
        original_prompt_length=3,
        bos_token_id=1,
        apply_to_generated=True,
    )

    assert torch.equal(
        prompt_only,
        torch.tensor(
            [[False, True, True, False, False], [True, False, True, False, False]]
        ),
    )
    assert torch.equal(
        include_generated,
        torch.tensor(
            [[False, True, True, True, False], [True, False, True, True, True]]
        ),
    )


def test_position_mask_without_bos_id_only_applies_scope() -> None:
    input_ids = torch.tensor([[1, 2, 3, 4]])
    mask = position_mask(
        input_ids,
        original_prompt_length=2,
        bos_token_id=None,
        apply_to_generated=False,
    )
    assert torch.equal(mask, torch.tensor([[True, True, False, False]]))


@pytest.mark.parametrize(
    ("input_ids", "prompt_length"),
    [(torch.ones(3), 1), (torch.ones(1, 2), -1), (torch.ones(1, 2), 3)],
)
def test_position_mask_rejects_invalid_shapes_and_lengths(
    input_ids: torch.Tensor, prompt_length: int
) -> None:
    with pytest.raises(ValidationError, match="input_ids|original_prompt_length"):
        position_mask(
            input_ids,
            original_prompt_length=prompt_length,
            bos_token_id=None,
            apply_to_generated=False,
        )


def test_mask_shape_mismatch_fails_closed() -> None:
    with pytest.raises(ValidationError, match="position mask shape"):
        steer(
            torch.ones(1, 2, 3),
            torch.tensor([1.0, 0.0, 0.0]),
            1.0,
            mask=torch.ones(1, 3, dtype=torch.bool),
        )


class _FakeTokenizer:
    def __init__(self) -> None:
        self.calls: list[tuple[str, bool]] = []
        self.encodings = {
            " whale": [101],
            "blue whale": [102, 101],
            "fish": [103],
        }
        self.decodings = {101: " whale", 102: "blue", 103: "fish"}

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        self.calls.append((text, add_special_tokens))
        return self.encodings[text]

    def decode(self, token_ids: list[int]) -> str:
        assert len(token_ids) == 1
        return self.decodings[token_ids[0]]


def test_token_pieces_preserves_exact_text_ids_and_leading_space() -> None:
    tokenizer = _FakeTokenizer()

    result = token_pieces(tokenizer, " whale")

    assert result == {
        "text": " whale",
        "token_ids": [101],
        "pieces": [{"id": 101, "text": " whale"}],
        "is_single_token": True,
    }
    assert tokenizer.calls == [(" whale", False)]


def test_multitoken_typed_target_is_rejected_with_pieces() -> None:
    tokenizer = _FakeTokenizer()

    with pytest.raises(ValidationError, match="exactly one token") as exc_info:
        validate_single_token_text(
            tokenizer, "blue whale", field="target_token", expected_id=101
        )

    assert exc_info.value.details == {
        "text": "blue whale",
        "token_ids": [102, 101],
        "pieces": [
            {"id": 102, "text": "blue"},
            {"id": 101, "text": " whale"},
        ],
        "is_single_token": False,
    }
    assert tokenizer.calls == [("blue whale", False)]


def test_numeric_token_id_remains_authoritative_over_typed_text() -> None:
    tokenizer = _FakeTokenizer()

    with pytest.raises(ValidationError, match="authoritative numeric") as exc_info:
        validate_single_token_text(
            tokenizer, "fish", field="source_token", expected_id=101
        )

    assert exc_info.value.details["token_ids"] == [103]
    assert exc_info.value.details["expected_id"] == 101


@pytest.mark.parametrize(
    "layers",
    [
        [],
        [51],
        [-1],
        [0, 0],
        [True],
        ["0"],
        0,
        "0",
    ],
)
def test_invalid_intervention_layers_fail_closed(layers: Any) -> None:
    with pytest.raises(ValidationError):
        validate_layers(layers)


def test_valid_layers_are_canonicalized_and_block_50_is_writable() -> None:
    assert validate_layers([50, 0, 26]) == (0, 26, 50)


def test_all_fitted_source_layers_are_valid_for_intervention() -> None:
    layers = list(reversed(range(MAX_INTERVENTION_LAYERS)))

    assert validate_layers(layers) == tuple(range(51))


def test_intervention_layer_count_overflow_fails_closed() -> None:
    with pytest.raises(ValidationError, match="at most 51 layers"):
        validate_layers(list(range(MAX_INTERVENTION_LAYERS + 1)))


def test_all_model_layers_are_valid_for_readout() -> None:
    layers = list(reversed(range(MAX_READOUT_LAYERS)))

    assert validate_readout_layers(layers) == tuple(range(52))


def test_readout_layer_count_overflow_fails_closed() -> None:
    with pytest.raises(ValidationError, match="1-52 layer integers"):
        validate_readout_layers(list(range(MAX_READOUT_LAYERS + 1)))


@pytest.mark.parametrize(
    "target",
    [None, -1, VOCAB_SIZE, True, 1.0, "101"],
)
def test_swap_rejects_missing_or_invalid_numeric_target(target: Any) -> None:
    with pytest.raises(ValidationError, match="target_token_id"):
        validate_intervention(
            {
                "mode": "swap",
                "lens_type": "jacobian",
                "layers": [0],
                "source_token_ids": [7],
                "target_token_id": target,
            }
        )


def test_swap_rejects_multiple_sources() -> None:
    with pytest.raises(ValidationError, match="exactly one source"):
        validate_intervention(
            {
                "mode": "swap",
                "lens_type": "jacobian",
                "layers": [0],
                "source_token_ids": [7, 8],
                "target_token_id": 9,
            }
        )


def test_non_swap_rejects_target_token() -> None:
    with pytest.raises(ValidationError, match="accepted only for swap"):
        validate_intervention(
            {
                "mode": "steer",
                "lens_type": "jacobian",
                "layers": [0],
                "source_token_ids": [7],
                "target_token_id": 9,
            }
        )


@pytest.mark.parametrize("strength", [-2.0, 0, 2.0])
def test_strength_accepts_finite_closed_range(strength: float) -> None:
    assert validate_strength(strength) == float(strength)


@pytest.mark.parametrize(
    "strength", [-2.0001, 2.0001, float("nan"), float("inf"), True, "0.1"]
)
def test_strength_rejects_invalid_values(strength: Any) -> None:
    with pytest.raises(ValidationError, match="strength"):
        validate_strength(strength)


def test_validate_intervention_sets_defaults_and_sorts_layers() -> None:
    actual = validate_intervention(
        {
            "mode": "ablate",
            "lens_type": "logit",
            "layers": [7, 3],
            "source_token_ids": [101],
        }
    )
    assert actual == InterventionSpec(
        mode="ablate",
        lens_type="logit",
        layers=(3, 7),
        source_token_ids=(101,),
        strength=-0.1,
        apply_to_generated=False,
    )


def test_direction_cache_is_bounded_and_uses_lru_eviction() -> None:
    lm_head = nn.Linear(3, 4, bias=False)
    with torch.no_grad():
        lm_head.weight.copy_(
            torch.tensor(
                [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 1.0, 0.0]]
            )
        )
    cache = DirectionCache(lm_head=lm_head, jacobians={}, max_entries=2)

    first = cache.get("logit", 0, 0)
    evicted = cache.get("logit", 1, 0)
    assert cache.get("logit", 0, 0) is first
    cache.get("logit", 2, 0)
    assert len(cache) == 2

    with torch.no_grad():
        lm_head.weight[1].copy_(torch.tensor([1.0, 1.0, 0.0]))
    recomputed = cache.get("logit", 1, 0)

    assert recomputed is not evicted
    torch.testing.assert_close(
        recomputed, unit_normalize(torch.tensor([1.0, 1.0, 0.0]))
    )
    assert len(cache) == 2


def test_direction_cache_key_includes_layer_and_lens_type() -> None:
    lm_head = nn.Linear(2, 1, bias=False)
    with torch.no_grad():
        lm_head.weight.copy_(torch.tensor([[1.0, 0.0]]))
    cache = DirectionCache(
        lm_head=lm_head,
        jacobians={0: torch.eye(2), 1: torch.tensor([[0.0, 1.0], [1.0, 0.0]])},
        max_entries=4,
    )

    logit_layer_0 = cache.get("logit", 0, 0)
    logit_layer_1 = cache.get("logit", 0, 1)
    jacobian_layer_1 = cache.get("jacobian", 0, 1)

    assert logit_layer_0 is not logit_layer_1
    assert len(cache) == 3
    torch.testing.assert_close(jacobian_layer_1, torch.tensor([0.0, 1.0]))


def test_direction_cache_rejects_unknown_lens_and_missing_jacobian() -> None:
    cache = DirectionCache(
        lm_head=nn.Linear(2, 2, bias=False), jacobians={}, max_entries=2
    )
    with pytest.raises(ValidationError, match="unknown lens type"):
        cache.get("unknown", 0, 0)
    with pytest.raises(ValidationError, match="no fitted Jacobian"):
        cache.get("jacobian", 0, 0)


def test_combined_multi_source_sums_unit_directions_without_renormalizing() -> None:
    lm_head = nn.Linear(2, 3, bias=False)
    with torch.no_grad():
        lm_head.weight.copy_(torch.tensor([[3.0, 0.0], [0.0, 4.0], [-2.0, 0.0]]))
    cache = DirectionCache(lm_head=lm_head, jacobians={}, max_entries=4)
    spec = _spec(lens_type="logit", source_token_ids=(0, 1), strength=0.25)

    source, target = cache.combined(spec, layer=0)

    assert target is None
    torch.testing.assert_close(source, torch.tensor([1.0, 1.0]), rtol=0, atol=0)
    torch.testing.assert_close(
        torch.linalg.vector_norm(source), torch.tensor(2.0).sqrt(), rtol=0, atol=0
    )


class _TensorBlock(nn.Module):
    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return hidden


class _TupleBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.aux = object()

    def forward(self, hidden: torch.Tensor) -> tuple[torch.Tensor, object]:
        return hidden, self.aux


_NamedOutput = namedtuple("_NamedOutput", ["hidden", "aux"])


class _NamedTupleBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.aux = object()

    def forward(self, hidden: torch.Tensor) -> _NamedOutput:
        return _NamedOutput(hidden=hidden, aux=self.aux)


def _hook_context() -> ForwardContext:
    context = ForwardContext(
        original_prompt_length=2,
        bos_token_id=1,
        apply_to_generated=False,
    )
    context.set_input_ids(torch.tensor([[1, 5, 6]]))
    return context


def _hook_session(block: nn.Module, **kwargs: Any) -> HookSession:
    return HookSession(
        layers=[block],
        context=_hook_context(),
        intervention=_spec(),
        directions={0: (torch.tensor([1.0, 0.0]), None)},
        capture_layers=(0,),
        **kwargs,
    )


def test_tensor_hook_returns_and_captures_modified_output_then_cleans_up() -> None:
    block = _TensorBlock()
    hidden = torch.tensor([[[1.0, 0.0], [3.0, 4.0], [0.0, 2.0]]])
    original = hidden.clone()
    session = _hook_session(block)

    with session:
        result = block(hidden)
        assert len(block._forward_hooks) == 2

    expected = torch.tensor([[[1.0, 0.0], [5.5, 4.0], [0.0, 2.0]]])
    assert isinstance(result, torch.Tensor)
    torch.testing.assert_close(result, expected, rtol=0, atol=0)
    torch.testing.assert_close(session.captures[0], expected, rtol=0, atol=0)
    assert torch.equal(hidden, original)
    assert len(block._forward_hooks) == 0


def test_tuple_hook_preserves_tail_and_captures_modified_output() -> None:
    block = _TupleBlock()
    hidden = torch.tensor([[[1.0, 0.0], [3.0, 4.0], [0.0, 2.0]]])
    session = _hook_session(block)

    with session:
        result = block(hidden)

    expected = torch.tensor([[[1.0, 0.0], [5.5, 4.0], [0.0, 2.0]]])
    assert type(result) is tuple
    assert result[1] is block.aux
    torch.testing.assert_close(result[0], expected, rtol=0, atol=0)
    torch.testing.assert_close(session.captures[0], expected, rtol=0, atol=0)
    assert len(block._forward_hooks) == 0


def test_named_tuple_hook_preserves_output_type_and_fields() -> None:
    block = _NamedTupleBlock()
    hidden = torch.tensor([[[1.0, 0.0], [3.0, 4.0], [0.0, 2.0]]])

    with _hook_session(block):
        result = block(hidden)

    assert isinstance(result, _NamedOutput)
    assert result.aux is block.aux
    torch.testing.assert_close(
        result.hidden,
        torch.tensor([[[1.0, 0.0], [5.5, 4.0], [0.0, 2.0]]]),
        rtol=0,
        atol=0,
    )
    assert len(block._forward_hooks) == 0


def test_hook_session_removes_hooks_when_body_raises() -> None:
    block = _TensorBlock()
    session = HookSession(layers=[block], context=_hook_context(), capture_layers=(0,))

    with pytest.raises(RuntimeError, match="body failed"):
        with session:
            assert len(block._forward_hooks) == 1
            raise RuntimeError("body failed")

    assert len(block._forward_hooks) == 0


def test_hook_session_removes_hooks_when_hook_raises() -> None:
    block = _TensorBlock()
    context = ForwardContext(
        original_prompt_length=1,
        bos_token_id=None,
        apply_to_generated=False,
    )
    session = HookSession(
        layers=[block],
        context=context,
        intervention=_spec(),
        directions={0: (torch.tensor([1.0, 0.0]), None)},
    )

    with pytest.raises(RuntimeError, match="no current input_ids"):
        with session:
            block(torch.ones(1, 1, 2))

    assert len(block._forward_hooks) == 0


def test_hook_session_cancellation_raises_and_removes_all_hooks() -> None:
    block = _TensorBlock()
    cancelled = threading.Event()
    cancelled.set()
    session = _hook_session(block, cancel_event=cancelled)

    with pytest.raises(InferenceCancelled, match="request cancelled"):
        with session:
            block(torch.ones(1, 3, 2))

    assert session.captures == {}
    assert len(block._forward_hooks) == 0


class _TrackingHandle:
    def __init__(self, owner: _RegistrationLayer, hook: Any) -> None:
        self.owner = owner
        self.hook = hook
        self.remove_calls = 0

    def remove(self) -> None:
        self.remove_calls += 1
        self.owner.active.remove(self.hook)


class _RegistrationLayer:
    def __init__(self, *, fail_on: int) -> None:
        self.fail_on = fail_on
        self.attempts = 0
        self.active: list[Any] = []
        self.handles: list[_TrackingHandle] = []

    def register_forward_hook(self, hook: Any) -> _TrackingHandle:
        self.attempts += 1
        if self.attempts == self.fail_on:
            raise RuntimeError("registration failed")
        self.active.append(hook)
        handle = _TrackingHandle(self, hook)
        self.handles.append(handle)
        return handle


def test_partial_hook_registration_failure_rolls_back_registered_hooks() -> None:
    layer = _RegistrationLayer(fail_on=2)
    session = HookSession(
        layers=[layer],
        context=_hook_context(),
        intervention=_spec(),
        directions={0: (torch.tensor([1.0, 0.0]), None)},
        capture_layers=(0,),
    )

    with pytest.raises(RuntimeError, match="registration failed"):
        session.__enter__()

    assert layer.attempts == 2
    assert layer.active == []
    assert len(layer.handles) == 1
    assert layer.handles[0].remove_calls == 1
    assert session._handles == []


def test_hook_session_close_is_idempotent() -> None:
    block = _TensorBlock()
    session = HookSession(layers=[block], context=_hook_context(), capture_layers=(0,))

    session.__enter__()
    session.close()
    session.close()

    assert len(block._forward_hooks) == 0
