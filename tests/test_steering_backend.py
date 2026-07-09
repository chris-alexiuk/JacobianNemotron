"""CPU integration tests for the persistent steering backend and coordinator."""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from accelerate.hooks import AlignDevicesHook, add_hook_to_module
from torch import nn

import nemotron_steering.backend as backend_module
from nemotron_steering.backend import SteeringBackend
from nemotron_steering.constants import MODEL_ID, MODEL_REVISION
from nemotron_steering.errors import InferenceBusy, InferenceCancelled, ValidationError
from nemotron_steering.interventions import ForwardContext, HookSession
from nemotron_steering.provenance import ValidatedLens
from nemotron_steering.requests import GenerationSpec, InferenceRequest
from nemotron_steering.service import InferenceService
from nemotron_steering.validation import InterventionSpec


class TinyTokenizer:
    """Small tokenizer with exact whitespace pieces and a chat template."""

    bos_token_id = 1
    eos_token_id = None

    def __init__(self) -> None:
        self.pieces = {
            0: "<pad>",
            1: "<BOS>",
            2: "alpha",
            3: " beta",
            4: " clean",
            5: " shifted",
            6: " topic",
            7: " target",
            8: "<user>",
            9: "<assistant>",
            10: "?",
            11: " end",
        }
        self.text_to_ids = {
            "alpha": [2],
            " beta": [3],
            "alpha beta": [2, 3],
            " clean": [4],
            " shifted": [5],
            " topic": [6],
            " target": [7],
            "?": [10],
        }
        self.chat_calls: list[dict[str, Any]] = []

    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        assert add_special_tokens is False
        return list(self.text_to_ids.get(text, [10]))

    def __call__(
        self,
        text: str,
        *,
        return_tensors: str,
        add_special_tokens: bool,
        truncation: bool,
    ) -> dict[str, torch.Tensor]:
        assert return_tensors == "pt"
        assert add_special_tokens is True
        assert truncation is False
        ids = self.text_to_ids.get(text)
        if ids is None:
            ids = [10]
        return {"input_ids": torch.tensor([[self.bos_token_id, *ids]])}

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
        enable_thinking: bool,
        return_tensors: str,
    ) -> torch.Tensor:
        self.chat_calls.append(
            {
                "messages": messages,
                "tokenize": tokenize,
                "add_generation_prompt": add_generation_prompt,
                "enable_thinking": enable_thinking,
                "return_tensors": return_tensors,
            }
        )
        content_ids: list[int] = []
        for message in messages:
            content_ids.extend(self.text_to_ids.get(message["content"], [10]))
        return torch.tensor([[self.bos_token_id, 8, *content_ids, 9]])

    def decode(self, token_ids: list[int], **_kwargs: Any) -> str:
        return "".join(self.pieces[int(token_id)] for token_id in token_ids)


class MixingLayer(nn.Module):
    """Causal mixing makes an early hook observable at downstream logits."""

    def __init__(self, width: int, index: int) -> None:
        super().__init__()
        matrix = torch.eye(width)
        matrix[index % width, (index + 1) % width] = 0.15 + index * 0.05
        self.register_buffer("matrix", matrix)
        self.mix = 0.08 + index * 0.03

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        mixed = hidden + self.mix * torch.cumsum(hidden, dim=1)
        return mixed @ self.matrix.T


class TinyBackbone(nn.Module):
    def __init__(self, *, block_on_call: int | None = None) -> None:
        super().__init__()
        width = 4
        self.embedding = nn.Embedding(12, width)
        with torch.no_grad():
            for token_id in range(12):
                self.embedding.weight[token_id] = torch.tensor(
                    [
                        0.2 + token_id * 0.07,
                        ((token_id % 3) - 1) * 0.3,
                        0.1 + (token_id % 4) * 0.11,
                        -0.2 + token_id * 0.025,
                    ]
                )
        self.layers = nn.ModuleList(MixingLayer(width, index) for index in range(3))
        self.block_on_call = block_on_call
        self.block_entered = threading.Event()
        self.block_release = threading.Event()
        self._call_lock = threading.Lock()
        self.call_count = 0

    def forward(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        use_cache: bool,
        return_dict: bool,
    ) -> SimpleNamespace:
        assert torch.equal(attention_mask, torch.ones_like(input_ids))
        assert use_cache is False
        assert return_dict is True
        with self._call_lock:
            self.call_count += 1
            call_number = self.call_count
        if self.block_on_call == call_number:
            self.block_entered.set()
            if not self.block_release.wait(timeout=10):
                raise RuntimeError("test backbone was not released")
        hidden = self.embedding(input_ids)
        for layer in self.layers:
            hidden = layer(hidden)
        return SimpleNamespace(last_hidden_state=hidden)


class TinyLensModel:
    input_device = torch.device("cpu")

    def __init__(self, lm_head: nn.Linear) -> None:
        self.lm_head = lm_head

    def unembed(self, residual: torch.Tensor) -> torch.Tensor:
        return self.lm_head(residual.to(self.lm_head.weight.dtype))


def make_tiny_backend(*, block_on_call: int | None = None) -> SteeringBackend:
    """Construct an HF-shaped model and ValidatedLens-like CPU bundle."""
    backbone = TinyBackbone(block_on_call=block_on_call)
    lm_head = nn.Linear(4, 12, bias=False)
    with torch.no_grad():
        lm_head.weight.copy_(
            torch.tensor(
                [
                    [0.00, 0.00, 0.00, 0.00],
                    [0.15, -0.10, 0.05, 0.08],
                    [0.80, -0.30, 0.20, 0.10],
                    [0.35, 0.75, -0.20, 0.15],
                    [0.10, -0.45, 0.95, 0.30],
                    [-0.30, 0.20, 0.40, 0.85],
                    [1.00, 0.50, -0.25, 0.75],
                    [-0.55, 0.90, 0.15, 0.20],
                    [0.25, 0.15, 0.65, -0.40],
                    [0.45, -0.20, 0.10, 0.70],
                    [-0.10, 0.30, 0.20, -0.15],
                    [0.60, 0.10, -0.35, 0.45],
                ]
            )
        )
    tokenizer = TinyTokenizer()
    hf_model = SimpleNamespace(
        backbone=backbone,
        lm_head=lm_head,
        generation_config=SimpleNamespace(eos_token_id=None),
    )
    lens_model = TinyLensModel(lm_head)
    loaded = SimpleNamespace(
        hf_model=hf_model,
        tokenizer=tokenizer,
        lens_model=lens_model,
        model_id=MODEL_ID,
        revision=MODEL_REVISION,
        mamba_backend="fused-or-auto",
        patched_mamba_layers=0,
        runtime_identity={"test": "tiny-cpu"},
    )
    jacobians = {
        0: torch.tensor(
            [
                [1.0, 0.2, 0.0, 0.0],
                [0.0, 0.9, 0.1, 0.0],
                [0.0, 0.0, 1.1, 0.3],
                [0.1, 0.0, 0.0, 0.8],
            ]
        ),
        1: torch.eye(4),
        2: torch.eye(4),
    }
    metadata = {
        "runtime": loaded.runtime_identity,
        "acceptance": {"tier": "pilot", "status": "accepted", "is_final": False},
        "n_prompts": 100,
        "adaptation_source_sha256": "tiny-fit-source",
    }
    bundle = ValidatedLens(
        lens=SimpleNamespace(jacobians=jacobians),
        metadata=metadata,
        validation={"status": "tiny-test"},
        application_source_sha256="tiny-application-source",
    )
    return SteeringBackend(loaded, bundle, strict_identity=False)


def make_request(
    *,
    intervention: InterventionSpec | None = None,
    max_new_tokens: int = 2,
    sampling: bool = False,
    seed: int = 17,
    layers: tuple[int, ...] = (0, 1),
) -> InferenceRequest:
    return InferenceRequest(
        prompt="alpha beta",
        messages=None,
        layers=layers,
        top_k=4,
        generation=GenerationSpec(
            max_new_tokens=max_new_tokens,
            sampling=sampling,
            temperature=0.85,
            top_p=0.9,
            seed=seed,
        ),
        intervention=intervention,
    )


def make_spec(
    *,
    layer: int = 0,
    strength: float = 0.65,
    apply_to_generated: bool = False,
) -> InterventionSpec:
    return InterventionSpec(
        mode="steer",
        lens_type="logit",
        layers=(layer,),
        source_token_ids=(6,),
        strength=strength,
        apply_to_generated=apply_to_generated,
    )


def _hooked_forward(
    backend: SteeringBackend,
    input_ids: torch.Tensor,
    *,
    intervention: InterventionSpec | None,
    prompt_length: int,
    capture_layers: tuple[int, ...],
) -> tuple[torch.Tensor, dict[int, torch.Tensor]]:
    context = ForwardContext(
        original_prompt_length=prompt_length,
        bos_token_id=backend.tokenizer.bos_token_id,
        apply_to_generated=(
            intervention.apply_to_generated if intervention is not None else False
        ),
        input_ids=input_ids,
    )
    directions = backend._direction_map(intervention)
    with HookSession(
        backend.layers,
        context,
        intervention=intervention,
        directions=directions,
        capture_layers=capture_layers,
    ) as hooks:
        logits = backend._next_logits(input_ids)
        captures = {layer: value.clone() for layer, value in hooks.captures.items()}
    return logits, captures


def _stable_condition(condition: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in condition.items()
        if key not in {"name", "elapsed_seconds"}
    }


def test_capture_is_post_intervention_and_downstream_logits_change() -> None:
    backend = make_tiny_backend()
    input_ids = torch.tensor([[1, 2, 3]])
    clean_logits, clean = _hooked_forward(
        backend,
        input_ids,
        intervention=None,
        prompt_length=3,
        capture_layers=(0, 1),
    )
    changed_logits, changed = _hooked_forward(
        backend,
        input_ids,
        intervention=make_spec(layer=0),
        prompt_length=3,
        capture_layers=(0, 1),
    )

    # The BOS is ineligible, while the capture on layer 0 sees the returned h'.
    torch.testing.assert_close(changed[0][:, 0], clean[0][:, 0], rtol=0, atol=0)
    assert not torch.equal(changed[0][:, 1:], clean[0][:, 1:])
    assert not torch.equal(changed[1], clean[1])
    assert not torch.equal(changed_logits, clean_logits)
    assert all(not layer._forward_hooks for layer in backend.layers)


def test_clean_intervention_clean_cycle_has_no_state_or_hook_leakage() -> None:
    backend = make_tiny_backend()
    baseline_request = make_request()

    first = backend.run(baseline_request, paired=False)
    backend.run(make_request(intervention=make_spec(strength=-0.8)), paired=True)
    second = backend.run(baseline_request, paired=False)

    assert _stable_condition(first["clean"]) == _stable_condition(second["clean"])
    assert first["diagnostics"]["hooks_after"] == 0
    assert second["diagnostics"]["hooks_after"] == 0
    assert all(not layer._forward_hooks for layer in backend.layers)


def test_prompt_only_and_generated_position_scopes_are_distinct() -> None:
    backend = make_tiny_backend()
    fixed_prefix = torch.tensor([[1, 2, 3, 4]])
    _, clean = _hooked_forward(
        backend,
        fixed_prefix,
        intervention=None,
        prompt_length=3,
        capture_layers=(2,),
    )
    _, prompt_only = _hooked_forward(
        backend,
        fixed_prefix,
        intervention=make_spec(layer=2, apply_to_generated=False),
        prompt_length=3,
        capture_layers=(2,),
    )
    _, with_generated = _hooked_forward(
        backend,
        fixed_prefix,
        intervention=make_spec(layer=2, apply_to_generated=True),
        prompt_length=3,
        capture_layers=(2,),
    )

    torch.testing.assert_close(prompt_only[2][:, 0], clean[2][:, 0], rtol=0, atol=0)
    assert not torch.equal(prompt_only[2][:, 1:3], clean[2][:, 1:3])
    torch.testing.assert_close(prompt_only[2][:, 3], clean[2][:, 3], rtol=0, atol=0)
    assert not torch.equal(with_generated[2][:, 3], clean[2][:, 3])


def test_paired_sampling_resets_the_same_local_seed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = make_tiny_backend()

    def sample_from_rng_only(
        _logits: torch.Tensor, _spec: GenerationSpec, rng: torch.Generator
    ) -> int:
        return int(torch.randint(2, 12, (1,), generator=rng).item())

    monkeypatch.setattr(
        SteeringBackend, "_sample_token", staticmethod(sample_from_rng_only)
    )
    request = make_request(
        intervention=make_spec(strength=0.9),
        max_new_tokens=5,
        sampling=True,
        seed=123456,
    )

    first = backend.run(request, paired=True)
    second = backend.run(request, paired=True)

    assert (
        first["clean"]["completion_token_ids"]
        == first["intervened"]["completion_token_ids"]
    )
    assert (
        first["clean"]["completion_token_ids"]
        == second["clean"]["completion_token_ids"]
    )


def test_zero_strength_endpoint_path_has_exact_condition_parity() -> None:
    backend = make_tiny_backend()
    result = backend.run(
        make_request(intervention=make_spec(strength=0.0), max_new_tokens=3),
        paired=True,
    )

    assert _stable_condition(result["clean"]) == _stable_condition(result["intervened"])
    assert result["diagnostics"]["hooks_before"] == 0
    assert result["diagnostics"]["hooks_after"] == 0


def test_selected_readout_matches_full_projection_with_norm_bias_and_softcap() -> (
    None
):
    backend = make_tiny_backend()
    backend.lm_head.bias = nn.Parameter(
        torch.linspace(-0.25, 0.3, backend.lm_head.weight.shape[0])
    )
    backend.final_norm = nn.LayerNorm(4)
    with torch.no_grad():
        backend.final_norm.weight.copy_(torch.tensor([0.8, 1.1, 0.9, 1.2]))
        backend.final_norm.bias.copy_(torch.tensor([0.1, -0.05, 0.02, -0.03]))
    backend.lens_model._logit_softcap = 2.5
    residual = torch.linspace(-1.5, 2.0, 35 * 4).reshape(35, 4)
    token_ids = (7, 2, 7, 0)

    expected = backend._unembed_readout(residual)[:, list(token_ids)].float().cpu()
    observed = backend.selected_readout(residual, token_ids)

    assert observed.shape == (35, 4)
    assert observed.dtype == torch.float32
    assert observed.device.type == "cpu"
    torch.testing.assert_close(observed, expected)


def test_selected_readout_rejects_invalid_inputs_and_nonfinite_values() -> None:
    backend = make_tiny_backend()
    backend.selected_readout(torch.ones(2, 4), (1,))
    with pytest.raises(ValidationError, match="at least one token"):
        backend.selected_readout(torch.ones(2, 4), ())
    with pytest.raises(ValidationError, match="outside the vocabulary"):
        backend.selected_readout(torch.ones(2, 4), (12,))
    with pytest.raises(ValidationError, match="only integers"):
        backend.selected_readout(torch.ones(2, 4), (True,))
    with pytest.raises(ValidationError, match="residual width"):
        backend.selected_readout(torch.ones(2, 3), (2,))
    bad = torch.ones(2, 4)
    bad[0, 0] = torch.nan
    with pytest.raises(RuntimeError, match="residual contains NaN or Inf"):
        backend.selected_readout(bad, (2,))


def test_selected_readout_caches_one_batched_parameter_gather(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = make_tiny_backend()
    original = backend_module.unembedding_parameters
    calls: list[tuple[int, ...]] = []

    def record(lm_head: Any, token_ids: tuple[int, ...]):
        calls.append(tuple(token_ids))
        return original(lm_head, token_ids)

    monkeypatch.setattr(backend_module, "unembedding_parameters", record)
    backend.selected_readout(torch.ones(2, 4), (7, 2, 7))
    backend.selected_readout(torch.zeros(3, 4), (7, 2, 7))

    assert calls == [(7, 2, 7)]


def test_accelerate_meta_offloaded_head_supports_logits_directions_and_readouts() -> (
    None
):
    backend = make_tiny_backend()
    original_weight = backend.lm_head.weight.detach().clone()
    backend.lm_head.bias = nn.Parameter(
        torch.linspace(-0.1, 0.2, backend.lm_head.weight.shape[0])
    )
    residual = torch.linspace(-0.8, 1.1, 19 * 4).reshape(19, 4)
    token_ids = (8, 3, 8, 1)
    expected_selected = backend._unembed_readout(residual)[
        :, list(token_ids)
    ].float()
    add_hook_to_module(
        backend.lm_head,
        AlignDevicesHook(execution_device=torch.device("cpu"), offload=True),
    )
    assert backend.lm_head.weight.device.type == "meta"

    logits = backend._next_logits(torch.tensor([[1, 2, 3]]))
    assert logits.shape == (12,)
    assert torch.isfinite(logits).all()
    assert backend.lm_head.weight.device.type == "meta"

    direction = backend.directions.get("logit", 6, 0)
    torch.testing.assert_close(
        direction, original_weight[6] / original_weight[6].norm()
    )

    observed_selected = backend.selected_readout(residual, token_ids)
    torch.testing.assert_close(observed_selected, expected_selected)
    assert backend.lm_head.weight.device.type == "meta"
    assert backend.lm_head.bias.device.type == "meta"

    backend.final_norm = nn.Identity()
    rows = backend._top_readout(torch.ones(2, 4), top_k=2)
    assert len(rows) == 2
    assert all(len(row) == 2 for row in rows)
    assert backend.lm_head.weight.device.type == "meta"


def test_chat_request_uses_tokenizer_template_without_manual_flattening() -> None:
    backend = make_tiny_backend()
    request = InferenceRequest(
        prompt=None,
        messages=({"role": "user", "content": "alpha beta"},),
        layers=(0,),
        top_k=2,
        generation=GenerationSpec(max_new_tokens=1),
    )

    input_ids, formatted = backend._tokenize(request)

    assert input_ids.tolist() == [[1, 8, 2, 3, 9]]
    assert formatted == "<BOS><user>alpha beta<assistant>"
    assert backend.tokenizer.chat_calls == [
        {
            "messages": [{"role": "user", "content": "alpha beta"}],
            "tokenize": True,
            "add_generation_prompt": True,
            "enable_thinking": False,
            "return_tensors": "pt",
        }
    ]


def test_final_block_readout_is_logit_only(monkeypatch: pytest.MonkeyPatch) -> None:
    backend = make_tiny_backend()
    monkeypatch.setattr(backend_module, "SOURCE_LAYERS", (0, 1))
    request = make_request(max_new_tokens=1, layers=(2,))
    sequence = torch.tensor([[1, 2, 3]])

    readouts = backend._teacher_forced_readouts(
        sequence,
        request,
        None,
        cancel_event=None,
        progress=None,
        condition="clean",
        prompt_length=2,
    )

    assert readouts["layers"] == [2]
    assert set(readouts["logit"]) == {"2"}
    assert readouts["jacobian"] == {}


def test_service_rejects_a_concurrent_request_as_busy() -> None:
    backend = make_tiny_backend(block_on_call=1)
    service = InferenceService(backend)
    request = make_request(max_new_tokens=1)

    with ThreadPoolExecutor(max_workers=2) as pool:
        active = pool.submit(service.run, request, paired=False, request_id="active_01")
        assert backend.backbone.block_entered.wait(timeout=5)
        assert service.health()["busy"] is True
        with pytest.raises(InferenceBusy, match="another inference request"):
            service.run(request, paired=False, request_id="waiting_02")
        busy = service.status("waiting_02")
        assert busy is not None
        assert busy["status"] == "busy"
        assert busy["active_request_id"] == "active_01"
        backend.backbone.block_release.set()
        result = active.result(timeout=5)

    assert result["request_id"] == "active_01"
    assert service.health()["busy"] is False
    assert all(not layer._forward_hooks for layer in backend.layers)


def test_service_cancellation_during_intervention_cleans_hooks_and_state() -> None:
    # One-token paired runs use calls 1-2 for clean generation/readout. Call 3
    # blocks inside an active intervention HookSession.
    backend = make_tiny_backend(block_on_call=3)
    service = InferenceService(backend)
    request = make_request(intervention=make_spec(), max_new_tokens=1)

    with ThreadPoolExecutor(max_workers=1) as pool:
        active = pool.submit(service.run, request, paired=True, request_id="cancel_01")
        assert backend.backbone.block_entered.wait(timeout=5)
        assert service.cancel("cancel_01") is True
        backend.backbone.block_release.set()
        with pytest.raises(InferenceCancelled, match="request cancelled"):
            active.result(timeout=5)

    status = service.status("cancel_01")
    assert status is not None
    assert status["status"] == "cancelled"
    assert service.health()["busy"] is False
    assert all(not layer._forward_hooks for layer in backend.layers)
