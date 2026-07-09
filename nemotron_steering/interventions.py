"""Neuronpedia-compatible activation intervention math and hook lifecycle.

The behavior follows Neuronpedia commit
fba06912787a1cd92fa68db2b708a7a3d1c4a5c7. See THIRD_PARTY_NOTICES.md.
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from collections.abc import Callable, Mapping, Sequence
from contextlib import AbstractContextManager
from dataclasses import dataclass, field
from typing import Any

import torch

from nemotron_steering.constants import DIRECTION_CACHE_SIZE, LENS_SHA256
from nemotron_steering.errors import InferenceCancelled, ValidationError
from nemotron_steering.validation import InterventionSpec


def unit_normalize(direction: torch.Tensor) -> torch.Tensor:
    """Normalize one direction, leaving an exact zero direction unchanged."""
    direction = direction.float()
    norm = torch.linalg.vector_norm(direction)
    if not bool(torch.isfinite(norm)):
        raise ValidationError("intervention direction contains NaN or Inf")
    if float(norm.item()) == 0.0:
        return torch.zeros_like(direction)
    return direction / norm


def jacobian_direction(
    unembedding: torch.Tensor, jacobian: torch.Tensor
) -> torch.Tensor:
    """Return normalize(w_t @ J_l), with J mapping source to final basis."""
    if unembedding.ndim != 1 or jacobian.ndim != 2:
        raise ValidationError("direction inputs must be a vector and a matrix")
    if jacobian.shape != (unembedding.numel(), unembedding.numel()):
        raise ValidationError("Jacobian shape does not match the unembedding width")
    return unit_normalize(unembedding.float().cpu() @ jacobian.float().cpu())


def logit_direction(unembedding: torch.Tensor) -> torch.Tensor:
    return unit_normalize(unembedding.float().cpu())


def position_mask(
    input_ids: torch.Tensor,
    *,
    original_prompt_length: int,
    bos_token_id: int | None,
    apply_to_generated: bool,
) -> torch.Tensor:
    """Select all eligible non-BOS positions for the current full prefix."""
    if input_ids.ndim != 2:
        raise ValidationError("input_ids must have shape [batch, sequence]")
    if not 0 <= original_prompt_length <= input_ids.shape[1]:
        raise ValidationError("original_prompt_length is outside the current prefix")
    positions = torch.arange(input_ids.shape[1], device=input_ids.device)
    mask = torch.ones_like(input_ids, dtype=torch.bool)
    if not apply_to_generated:
        mask &= positions.unsqueeze(0) < original_prompt_length
    if bos_token_id is not None:
        mask &= input_ids != bos_token_id
    return mask


def _masked_replace(
    hidden: torch.Tensor, changed: torch.Tensor, mask: torch.Tensor | None
) -> torch.Tensor:
    if mask is None:
        return changed
    if mask.shape != hidden.shape[:-1]:
        raise ValidationError(
            f"position mask shape {tuple(mask.shape)} does not match hidden states "
            f"{tuple(hidden.shape[:-1])}"
        )
    return torch.where(mask.to(hidden.device).unsqueeze(-1), changed, hidden)


def steer(
    hidden: torch.Tensor,
    direction: torch.Tensor,
    strength: float,
    *,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Apply h' = h + strength * ||h||_2 * direction per position."""
    if strength == 0.0 or float(torch.linalg.vector_norm(direction).item()) == 0.0:
        return hidden
    work = hidden.float()
    delta = direction.to(device=hidden.device, dtype=torch.float32)
    changed = (
        work + strength * torch.linalg.vector_norm(work, dim=-1, keepdim=True) * delta
    )
    return _masked_replace(hidden, changed.to(hidden.dtype), mask)


def ablate(
    hidden: torch.Tensor,
    direction: torch.Tensor,
    *,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Remove the component of h along the unit direction."""
    normalized = unit_normalize(direction).to(hidden.device)
    if float(torch.linalg.vector_norm(normalized).item()) == 0.0:
        return hidden
    work = hidden.float()
    projection = (work * normalized).sum(dim=-1, keepdim=True)
    changed = work - projection * normalized
    return _masked_replace(hidden, changed.to(hidden.dtype), mask)


def swap(
    hidden: torch.Tensor,
    source_direction: torch.Tensor,
    target_direction: torch.Tensor,
    *,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Replace the signed source projection with the same target coefficient."""
    source = unit_normalize(source_direction).to(hidden.device)
    target = unit_normalize(target_direction).to(hidden.device)
    if (
        float(torch.linalg.vector_norm(source).item()) == 0.0
        or float(torch.linalg.vector_norm(target).item()) == 0.0
    ):
        return hidden
    work = hidden.float()
    coefficient = (work * source).sum(dim=-1, keepdim=True)
    changed = work - coefficient * source + coefficient * target
    return _masked_replace(hidden, changed.to(hidden.dtype), mask)


def apply_intervention(
    hidden: torch.Tensor,
    spec: InterventionSpec,
    source_direction: torch.Tensor,
    *,
    target_direction: torch.Tensor | None = None,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Apply swap first, then ablation, then additive steering."""
    if spec.mode == "swap":
        if target_direction is None:
            raise ValidationError("swap requires a target direction")
        return swap(hidden, source_direction, target_direction, mask=mask)
    if spec.mode == "ablate":
        return ablate(hidden, source_direction, mask=mask)
    return steer(hidden, source_direction, spec.strength, mask=mask)


def _accelerate_parameter(module: Any, name: str) -> torch.Tensor | None:
    """Resolve one parameter without materializing an offloaded module."""
    parameter = getattr(module, name, None)
    if parameter is None:
        return None
    if not isinstance(parameter, torch.Tensor):
        raise RuntimeError(f"{type(module).__name__}.{name} is not a tensor")
    if parameter.device.type != "meta":
        return parameter

    pending = [getattr(module, "_hf_hook", None)]
    visited: set[int] = set()
    while pending:
        hook = pending.pop()
        if hook is None or id(hook) in visited:
            continue
        visited.add(id(hook))
        pending.extend(getattr(hook, "hooks", ()))
        weights_map = getattr(hook, "weights_map", None)
        if weights_map is None:
            continue
        try:
            stored = weights_map[name]
        except (KeyError, TypeError):
            continue
        if isinstance(stored, torch.Tensor):
            if tuple(stored.shape) != tuple(parameter.shape):
                raise RuntimeError(
                    f"offloaded {type(module).__name__}.{name} shape "
                    f"{tuple(stored.shape)} differs from {tuple(parameter.shape)}"
                )
            return stored
    raise RuntimeError(
        f"{type(module).__name__}.{name} is on the meta device but its "
        "Accelerate offload map is unavailable"
    )


def unembedding_parameters(
    lm_head: Any,
    token_ids: Sequence[int],
    *,
    include_bias: bool = True,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Return selected LM-head rows and bias as detached CPU FP32 tensors.

    Resident and Accelerate-offloaded heads share this path, so callers never
    need to materialize the full vocabulary projection merely to read a small
    fixed token set. Duplicate IDs are retained in the requested order.
    """
    ids = tuple(token_ids)
    if not ids:
        raise ValidationError("token_ids must contain at least one token ID")
    if any(
        isinstance(token_id, bool) or not isinstance(token_id, int)
        for token_id in ids
    ):
        raise ValidationError("token_ids must contain only integers")

    weight = _accelerate_parameter(lm_head, "weight")
    if weight is None or weight.ndim != 2:
        raise RuntimeError("lm_head.weight must be a rank-2 tensor")
    vocab_size = weight.shape[0]
    invalid = sorted(
        {token_id for token_id in ids if not 0 <= token_id < vocab_size}
    )
    if invalid:
        raise ValidationError(
            f"token IDs {invalid} are outside the vocabulary range [0, {vocab_size - 1}]"
        )

    index = torch.tensor(ids, device=weight.device, dtype=torch.long)
    rows = weight.index_select(0, index).detach().to(device="cpu", dtype=torch.float32)
    if not bool(torch.isfinite(rows).all()):
        raise RuntimeError("selected lm_head weights contain NaN or Inf")

    bias: torch.Tensor | None = None
    if include_bias:
        stored_bias = _accelerate_parameter(lm_head, "bias")
        if stored_bias is not None:
            if stored_bias.ndim != 1 or stored_bias.shape[0] != vocab_size:
                raise RuntimeError("lm_head.bias does not match lm_head.weight")
            bias = (
                stored_bias.index_select(
                    0, torch.tensor(ids, device=stored_bias.device, dtype=torch.long)
                )
                .detach()
                .to(device="cpu", dtype=torch.float32)
            )
            if not bool(torch.isfinite(bias).all()):
                raise RuntimeError("selected lm_head bias contains NaN or Inf")
    return rows, bias


class DirectionCache:
    """Bounded cache of requested FP32 direction vectors, never full GPU J matrices."""

    def __init__(
        self,
        *,
        lm_head: Any,
        jacobians: Mapping[int, torch.Tensor],
        lens_sha256: str = LENS_SHA256,
        max_entries: int = DIRECTION_CACHE_SIZE,
    ) -> None:
        if max_entries <= 0:
            raise ValueError("max_entries must be positive")
        self._lm_head = lm_head
        self._jacobians = jacobians
        self._lens_sha256 = lens_sha256
        self._max_entries = max_entries
        self._values: OrderedDict[tuple[str, str, int, int], torch.Tensor] = (
            OrderedDict()
        )
        self._lock = threading.Lock()

    def get(self, lens_type: str, token_id: int, layer: int) -> torch.Tensor:
        key = (self._lens_sha256, lens_type, token_id, layer)
        with self._lock:
            cached = self._values.get(key)
            if cached is not None:
                self._values.move_to_end(key)
                return cached
        weight = unembedding_parameters(
            self._lm_head, (token_id,), include_bias=False
        )[0][0]
        if lens_type == "jacobian":
            try:
                jacobian = self._jacobians[layer]
            except KeyError as exc:
                raise ValidationError(
                    f"layer {layer} has no fitted Jacobian direction"
                ) from exc
            value = jacobian_direction(weight, jacobian)
        elif lens_type == "logit":
            value = logit_direction(weight)
        else:
            raise ValidationError(f"unknown lens type {lens_type!r}")
        with self._lock:
            self._values[key] = value
            self._values.move_to_end(key)
            while len(self._values) > self._max_entries:
                self._values.popitem(last=False)
        return value

    def combined(
        self, spec: InterventionSpec, layer: int
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        sources = [
            self.get(spec.lens_type, token_id, layer)
            for token_id in spec.source_token_ids
        ]
        source = torch.stack(sources).sum(dim=0)
        target = (
            self.get(spec.lens_type, spec.target_token_id, layer)
            if spec.target_token_id is not None
            else None
        )
        return source, target

    def __len__(self) -> int:
        with self._lock:
            return len(self._values)


@dataclass
class ForwardContext:
    """Mutable per-condition state read by hooks during each full-prefix forward."""

    original_prompt_length: int
    bos_token_id: int | None
    apply_to_generated: bool
    input_ids: torch.Tensor | None = None

    def set_input_ids(self, input_ids: torch.Tensor) -> None:
        self.input_ids = input_ids

    def mask_for(self, hidden: torch.Tensor) -> torch.Tensor:
        if self.input_ids is None:
            raise RuntimeError("hook context has no current input_ids")
        return position_mask(
            self.input_ids.to(hidden.device),
            original_prompt_length=self.original_prompt_length,
            bos_token_id=self.bos_token_id,
            apply_to_generated=self.apply_to_generated,
        )


def _hidden_from_output(output: Any) -> torch.Tensor:
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, tuple) and output and isinstance(output[0], torch.Tensor):
        return output[0]
    raise TypeError("residual block output must be a tensor or tensor-first tuple")


def _replace_hidden(output: Any, hidden: torch.Tensor) -> Any:
    if isinstance(output, torch.Tensor):
        return hidden
    if isinstance(output, tuple):
        if hasattr(output, "_fields") and hasattr(output, "_replace"):
            return output._replace(**{output._fields[0]: hidden})
        return (hidden, *output[1:])
    raise TypeError("residual block output must be a tensor or tensor-first tuple")


@dataclass
class HookSession(AbstractContextManager["HookSession"]):
    """Register ordered intervention/capture hooks and always remove them."""

    layers: Sequence[Any]
    context: ForwardContext
    intervention: InterventionSpec | None = None
    directions: Mapping[int, tuple[torch.Tensor, torch.Tensor | None]] = field(
        default_factory=dict
    )
    capture_layers: Sequence[int] = field(default_factory=tuple)
    cancel_event: threading.Event | None = None
    captures: dict[int, torch.Tensor] = field(default_factory=dict, init=False)
    _handles: list[Any] = field(default_factory=list, init=False)

    def __enter__(self) -> HookSession:
        try:
            if self.intervention is not None:
                for layer in self.intervention.layers:
                    self._handles.append(
                        self.layers[layer].register_forward_hook(
                            self._intervention_hook(layer)
                        )
                    )
            for layer in self.capture_layers:
                self._handles.append(
                    self.layers[layer].register_forward_hook(self._capture_hook(layer))
                )
        except BaseException:
            self.close()
            raise
        return self

    def _intervention_hook(self, layer: int) -> Callable[..., Any]:
        def hook(_module: Any, _inputs: Any, output: Any) -> Any:
            if self.cancel_event is not None and self.cancel_event.is_set():
                raise InferenceCancelled("request cancelled")
            hidden = _hidden_from_output(output)
            source, target = self.directions[layer]
            changed = apply_intervention(
                hidden,
                self.intervention,
                source,
                target_direction=target,
                mask=self.context.mask_for(hidden),
            )
            return _replace_hidden(output, changed)

        return hook

    def _capture_hook(self, layer: int) -> Callable[..., None]:
        def hook(_module: Any, _inputs: Any, output: Any) -> None:
            hidden = _hidden_from_output(output)
            self.captures[layer] = hidden.detach().to(device="cpu", dtype=torch.float32)

        return hook

    def close(self) -> None:
        while self._handles:
            self._handles.pop().remove()

    def __exit__(self, *_exc: object) -> None:
        self.close()
