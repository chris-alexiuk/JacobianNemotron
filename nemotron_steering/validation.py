"""Strict validation for token identities and intervention parameters."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from nemotron_steering.constants import (
    DEFAULT_STRENGTH,
    INTERVENTION_MODES,
    LENS_TYPES,
    MAX_INTERVENTION_LAYERS,
    MAX_READOUT_LAYERS,
    MAX_SOURCE_TOKENS,
    MAX_STRENGTH,
    MIN_STRENGTH,
    N_LAYERS,
    SOURCE_LAYERS,
    VOCAB_SIZE,
)
from nemotron_steering.errors import ValidationError


@dataclass(frozen=True)
class InterventionSpec:
    """A validated intervention independent of HTTP representation."""

    mode: str
    lens_type: str
    layers: tuple[int, ...]
    source_token_ids: tuple[int, ...]
    target_token_id: int | None = None
    strength: float = DEFAULT_STRENGTH
    apply_to_generated: bool = False


def _strict_int(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValidationError(f"{field} must be an integer")
    return value


def validate_token_id(value: object, *, field: str) -> int:
    token_id = _strict_int(value, field=field)
    if not 0 <= token_id < VOCAB_SIZE:
        raise ValidationError(f"{field} must be in [0, {VOCAB_SIZE - 1}]")
    return token_id


def validate_layers(value: object, *, field: str = "layers") -> tuple[int, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValidationError(f"{field} must be an array of source-layer integers")
    if not value:
        raise ValidationError(f"{field} must not be empty")
    if len(value) > MAX_INTERVENTION_LAYERS:
        raise ValidationError(
            f"{field} may contain at most {MAX_INTERVENTION_LAYERS} layers"
        )
    layers = tuple(_strict_int(layer, field=f"{field}[]") for layer in value)
    if len(set(layers)) != len(layers):
        raise ValidationError(f"{field} must not contain duplicate layers")
    invalid = sorted(set(layers) - set(SOURCE_LAYERS))
    if invalid:
        raise ValidationError(
            f"intervention layers {invalid} are invalid; fitted source layers are 0-50 "
            "and block 51 is read-only"
        )
    return tuple(sorted(layers))


def validate_readout_layers(value: object) -> tuple[int, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValidationError("layers must be an array of layer integers")
    if not value or len(value) > MAX_READOUT_LAYERS:
        raise ValidationError(
            f"layers must contain 1-{MAX_READOUT_LAYERS} layer integers"
        )
    layers = tuple(_strict_int(layer, field="layers[]") for layer in value)
    if len(set(layers)) != len(layers):
        raise ValidationError("layers must not contain duplicates")
    invalid = sorted(layer for layer in layers if not 0 <= layer < N_LAYERS)
    if invalid:
        raise ValidationError(
            f"readout layers {invalid} are outside model layers 0-{N_LAYERS - 1}"
        )
    return tuple(sorted(layers))


def validate_strength(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValidationError("strength must be a finite number")
    strength = float(value)
    if not math.isfinite(strength) or not MIN_STRENGTH <= strength <= MAX_STRENGTH:
        raise ValidationError(
            f"strength must be finite and in [{MIN_STRENGTH:g}, {MAX_STRENGTH:g}]"
        )
    return strength


def validate_intervention(value: Mapping[str, Any]) -> InterventionSpec:
    if not isinstance(value, Mapping):
        raise ValidationError("intervention must be an object")
    mode = value.get("mode")
    if mode not in INTERVENTION_MODES:
        raise ValidationError(f"mode must be one of {sorted(INTERVENTION_MODES)}")
    lens_type = value.get("lens_type")
    if lens_type not in LENS_TYPES:
        raise ValidationError(f"lens_type must be one of {sorted(LENS_TYPES)}")
    layers = validate_layers(value.get("layers"))

    raw_sources = value.get("source_token_ids")
    if isinstance(raw_sources, (str, bytes)) or not isinstance(raw_sources, Sequence):
        raise ValidationError("source_token_ids must be an array of numeric token IDs")
    if not raw_sources or len(raw_sources) > MAX_SOURCE_TOKENS:
        raise ValidationError(
            f"source_token_ids must contain 1-{MAX_SOURCE_TOKENS} IDs"
        )
    source_ids = tuple(
        validate_token_id(token_id, field="source_token_ids[]")
        for token_id in raw_sources
    )
    if len(set(source_ids)) != len(source_ids):
        raise ValidationError("source_token_ids must not contain duplicates")

    target_id = value.get("target_token_id")
    if mode == "swap":
        if len(source_ids) != 1:
            raise ValidationError("swap requires exactly one source token ID")
        target_id = validate_token_id(target_id, field="target_token_id")
    elif target_id is not None:
        raise ValidationError("target_token_id is accepted only for swap")

    strength = validate_strength(value.get("strength", DEFAULT_STRENGTH))
    apply_to_generated = value.get("apply_to_generated", False)
    if not isinstance(apply_to_generated, bool):
        raise ValidationError("apply_to_generated must be boolean")
    return InterventionSpec(
        mode=mode,
        lens_type=lens_type,
        layers=layers,
        source_token_ids=source_ids,
        target_token_id=target_id,
        strength=strength,
        apply_to_generated=apply_to_generated,
    )


def token_pieces(tokenizer: Any, text: str) -> dict[str, Any]:
    if not isinstance(text, str) or not text:
        raise ValidationError("token text must be a non-empty string")
    raw_ids = tokenizer.encode(text, add_special_tokens=False)
    if hasattr(raw_ids, "tolist"):
        raw_ids = raw_ids.tolist()
    if raw_ids and isinstance(raw_ids[0], list):
        raw_ids = raw_ids[0]
    ids = [int(token_id) for token_id in raw_ids]
    pieces = []
    for token_id in ids:
        try:
            decoded = tokenizer.decode(
                [token_id],
                clean_up_tokenization_spaces=False,
                skip_special_tokens=False,
            )
        except TypeError:
            decoded = tokenizer.decode([token_id])
        pieces.append({"id": token_id, "text": decoded})
    return {
        "text": text,
        "token_ids": ids,
        "pieces": pieces,
        "is_single_token": len(ids) == 1,
    }


def validate_single_token_text(
    tokenizer: Any,
    text: str,
    *,
    field: str,
    expected_id: int | None = None,
) -> int:
    result = token_pieces(tokenizer, text)
    if not result["is_single_token"]:
        raise ValidationError(
            f"{field} must tokenize to exactly one token with add_special_tokens=False",
            details=result,
        )
    token_id = validate_token_id(result["token_ids"][0], field=field)
    if expected_id is not None and token_id != expected_id:
        raise ValidationError(
            f"{field} does not match the authoritative numeric token ID",
            details={**result, "expected_id": expected_id},
        )
    return token_id
