"""Bounded API request parsing with no model or path inputs."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from nemotron_steering.constants import (
    DEFAULT_MAX_NEW_TOKENS,
    DEFAULT_READOUT_LAYERS,
    DEFAULT_TOP_K,
    MAX_NEW_TOKENS,
    MAX_TOP_K,
)
from nemotron_steering.errors import ValidationError
from nemotron_steering.validation import (
    InterventionSpec,
    validate_intervention,
    validate_readout_layers,
    validate_single_token_text,
)


@dataclass(frozen=True)
class GenerationSpec:
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS
    sampling: bool = False
    temperature: float = 1.0
    top_p: float = 1.0
    seed: int = 0


@dataclass(frozen=True)
class InferenceRequest:
    prompt: str | None
    messages: tuple[dict[str, str], ...] | None
    layers: tuple[int, ...]
    top_k: int
    generation: GenerationSpec
    enable_thinking: bool = False
    intervention: InterventionSpec | None = None


_COMMON_FIELDS = {
    "prompt",
    "messages",
    "layers",
    "top_k",
    "max_new_tokens",
    "sampling",
    "temperature",
    "top_p",
    "seed",
    "enable_thinking",
}
_INTERVENTION_FIELDS = {
    "intervention",
    "source_token_texts",
    "target_token_text",
}


def _integer(value: object, *, field: str, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValidationError(f"{field} must be an integer")
    if not minimum <= value <= maximum:
        raise ValidationError(f"{field} must be in [{minimum}, {maximum}]")
    return value


def _finite_number(
    value: object, *, field: str, minimum: float, maximum: float
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValidationError(f"{field} must be a finite number")
    number = float(value)
    if not math.isfinite(number) or not minimum <= number <= maximum:
        raise ValidationError(f"{field} must be in [{minimum:g}, {maximum:g}]")
    return number


def _messages(value: object) -> tuple[dict[str, str], ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValidationError("messages must be an array")
    if not 1 <= len(value) <= 32:
        raise ValidationError("messages must contain 1-32 entries")
    parsed: list[dict[str, str]] = []
    for index, message in enumerate(value):
        if not isinstance(message, Mapping):
            raise ValidationError(f"messages[{index}] must be an object")
        if set(message) != {"role", "content"}:
            raise ValidationError(
                f"messages[{index}] must contain exactly role and content"
            )
        role = message.get("role")
        content = message.get("content")
        if role not in {"system", "user", "assistant"}:
            raise ValidationError(
                f"messages[{index}].role must be system, user, or assistant"
            )
        if not isinstance(content, str) or not content:
            raise ValidationError(f"messages[{index}].content must be non-empty text")
        parsed.append({"role": role, "content": content})

    offset = 1 if parsed[0]["role"] == "system" else 0
    conversation = parsed[offset:]
    if not conversation:
        raise ValidationError(
            "messages must include a user message after the system message"
        )
    for index, message in enumerate(conversation):
        expected = "user" if index % 2 == 0 else "assistant"
        if message["role"] != expected:
            actual_index = index + offset
            raise ValidationError(
                f"messages[{actual_index}].role must be {expected} for an alternating chat"
            )
    if conversation[-1]["role"] != "user":
        raise ValidationError(
            "messages must end with user before generating an assistant reply"
        )
    return tuple(parsed)


def parse_request(
    body: object,
    *,
    require_intervention: bool,
    tokenizer: Any | None = None,
) -> InferenceRequest:
    if not isinstance(body, Mapping):
        raise ValidationError("request body must be a JSON object")
    allowed = _COMMON_FIELDS | (_INTERVENTION_FIELDS if require_intervention else set())
    unknown = sorted(set(body) - allowed)
    if unknown:
        raise ValidationError(f"unknown request fields: {unknown}")

    has_prompt = body.get("prompt") is not None
    has_messages = body.get("messages") is not None
    if has_prompt == has_messages:
        raise ValidationError("provide exactly one of prompt or messages")
    prompt: str | None = None
    messages: tuple[dict[str, str], ...] | None = None
    if has_prompt:
        prompt = body["prompt"]
        if not isinstance(prompt, str) or not prompt:
            raise ValidationError("prompt must be non-empty text")
    else:
        messages = _messages(body["messages"])

    if has_prompt and "enable_thinking" in body:
        raise ValidationError("enable_thinking is accepted only with chat messages")
    enable_thinking = body.get("enable_thinking", False)
    if not isinstance(enable_thinking, bool):
        raise ValidationError("enable_thinking must be boolean")

    layers = validate_readout_layers(body.get("layers", list(DEFAULT_READOUT_LAYERS)))
    top_k = _integer(
        body.get("top_k", DEFAULT_TOP_K),
        field="top_k",
        minimum=1,
        maximum=MAX_TOP_K,
    )
    max_new_tokens = _integer(
        body.get("max_new_tokens", DEFAULT_MAX_NEW_TOKENS),
        field="max_new_tokens",
        minimum=1,
        maximum=MAX_NEW_TOKENS,
    )
    sampling = body.get("sampling", False)
    if not isinstance(sampling, bool):
        raise ValidationError("sampling must be boolean")
    temperature = _finite_number(
        body.get("temperature", 1.0),
        field="temperature",
        minimum=0.01,
        maximum=10.0,
    )
    top_p = _finite_number(
        body.get("top_p", 1.0), field="top_p", minimum=0.01, maximum=1.0
    )
    seed = _integer(body.get("seed", 0), field="seed", minimum=0, maximum=2**63 - 1)
    generation = GenerationSpec(
        max_new_tokens=max_new_tokens,
        sampling=sampling,
        temperature=temperature,
        top_p=top_p,
        seed=seed,
    )

    intervention: InterventionSpec | None = None
    if require_intervention:
        intervention = validate_intervention(body.get("intervention"))
        if tokenizer is not None:
            source_texts = body.get("source_token_texts")
            if source_texts is not None:
                if (
                    isinstance(source_texts, (str, bytes))
                    or not isinstance(source_texts, Sequence)
                    or len(source_texts) != len(intervention.source_token_ids)
                ):
                    raise ValidationError(
                        "source_token_texts must align with source_token_ids"
                    )
                for text, token_id in zip(
                    source_texts, intervention.source_token_ids, strict=True
                ):
                    validate_single_token_text(
                        tokenizer,
                        text,
                        field="source_token_texts[]",
                        expected_id=token_id,
                    )
            target_text = body.get("target_token_text")
            if target_text is not None:
                if intervention.target_token_id is None:
                    raise ValidationError("target_token_text is accepted only for swap")
                validate_single_token_text(
                    tokenizer,
                    target_text,
                    field="target_token_text",
                    expected_id=intervention.target_token_id,
                )
    return InferenceRequest(
        prompt=prompt,
        messages=messages,
        layers=layers,
        top_k=top_k,
        generation=generation,
        enable_thinking=enable_thinking,
        intervention=intervention,
    )
