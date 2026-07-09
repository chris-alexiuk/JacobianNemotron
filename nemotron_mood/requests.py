# Portions adapted from jlens-mood by Eric W. Tramel.
# Copyright (c) 2026 Eric W. Tramel
# SPDX-License-Identifier: MIT
"""Strict, service-independent validation for Nano mood requests."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

DEFAULT_MOOD_LAYERS = tuple(range(13, 51))
# The accepted lens was fitted on 128-token samples. Keep each independent
# mood prefill inside that fitted sequence-length regime.
DEFAULT_CHUNK_TOKENS = 128
MAX_CHUNK_TOKENS = 128
SOURCE_LAYERS = frozenset(range(51))
MAX_TEXT_BYTES = 64 * 1024


class MoodRequestError(ValueError):
    """A request error safe for an HTTP 422 response."""


@dataclass(frozen=True)
class MoodRequest:
    text: str
    layers: tuple[int, ...]
    chunk_tokens: int

    def to_payload(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "layers": list(self.layers),
            "chunk_tokens": self.chunk_tokens,
        }


def _layers(value: object) -> tuple[int, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise MoodRequestError("layers must be an array of layer integers")
    if not value or len(value) > len(SOURCE_LAYERS):
        raise MoodRequestError("layers must contain 1-51 layer integers")
    parsed: list[int] = []
    for index, layer in enumerate(value):
        if isinstance(layer, bool) or not isinstance(layer, int):
            raise MoodRequestError(f"layers[{index}] must be an integer")
        parsed.append(layer)
    if len(set(parsed)) != len(parsed):
        raise MoodRequestError("layers must not contain duplicates")
    invalid = sorted(set(parsed) - SOURCE_LAYERS)
    if invalid:
        raise MoodRequestError(
            f"mood layers {invalid} are invalid; fitted source layers are 0-50"
        )
    return tuple(sorted(parsed))


def parse_mood_request(body: object) -> MoodRequest:
    if not isinstance(body, Mapping):
        raise MoodRequestError("request body must be a JSON object")
    unknown = sorted(set(body) - {"text", "layers", "chunk_tokens"})
    if unknown:
        raise MoodRequestError(f"unknown request fields: {unknown}")

    text = body.get("text")
    if not isinstance(text, str) or not text.strip():
        raise MoodRequestError("text must be non-empty text")
    if len(text.encode("utf-8")) > MAX_TEXT_BYTES:
        raise MoodRequestError(
            f"text must be at most {MAX_TEXT_BYTES} UTF-8 bytes"
        )

    layers = _layers(body.get("layers", DEFAULT_MOOD_LAYERS))
    chunk_tokens = body.get("chunk_tokens", DEFAULT_CHUNK_TOKENS)
    if isinstance(chunk_tokens, bool) or not isinstance(chunk_tokens, int):
        raise MoodRequestError("chunk_tokens must be an integer")
    if not 1 <= chunk_tokens <= MAX_CHUNK_TOKENS:
        raise MoodRequestError(
            f"chunk_tokens must be in [1, {MAX_CHUNK_TOKENS}]"
        )
    return MoodRequest(text=text, layers=layers, chunk_tokens=chunk_tokens)
