# Portions adapted from jlens-mood by Eric W. Tramel.
# Copyright (c) 2026 Eric W. Tramel
# SPDX-License-Identifier: MIT
"""Emotion anchor words and tokenizer-specific cluster resolution."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol


class TokenizerLike(Protocol):
    def encode(self, text: str, *, add_special_tokens: bool = False) -> Any: ...


# These are the upstream semantic anchors. They are resolved against the
# pinned Nano tokenizer at runtime; no Qwen token IDs are carried over.
EMOTION_ANCHORS: dict[str, tuple[str, ...]] = {
    "sadness": (
        "sad",
        "sadness",
        "grief",
        "sorrow",
        "cry",
        "crying",
        "tears",
        "mourning",
        "lonely",
        "despair",
        "misery",
        "mourn",
        "ache",
        "anguish",
    ),
    "surprise": (
        "surprise",
        "surprised",
        "shock",
        "shocked",
        "astonished",
        "amazed",
        "sudden",
        "unexpected",
        "startled",
        "stunned",
        "disbelief",
        "marvel",
        "wow",
        "abrupt",
    ),
    "joy": (
        "joy",
        "happy",
        "happiness",
        "delight",
        "laugh",
        "laughter",
        "smile",
        "celebrate",
        "cheerful",
        "wonderful",
        "glad",
        "thrilled",
        "bliss",
    ),
    "disgust": (
        "disgust",
        "disgusting",
        "gross",
        "nausea",
        "filthy",
        "vile",
        "rotten",
        "foul",
        "nasty",
        "slime",
        "rot",
        "sewage",
        "mold",
    ),
    "fear": (
        "fear",
        "afraid",
        "terror",
        "panic",
        "dread",
        "scared",
        "anxious",
        "anxiety",
        "horror",
        "threat",
        "worried",
        "frightened",
    ),
    "anger": (
        "anger",
        "angry",
        "rage",
        "furious",
        "fury",
        "outrage",
        "hate",
        "hatred",
        "resentment",
        "hostile",
        "mad",
        "irritated",
    ),
    "curiosity": (
        "curious",
        "curiosity",
        "wonder",
        "wondering",
        "intrigued",
        "fascinated",
        "fascinating",
        "explore",
        "mystery",
        "puzzle",
        "inquiry",
        "interested",
        "discover",
        "investigate",
    ),
    "neutral": (
        "bland",
        "boring",
        "bored",
        "boredom",
        "dull",
        "mundane",
        "ordinary",
        "routine",
        "tedious",
        "meh",
    ),
}

EMOTIONS: tuple[str, ...] = tuple(EMOTION_ANCHORS)


def _flat_ids(value: Any) -> list[int]:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if value and isinstance(value[0], list):
        if len(value) != 1:
            raise ValueError("tokenizer returned more than one encoded sequence")
        value = value[0]
    return [int(token_id) for token_id in value]


def anchor_token_ids(
    tokenizer: TokenizerLike, words: Sequence[str]
) -> tuple[list[int], list[str]]:
    """Return unique single-token lower/title variants and dropped words."""
    ids: set[int] = set()
    dropped: list[str] = []
    for word in words:
        found = False
        for variant in (f" {word}", f" {word.capitalize()}"):
            encoded = _flat_ids(
                tokenizer.encode(variant, add_special_tokens=False)
            )
            if len(encoded) == 1:
                ids.add(encoded[0])
                found = True
        if not found:
            dropped.append(word)
    return sorted(ids), dropped


def resolve_anchor_clusters(
    tokenizer: TokenizerLike,
) -> tuple[dict[str, list[int]], dict[str, list[str]]]:
    """Resolve all axes and fail closed on empty or overlapping clusters."""
    clusters: dict[str, list[int]] = {}
    dropped: dict[str, list[str]] = {}
    owner_by_id: dict[int, str] = {}
    for emotion, words in EMOTION_ANCHORS.items():
        token_ids, missing = anchor_token_ids(tokenizer, words)
        if not token_ids:
            raise ValueError(
                f"no single-token anchors remain for {emotion!r} in this tokenizer"
            )
        for token_id in token_ids:
            previous = owner_by_id.get(token_id)
            if previous is not None and previous != emotion:
                raise ValueError(
                    f"anchor token ID {token_id} belongs to both {previous!r} "
                    f"and {emotion!r}"
                )
            owner_by_id[token_id] = emotion
        clusters[emotion] = token_ids
        if missing:
            dropped[emotion] = missing
    return clusters, dropped


def anchor_metadata(
    clusters: dict[str, list[int]], dropped: dict[str, list[str]]
) -> dict[str, dict[str, list[int] | list[str]]]:
    """Build the stable API representation for resolved anchors."""
    return {
        emotion: {
            "token_ids": list(clusters[emotion]),
            "dropped_words": list(dropped.get(emotion, ())),
        }
        for emotion in EMOTIONS
    }
