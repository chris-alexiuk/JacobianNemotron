# Portions adapted from jlens-mood by Eric W. Tramel.
# Copyright (c) 2026 Eric W. Tramel
# SPDX-License-Identifier: MIT
"""Pure emotion-axis calibration, aggregation, and serialization helpers."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch

from nemotron_mood.anchors import EMOTIONS

SENTENCE_END = (".", "!", "?", '."', '!"', '?"', ".)", ".'")

EMOTION_COLORS = {
    "sadness": "#2a78d6",
    "surprise": "#1baf7a",
    "joy": "#eda100",
    "disgust": "#008300",
    "fear": "#4a3aa7",
    "anger": "#e34948",
    "curiosity": "#e87ba4",
    "neutral": "#8a8984",
}

NEUTRAL_REFERENCE_TEXTS = (
    "The report was printed on standard paper and filed in the second drawer.",
    "The train departs from platform four at nine fifteen each morning.",
    "The store closes at nine on weekdays and at six on Sundays.",
    "He walked to the corner shop, bought a loaf of bread, and came back "
    "before the kettle finished boiling.",
    "She parked the car in the usual spot and took the elevator to the third floor.",
)


def sentence_spans(tokens: list[str]) -> list[tuple[int, int]]:
    """Return half-open token spans, dropping trailing whitespace."""
    spans: list[tuple[int, int]] = []
    start = 0
    for index, token in enumerate(tokens):
        if token.rstrip().endswith(SENTENCE_END) and index - start >= 2:
            spans.append((start, index + 1))
            start = index + 1
    if start < len(tokens) and "".join(tokens[start:]).strip():
        spans.append((start, len(tokens)))
    return spans


@dataclass
class EmotionTrace:
    """Per-layer emotion scores aligned to exact tokenizer pieces."""

    tokens: list[str]
    token_ids: list[int]
    per_layer: dict[int, torch.Tensor]
    dropped: dict[str, list[str]]

    def __post_init__(self) -> None:
        if not self.tokens:
            raise ValueError("emotion trace must contain at least one token")
        if len(self.token_ids) != len(self.tokens):
            raise ValueError("token_ids must align one-to-one with tokens")
        if not self.per_layer:
            raise ValueError("emotion trace must contain at least one layer")
        expected = (len(self.tokens), len(EMOTIONS))
        for layer, scores in self.per_layer.items():
            if isinstance(layer, bool) or not isinstance(layer, int):
                raise ValueError("emotion trace layer keys must be integers")
            if not isinstance(scores, torch.Tensor) or tuple(scores.shape) != expected:
                raise ValueError(
                    f"layer {layer} scores have shape "
                    f"{tuple(getattr(scores, 'shape', ()))}, expected {expected}"
                )
            if not bool(torch.isfinite(scores).all()):
                raise ValueError(f"layer {layer} scores contain NaN or Inf")

    @property
    def mean(self) -> torch.Tensor:
        """Mean axis scores over layers, shaped ``[positions, emotions]``."""
        return torch.stack(
            [self.per_layer[layer] for layer in sorted(self.per_layer)]
        ).mean(dim=0)


@dataclass(frozen=True)
class SentenceMood:
    text: str
    dominant: str
    strongest: str
    intensity: float
    shares: dict[str, float]
    start: int
    end: int

    @property
    def mood(self) -> str:
        return self.dominant

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "mood": self.dominant,
            "strongest": self.strongest,
            "intensity": self.intensity,
            "shares": dict(self.shares),
            "start": self.start,
            "end": self.end,
        }


@dataclass(frozen=True)
class TokenMood:
    position: int
    id: int
    text: str
    dominant: str
    strongest: str
    intensity: float
    shares: dict[str, float]

    @property
    def mood(self) -> str:
        return self.dominant

    def to_dict(self) -> dict[str, Any]:
        return {
            "position": self.position,
            "id": self.id,
            "text": self.text,
            "mood": self.dominant,
            "strongest": self.strongest,
            "intensity": self.intensity,
            "shares": dict(self.shares),
        }


@dataclass(frozen=True)
class MoodSummary:
    dominant: str
    strongest: str
    intensity: float
    threshold: float
    shares: dict[str, float]
    sentences: list[SentenceMood]

    @property
    def mood(self) -> str:
        return self.dominant

    def to_dict(self) -> dict[str, Any]:
        return {
            "mood": self.dominant,
            "strongest": self.strongest,
            "intensity": self.intensity,
            "threshold": self.threshold,
            "shares": dict(self.shares),
            "sentences": [sentence.to_dict() for sentence in self.sentences],
        }


def calibrate(
    scores: torch.Tensor, baseline: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Subtract an axis baseline and softmax over the emotion dimension."""
    if not isinstance(scores, torch.Tensor) or scores.ndim < 1:
        raise ValueError("scores must be a tensor with an emotion dimension")
    if scores.shape[-1] != len(EMOTIONS):
        raise ValueError(
            f"scores must end with {len(EMOTIONS)} emotion values"
        )
    if not isinstance(baseline, torch.Tensor) or tuple(baseline.shape) != (
        len(EMOTIONS),
    ):
        raise ValueError(f"baseline must have shape ({len(EMOTIONS)},)")
    if not bool(torch.isfinite(scores).all()) or not bool(
        torch.isfinite(baseline).all()
    ):
        raise ValueError("scores and baseline must be finite")
    calibrated = scores - baseline.to(device=scores.device, dtype=scores.dtype)
    return calibrated, torch.softmax(calibrated, dim=-1)


def _score_summary(
    scores: torch.Tensor, baseline: torch.Tensor, threshold: float
) -> tuple[str, str, float, dict[str, float]]:
    calibrated, share_values = calibrate(scores.mean(dim=0), baseline)
    shares = {
        emotion: float(value)
        for emotion, value in zip(EMOTIONS, share_values.detach().cpu(), strict=True)
    }
    strongest = max(shares, key=shares.__getitem__)
    intensity = float(calibrated.max().detach().cpu())
    dominant = "neutral" if intensity < threshold else strongest
    return dominant, strongest, intensity, shares


def summarize_mood(
    trace: EmotionTrace,
    baseline: torch.Tensor,
    *,
    threshold: float,
) -> MoodSummary:
    """Uniformly aggregate positions and apply an explicit Nano threshold."""
    if isinstance(threshold, bool) or not isinstance(threshold, (int, float)):
        raise ValueError("threshold must be a finite number")
    threshold = float(threshold)
    if not math.isfinite(threshold):
        raise ValueError("threshold must be a finite number")

    scores = trace.mean
    dominant, strongest, intensity, shares = _score_summary(
        scores, baseline, threshold
    )
    sentences: list[SentenceMood] = []
    for start, end in sentence_spans(trace.tokens):
        sentence_mood, sentence_strongest, sentence_intensity, sentence_shares = (
            _score_summary(scores[start:end], baseline, threshold)
        )
        sentences.append(
            SentenceMood(
                text="".join(trace.tokens[start:end]).strip(),
                dominant=sentence_mood,
                strongest=sentence_strongest,
                intensity=sentence_intensity,
                shares=sentence_shares,
                start=start,
                end=end,
            )
        )
    return MoodSummary(
        dominant=dominant,
        strongest=strongest,
        intensity=intensity,
        threshold=threshold,
        shares=shares,
        sentences=sentences,
    )


def summarize_tokens(
    trace: EmotionTrace,
    baseline: torch.Tensor,
    *,
    threshold: float,
) -> list[TokenMood]:
    """Return calibrated, gated mood data for every tokenizer position."""
    if isinstance(threshold, bool) or not isinstance(threshold, (int, float)):
        raise ValueError("threshold must be a finite number")
    threshold = float(threshold)
    if not math.isfinite(threshold):
        raise ValueError("threshold must be a finite number")
    rows: list[TokenMood] = []
    for position, (token_id, text, scores) in enumerate(
        zip(trace.token_ids, trace.tokens, trace.mean, strict=True)
    ):
        dominant, strongest, intensity, shares = _score_summary(
            scores.unsqueeze(0), baseline, threshold
        )
        rows.append(
            TokenMood(
                position=position,
                id=token_id,
                text=text,
                dominant=dominant,
                strongest=strongest,
                intensity=intensity,
                shares=shares,
            )
        )
    return rows


def baseline_from_traces(traces: list[EmotionTrace]) -> torch.Tensor:
    """Uniform mean of per-text, per-position scores for neutral references."""
    if not traces:
        raise ValueError("at least one neutral trace is required")
    values = [trace.mean.mean(dim=0) for trace in traces]
    baseline = torch.stack(values).mean(dim=0)
    if not bool(torch.isfinite(baseline).all()):
        raise ValueError("neutral baseline contains NaN or Inf")
    return baseline
