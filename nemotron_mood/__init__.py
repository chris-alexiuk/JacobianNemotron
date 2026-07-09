# Portions adapted from jlens-mood by Eric W. Tramel.
# Copyright (c) 2026 Eric W. Tramel
# SPDX-License-Identifier: MIT
"""Emotion-axis readout helpers for the pinned Nemotron Nano lens."""

from nemotron_mood.analysis import (
    EMOTION_COLORS,
    NEUTRAL_REFERENCE_TEXTS,
    EmotionTrace,
    MoodSummary,
    SentenceMood,
    TokenMood,
    baseline_from_traces,
    calibrate,
    sentence_spans,
    summarize_mood,
    summarize_tokens,
)
from nemotron_mood.anchors import (
    EMOTION_ANCHORS,
    EMOTIONS,
    anchor_metadata,
    anchor_token_ids,
    resolve_anchor_clusters,
)
from nemotron_mood.requests import (
    DEFAULT_CHUNK_TOKENS,
    DEFAULT_MOOD_LAYERS,
    MoodRequest,
    MoodRequestError,
    parse_mood_request,
)
from nemotron_mood.tui import bin_columns, render_mood_tui

__all__ = [
    "DEFAULT_CHUNK_TOKENS",
    "DEFAULT_MOOD_LAYERS",
    "EMOTION_ANCHORS",
    "EMOTION_COLORS",
    "EMOTIONS",
    "NEUTRAL_REFERENCE_TEXTS",
    "EmotionTrace",
    "MoodRequest",
    "MoodRequestError",
    "MoodSummary",
    "SentenceMood",
    "TokenMood",
    "anchor_metadata",
    "anchor_token_ids",
    "baseline_from_traces",
    "bin_columns",
    "calibrate",
    "parse_mood_request",
    "resolve_anchor_clusters",
    "render_mood_tui",
    "sentence_spans",
    "summarize_mood",
    "summarize_tokens",
]
