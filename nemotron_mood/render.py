# Portions adapted from jlens-mood by Eric W. Tramel.
# Copyright (c) 2026 Eric W. Tramel
# SPDX-License-Identifier: MIT
"""Plain-terminal rendering for mood API responses."""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

from nemotron_mood.analysis import MoodSummary

BAR_WIDTH = 24
_CONTROL = re.compile(r"[\x00-\x08\x0a-\x1f\x7f-\x9f]")


def sanitize(text: str) -> str:
    return _CONTROL.sub(" ", text)


def truncate(text: str, limit: int = 60) -> str:
    text = sanitize(text.strip())
    return text if len(text) <= limit else text[: limit - 3] + "..."


def head_line(summary: MoodSummary) -> str:
    if summary.dominant == "neutral" and summary.strongest != "neutral":
        return (
            f"mood: neutral (strongest: {summary.strongest} "
            f"{summary.shares[summary.strongest]:.0%}, "
            f"intensity {summary.intensity:.2f}, threshold {summary.threshold:.2f})"
        )
    return (
        f"mood: {summary.dominant} ({summary.shares[summary.dominant]:.0%}, "
        f"intensity {summary.intensity:.2f}, threshold {summary.threshold:.2f})"
    )


def _response_head(response: Mapping[str, Any]) -> str:
    mood = str(response["mood"])
    strongest = str(response["strongest"])
    intensity = float(response["intensity"])
    threshold = float(response["threshold"])
    shares = response["shares"]
    if mood == "neutral" and strongest != "neutral":
        return (
            f"mood: neutral (strongest: {strongest} "
            f"{float(shares[strongest]):.0%}, intensity {intensity:.2f}, "
            f"threshold {threshold:.2f})"
        )
    return (
        f"mood: {mood} ({float(shares[mood]):.0%}, "
        f"intensity {intensity:.2f}, threshold {threshold:.2f})"
    )


def _bar_lines(shares: Mapping[str, Any]) -> list[str]:
    lines: list[str] = []
    for emotion, raw_share in sorted(
        shares.items(), key=lambda item: -float(item[1])
    ):
        share = float(raw_share)
        filled = max(0, min(BAR_WIDTH, round(share * BAR_WIDTH)))
        lines.append(
            f"  {emotion:<9} {'#' * filled}{'.' * (BAR_WIDTH - filled)} {share:>4.0%}"
        )
    return lines


def render_mood_response(
    response: Mapping[str, Any], *, include_tokens: bool = False
) -> str:
    """Render the stable `/api/mood` response as plain terminal text."""
    lines = [_response_head(response), "", *_bar_lines(response["shares"])]
    sentences = response.get("sentences", ())
    if len(sentences) > 1:
        lines.append("")
        for sentence in sentences:
            mood = str(sentence["mood"])
            strongest = str(sentence["strongest"])
            label = mood if mood != "neutral" or strongest == "neutral" else "-"
            share = (
                f"{float(sentence['shares'][mood]):>4.0%}"
                if label != "-"
                else "    "
            )
            lines.append(
                f"  {label:>9} {share}  {truncate(str(sentence['text']))}"
            )

    if include_tokens:
        lines.extend(("", f"{'pos':>4}  {'id':>7}  {'token':<20} {'mood':<10} {'intensity':>9}"))
        for token in response.get("tokens", ()):
            token_text = sanitize(str(token["text"]))
            lines.append(
                f"{int(token['position']):>4}  {int(token['id']):>7}  "
                f"{token_text!r:<20} {str(token['mood']):<10} "
                f"{float(token['intensity']):>9.2f}"
            )
    return "\n".join(lines)
