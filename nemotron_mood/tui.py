# Portions adapted from jlens-mood by Eric W. Tramel.
# Copyright (c) 2026 Eric W. Tramel
# SPDX-License-Identifier: MIT
"""Rich terminal rendering for a completed Nano mood API response."""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

import torch

from nemotron_mood.analysis import EMOTION_COLORS
from nemotron_mood.anchors import EMOTIONS
from nemotron_mood.render import BAR_WIDTH, sanitize, truncate

_HEX_COLOR = re.compile(r"#[0-9a-fA-F]{6}")


def _hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    """Convert a six-digit CSS hex color into an RGB tuple."""
    value = hex_color.removeprefix("#")
    if len(value) != 6 or any(character not in "0123456789abcdefABCDEF" for character in value):
        raise ValueError("color must contain exactly six hexadecimal digits")
    return int(value[0:2], 16), int(value[2:4], 16), int(value[4:6], 16)


def bin_columns(values: torch.Tensor, cols: int) -> torch.Tensor:
    """Average contiguous ``[positions, axes]`` rows into at most ``cols`` bins."""
    if not isinstance(values, torch.Tensor) or values.ndim != 2:
        raise ValueError("values must be a two-dimensional tensor")
    if not len(values):
        raise ValueError("values must contain at least one position")
    if isinstance(cols, bool) or not isinstance(cols, int) or cols <= 0:
        raise ValueError("cols must be a positive integer")
    edges = [round(index * len(values) / cols) for index in range(cols + 1)]
    return torch.stack(
        [
            values[start:end].mean(dim=0)
            for start, end in zip(edges[:-1], edges[1:], strict=True)
            if end > start
        ]
    )


def _colors(response: Mapping[str, Any]) -> dict[str, str]:
    anchors = response.get("anchors")
    anchors = anchors if isinstance(anchors, Mapping) else {}
    result: dict[str, str] = {}
    for emotion in EMOTIONS:
        anchor = anchors.get(emotion)
        candidate = anchor.get("color") if isinstance(anchor, Mapping) else None
        result[emotion] = (
            candidate
            if isinstance(candidate, str) and _HEX_COLOR.fullmatch(candidate)
            else EMOTION_COLORS[emotion]
        )
    return result


def _share_matrix(response: Mapping[str, Any]) -> torch.Tensor:
    rows = []
    for token in response.get("tokens", ()):
        shares = token["shares"]
        rows.append([float(shares[emotion]) for emotion in EMOTIONS])
    if not rows:
        raise ValueError("mood response contains no token rows")
    values = torch.tensor(rows, dtype=torch.float32)
    if not bool(torch.isfinite(values).all()) or bool((values < 0).any()):
        raise ValueError("token shares must be finite and non-negative")
    return values


def _header(response: Mapping[str, Any]) -> str:
    mood = sanitize(str(response["mood"]))
    strongest = sanitize(str(response["strongest"]))
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


def _river_chart(
    response: Mapping[str, Any],
    shares: torch.Tensor,
    colors: Mapping[str, str],
    *,
    width: int,
    plotext: Any,
) -> str:
    chart_width = max(20, min(width, 120))
    columns = min(len(shares), max(2, chart_width - 14))
    binned = bin_columns(shares, columns)
    edges = [round(index * len(shares) / len(binned)) for index in range(len(binned))]
    tokens = response["tokens"]

    plotext.clear_figure()
    plotext.theme("clear")
    plotext.plotsize(chart_width, 14)
    plotext.stacked_bar(
        list(range(len(binned))),
        [
            [float(value) for value in binned[:, axis]]
            for axis in range(len(EMOTIONS))
        ],
        color=[_hex_to_rgb(colors[emotion]) for emotion in EMOTIONS],
        width=1,
    )
    plotext.ylim(0, 1)
    plotext.yticks([0.0, 0.5, 1.0])
    tick_count = min(8, len(binned))
    tick_bins = [
        round(index * (len(binned) - 1) / max(1, tick_count - 1))
        for index in range(tick_count)
    ]
    labels = []
    for tick in tick_bins:
        label = sanitize(str(tokens[edges[tick]]["text"])).strip()
        labels.append(label or ".")
    plotext.xticks(tick_bins, labels)
    return plotext.build()


def render_mood_tui(
    response: Mapping[str, Any],
    include_tokens: bool = False,
    force_terminal: bool | None = None,
    width: int | None = None,
) -> None:
    """Print a colored token stream and mood river from `/api/mood` JSON."""
    try:
        import plotext as plt
        from rich.console import Console
        from rich.table import Table
        from rich.text import Text
    except ImportError as exc:
        raise RuntimeError(
            "the mood TUI requires the optional 'rich' and 'plotext' packages"
        ) from exc

    if width is not None and (isinstance(width, bool) or width <= 0):
        raise ValueError("width must be a positive integer or None")
    colors = _colors(response)
    share_values = _share_matrix(response)
    threshold = float(response["threshold"])
    console = Console(
        highlight=False,
        force_terminal=force_terminal,
        no_color=False if force_terminal else None,
        width=width,
    )

    console.print(_header(response), style="bold")
    console.print()
    for emotion, raw_share in sorted(
        response["shares"].items(), key=lambda item: -float(item[1])
    ):
        share = float(raw_share)
        filled = max(0, min(BAR_WIDTH, round(share * BAR_WIDTH)))
        line = Text(f"  {sanitize(str(emotion)):<9} ")
        line.append("█" * filled, style=colors[str(emotion)])
        line.append("░" * (BAR_WIDTH - filled), style="dim")
        line.append(f" {share:>4.0%}")
        console.print(line)

    console.print()
    stream = Text(no_wrap=False)
    for token in response["tokens"]:
        mood = str(token["mood"])
        strongest = str(token["strongest"])
        intensity = float(token["intensity"])
        if mood == "neutral" and strongest != "neutral":
            style = "dim"
        elif intensity >= 2 * threshold:
            style = f"bold {colors[strongest]}"
        else:
            style = colors[strongest]
        stream.append(sanitize(str(token["text"])), style=style)
    console.print(stream)
    console.print()

    chart = _river_chart(
        response,
        share_values,
        colors,
        width=min(console.width, 120),
        plotext=plt,
    )
    console.print(Text.from_ansi(chart), soft_wrap=True)

    sentences = response.get("sentences", ())
    if sentences:
        console.print()
        for sentence in sentences:
            mood = str(sentence["mood"])
            strongest = str(sentence["strongest"])
            line = Text("  ")
            if mood == "neutral" and strongest != "neutral":
                line.append(f"{'-':>9}      ", style="dim")
            else:
                line.append(f"{mood:>9} ", style=f"bold {colors[strongest]}")
                line.append(f"{float(sentence['shares'][mood]):>4.0%} ")
            line.append(truncate(str(sentence["text"])))
            console.print(line)

    if include_tokens:
        console.print()
        table = Table(show_header=True, header_style="bold", box=None, pad_edge=False)
        table.add_column("Pos", justify="right")
        table.add_column("ID", justify="right")
        table.add_column("Token")
        table.add_column("Mood")
        table.add_column("Share", justify="right")
        table.add_column("Intensity", justify="right")
        for token in response["tokens"]:
            mood = str(token["mood"])
            strongest = str(token["strongest"])
            gated = mood == "neutral" and strongest != "neutral"
            label = f"neutral ({strongest})" if gated else mood
            share_key = strongest if gated else mood
            style = "dim" if gated else colors[strongest]
            table.add_row(
                str(int(token["position"])),
                str(int(token["id"])),
                Text(repr(sanitize(str(token["text"])))),
                Text(label, style=style),
                f"{float(token['shares'][share_key]):.0%}",
                f"{float(token['intensity']):.2f}",
            )
        console.print(table)


__all__ = ["_hex_to_rgb", "bin_columns", "render_mood_tui"]
