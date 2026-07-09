# Portions adapted from jlens-mood by Eric W. Tramel.
# Copyright (c) 2026 Eric W. Tramel
# SPDX-License-Identifier: MIT
"""Command-line client for the persistent Nano mood service."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import urllib.error
import urllib.request
from collections.abc import Sequence
from typing import Any
from urllib.parse import urlsplit

from nemotron_mood.render import render_mood_response
from nemotron_mood.requests import (
    DEFAULT_CHUNK_TOKENS,
    MoodRequest,
    MoodRequestError,
    parse_mood_request,
)

DEFAULT_URL = "http://127.0.0.1:8000"
DEFAULT_TIMEOUT_SECONDS = 900.0


class MoodClientError(RuntimeError):
    pass


def _positive_float(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a number") from exc
    if not math.isfinite(parsed) or parsed <= 0:
        raise argparse.ArgumentTypeError("must be a finite positive number")
    return parsed


def _layer_list(value: str) -> list[int]:
    try:
        return [int(part.strip()) for part in value.split(",") if part.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "layers must be comma-separated integers"
        ) from exc


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Read text with the persistent Nemotron Nano Jacobian lens."
    )
    parser.add_argument("text", nargs="?", help="Text to analyze")
    parser.add_argument("--file", "-f", help="Read text from a UTF-8 file")
    output = parser.add_mutually_exclusive_group()
    output.add_argument(
        "--json", dest="output_mode", action="store_const", const="json",
        help="Print the exact API JSON",
    )
    output.add_argument(
        "--pretty", "--tui", dest="output_mode", action="store_const",
        const="pretty", help="Force the colored token stream and mood river",
    )
    output.add_argument(
        "--plain", dest="output_mode", action="store_const", const="plain",
        help="Force stable plain text for logs and pipes",
    )
    parser.set_defaults(output_mode="auto")
    parser.add_argument(
        "--tokens", "-T", action="store_true", help="Include per-token mood rows"
    )
    parser.add_argument(
        "--layers", type=_layer_list, help="Comma-separated source layers (0-50)"
    )
    parser.add_argument(
        "--chunk-tokens",
        type=int,
        default=DEFAULT_CHUNK_TOKENS,
        help=f"Independent prefill window size (default: {DEFAULT_CHUNK_TOKENS})",
    )
    parser.add_argument(
        "--url",
        default=os.environ.get("NEMOTRON_MOOD_URL", DEFAULT_URL),
        help=f"Server base URL or mood endpoint (default: {DEFAULT_URL})",
    )
    parser.add_argument(
        "--timeout",
        type=_positive_float,
        default=DEFAULT_TIMEOUT_SECONDS,
        help=f"HTTP timeout in seconds (default: {DEFAULT_TIMEOUT_SECONDS:g})",
    )
    return parser


def read_input(args: argparse.Namespace, parser: argparse.ArgumentParser) -> str:
    if args.text is not None and args.file is not None:
        parser.error("give a string or --file, not both")
    if args.text is not None:
        text = args.text
    elif args.file is not None:
        try:
            with open(args.file, encoding="utf-8") as handle:
                text = handle.read()
        except OSError as exc:
            parser.error(str(exc))
    elif not sys.stdin.isatty():
        text = sys.stdin.read()
    else:
        parser.error("no input: pass a string, --file PATH, or pipe text on stdin")
    if not text.strip():
        parser.error("input text is empty")
    return text


def mood_endpoint(url: str) -> str:
    value = url.strip().rstrip("/")
    parsed = urlsplit(value)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise MoodClientError("--url must be an absolute http:// or https:// URL")
    if parsed.path.endswith("/api/mood"):
        return value
    return f"{value}/api/mood"


def _error_message(payload: object, fallback: str) -> str:
    if isinstance(payload, dict):
        message = payload.get("error")
        request_id = payload.get("request_id")
        if isinstance(message, str) and message:
            return f"{message} (request {request_id})" if request_id else message
    return fallback


def request_mood(
    url: str, mood_request: MoodRequest, *, timeout: float
) -> dict[str, Any]:
    endpoint = mood_endpoint(url)
    data = json.dumps(mood_request.to_payload(), ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        endpoint,
        data=data,
        headers={"Content-Type": "application/json", "Accept": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read()
    except urllib.error.HTTPError as exc:
        try:
            payload = json.loads(exc.read().decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            payload = None
        raise MoodClientError(
            _error_message(payload, f"server returned HTTP {exc.code}")
        ) from exc
    except urllib.error.URLError as exc:
        raise MoodClientError(f"could not reach {endpoint}: {exc.reason}") from exc
    except TimeoutError as exc:
        raise MoodClientError(f"request timed out after {timeout:g} seconds") from exc

    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise MoodClientError("server returned invalid JSON") from exc
    if not isinstance(payload, dict):
        raise MoodClientError("server returned a non-object JSON response")
    if "error" in payload:
        raise MoodClientError(_error_message(payload, "mood inference failed"))
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    text = read_input(args, parser)
    body: dict[str, Any] = {
        "text": text,
        "chunk_tokens": args.chunk_tokens,
    }
    if args.layers is not None:
        body["layers"] = args.layers
    try:
        mood_request = parse_mood_request(body)
        response = request_mood(args.url, mood_request, timeout=args.timeout)
    except (MoodRequestError, MoodClientError) as exc:
        parser.exit(1, f"jlens-mood: error: {exc}\n")

    if args.output_mode == "json":
        print(json.dumps(response, ensure_ascii=False, sort_keys=True))
    elif args.output_mode == "pretty" or (
        args.output_mode == "auto" and sys.stdout.isatty()
    ):
        try:
            from nemotron_mood.tui import render_mood_tui
        except ImportError:
            parser.exit(
                1,
                "jlens-mood: error: pretty output requires rich and plotext\n",
            )
        render_mood_tui(
            response,
            include_tokens=args.tokens,
            force_terminal=True if args.output_mode == "pretty" else None,
        )
    else:
        print(render_mood_response(response, include_tokens=args.tokens))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
