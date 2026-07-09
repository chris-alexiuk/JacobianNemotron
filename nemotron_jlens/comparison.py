"""Compare recorded focus/control fixtures without loading model weights."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from statistics import median
from typing import Any

from nemotron_jlens.config import NEMOTRON, UPSTREAM_JLENS_COMMIT
from nemotron_jlens.provenance import require_sha256

COMPARISON_SCHEMA = "nemotron-jlens-comparison/v1"

_EXPECTED_EXAMPLES = {
    "focus": ("modulation-topic", "Concentrate on ocean creatures"),
    "neutral": ("modulation-topic-neutral", "Don't write anything else"),
    "suppression": ("modulation-topic-suppress", "Don't think about ocean creatures"),
    "suppress": ("modulation-topic-suppress", "Don't think about ocean creatures"),
    "mention": ("modulation-topic-mention-control", "contains a reference to ocean creatures"),
}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _token_key(value: str) -> str:
    return value.strip().lower()


def _path_value(root: Mapping[str, Any], path: Sequence[str]) -> Any:
    value: Any = root
    for key in path:
        if not isinstance(value, Mapping) or key not in value:
            return None
        value = value[key]
    return value


def _provenance_value(
    fixture: Mapping[str, Any],
    *,
    name: str,
    paths: Sequence[Sequence[str]],
) -> str:
    located = [(path, _path_value(fixture, path)) for path in paths]
    missing = [
        ".".join(path)
        for path, value in located
        if not isinstance(value, str) or not value
    ]
    if missing:
        raise ValueError(
            f"recorded fixture is missing {name} at {', '.join(missing)}"
        )
    values = [value for _, value in located]
    if len(set(values)) != 1:
        raise ValueError(f"recorded fixture has inconsistent {name} values")
    return values[0]


def _layers_by_type(fixture: Mapping[str, Any]) -> dict[str, tuple[int, ...]]:
    meta = fixture.get("meta")
    raw = meta.get("layers_by_type") if isinstance(meta, Mapping) else None
    if not isinstance(raw, Mapping) or not raw:
        raise ValueError("fixture is missing meta.layers_by_type")
    layers_by_type: dict[str, tuple[int, ...]] = {}
    for readout, values in raw.items():
        if not isinstance(readout, str) or not readout:
            raise ValueError("fixture has an invalid readout type")
        if (
            not isinstance(values, list)
            or not values
            or any(
                isinstance(value, bool) or not isinstance(value, int)
                for value in values
            )
            or len(values) != len(set(values))
        ):
            raise ValueError(f"fixture has invalid layers for {readout}")
        layers_by_type[readout] = tuple(values)
    return layers_by_type


def _tracked_entries(
    tracked: Any, *, position: int, readout: str
) -> dict[str, dict[str, Any]]:
    if isinstance(tracked, list):
        pairs = [
            (entry.get("token") if isinstance(entry, Mapping) else None, entry)
            for entry in tracked
        ]
    elif isinstance(tracked, Mapping):
        pairs = list(tracked.items())
    else:
        raise ValueError(
            f"generated position {position} has no tracked measurements for {readout}"
        )

    entries: dict[str, dict[str, Any]] = {}
    for map_key, raw_entry in pairs:
        if not isinstance(raw_entry, Mapping):
            raise ValueError(
                f"generated position {position} has an invalid tracked entry for {readout}"
            )
        label = raw_entry.get("token", map_key)
        if not isinstance(label, str) or not _token_key(label):
            raise ValueError(
                f"generated position {position} has an invalid tracked label for {readout}"
            )
        key = _token_key(label)
        if key in entries:
            raise ValueError(
                f"generated position {position} repeats tracked token {label!r} for {readout}"
            )
        entries[key] = {**raw_entry, "token": label}
    if not entries:
        raise ValueError(
            f"generated position {position} has no tracked measurements for {readout}"
        )
    return entries


def _measurement_rows(
    fixture: Mapping[str, Any],
    layers_by_type: Mapping[str, tuple[int, ...]],
) -> tuple[list[dict[str, Any]], dict[str, dict[str, list[dict[str, Any]]]]]:
    raw_tokens = fixture.get("tokens")
    if not isinstance(raw_tokens, list):
        raise ValueError("fixture tokens must be a list")
    generated = [
        token
        for token in raw_tokens
        if isinstance(token, Mapping) and token.get("is_generated") is True
    ]
    if not generated:
        raise ValueError("recorded fixture has no is_generated assistant positions")

    positions: set[int] = set()
    ordered: list[dict[str, Any]] = []
    for raw in generated:
        position = raw.get("position")
        if isinstance(position, bool) or not isinstance(position, int):
            raise ValueError("generated token has an invalid absolute position")
        if position in positions:
            raise ValueError(f"duplicate generated token position {position}")
        positions.add(position)
        ordered.append(dict(raw))
    ordered.sort(key=lambda row: row["position"])

    observations: dict[str, dict[str, list[dict[str, Any]]]] = {
        readout: {} for readout in layers_by_type
    }
    expected_keys: dict[str, set[str] | None] = {
        readout: None for readout in layers_by_type
    }
    token_ids: dict[tuple[str, str], int] = {}

    for response_position, token in enumerate(ordered):
        position = token["position"]
        results = token.get("results")
        if not isinstance(results, list):
            raise ValueError(f"generated position {position} is missing results")
        by_type: dict[str, Mapping[str, Any]] = {}
        for result in results:
            if not isinstance(result, Mapping) or not isinstance(
                result.get("type"), str
            ):
                continue
            readout = result["type"]
            if readout in by_type:
                raise ValueError(
                    f"generated position {position} repeats result type {readout}"
                )
            by_type[readout] = result

        for readout, layers in layers_by_type.items():
            if readout not in by_type:
                raise ValueError(
                    f"generated position {position} is missing result type {readout}"
                )
            entries = _tracked_entries(
                by_type[readout].get("tracked"),
                position=position,
                readout=readout,
            )
            keys = set(entries)
            if expected_keys[readout] is None:
                expected_keys[readout] = keys
            elif keys != expected_keys[readout]:
                raise ValueError(
                    f"tracked token set changes across assistant positions for {readout}"
                )

            for key, entry in entries.items():
                ranks = entry.get("ranks")
                probs = entry.get("probs")
                if not isinstance(ranks, list) or len(ranks) != len(layers):
                    raise ValueError(
                        f"tracked token {entry['token']!r} has invalid ranks for {readout}"
                    )
                if not isinstance(probs, list) or len(probs) != len(layers):
                    raise ValueError(
                        f"tracked token {entry['token']!r} has invalid probs for {readout}"
                    )
                token_id = entry.get("id")
                if token_id is not None:
                    if (
                        isinstance(token_id, bool)
                        or not isinstance(token_id, int)
                        or token_id < 0
                    ):
                        raise ValueError(
                            f"tracked token {entry['token']!r} has an invalid id"
                        )
                    identity = (readout, key)
                    if identity in token_ids and token_ids[identity] != token_id:
                        raise ValueError(
                            f"tracked token {entry['token']!r} changes id within {readout}"
                        )
                    token_ids[identity] = token_id

                for index, layer in enumerate(layers):
                    rank = ranks[index]
                    probability = probs[index]
                    if rank is not None and (
                        isinstance(rank, bool) or not isinstance(rank, int) or rank < 1
                    ):
                        raise ValueError(
                            f"tracked token {entry['token']!r} has an invalid rank"
                        )
                    if probability is not None and (
                        isinstance(probability, bool)
                        or not isinstance(probability, (int, float))
                        or not 0 <= probability <= 1
                    ):
                        raise ValueError(
                            f"tracked token {entry['token']!r} has an invalid probability"
                        )
                    observations[readout].setdefault(key, []).append(
                        {
                            "token": entry["token"],
                            "token_id": token_id,
                            "layer": layer,
                            "absolute_position": position,
                            "response_relative_position": response_position,
                            "rank": rank,
                            "probability": (
                                None if probability is None else float(probability)
                            ),
                        }
                    )
    return ordered, observations


def _summarize(observations: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    ranks = [row["rank"] for row in observations if row["rank"] is not None]
    probabilities = [
        row["probability"] for row in observations if row["probability"] is not None
    ]
    if not ranks or not probabilities:
        token = observations[0]["token"] if observations else "unknown"
        raise ValueError(
            f"tracked token {token!r} has no complete rank/probability data"
        )

    best = min(
        (row for row in observations if row["rank"] is not None),
        key=lambda row: (
            row["rank"],
            -(row["probability"] if row["probability"] is not None else -1.0),
            row["layer"],
            row["absolute_position"],
        ),
    )
    max_probability_at = min(
        (row for row in observations if row["probability"] is not None),
        key=lambda row: (
            -row["probability"],
            row["rank"] if row["rank"] is not None else float("inf"),
            row["layer"],
            row["absolute_position"],
        ),
    )
    known_ids = {row["token_id"] for row in observations if row["token_id"] is not None}
    if len(known_ids) > 1:
        raise ValueError(
            f"tracked token {observations[0]['token']!r} has inconsistent ids"
        )

    def location(row: Mapping[str, Any]) -> dict[str, Any]:
        return {
            "layer": row["layer"],
            "absolute_position": row["absolute_position"],
            "response_relative_position": row["response_relative_position"],
            "rank": row["rank"],
            "probability": row["probability"],
        }

    return {
        "token": observations[0]["token"],
        "token_id": next(iter(known_ids), None),
        "observation_count": len(observations),
        "rank_observation_count": len(ranks),
        "probability_observation_count": len(probabilities),
        "best_rank": min(ranks),
        "median_rank": float(median(ranks)),
        "max_probability": max(probabilities),
        "best_layer": best["layer"],
        "best_absolute_position": best["absolute_position"],
        "best_response_relative_position": best["response_relative_position"],
        "best": location(best),
        "max_probability_at": location(max_probability_at),
    }


def _validated_condition_identity(
    fixture: Mapping[str, Any], label: str
) -> dict[str, str]:
    if label not in _EXPECTED_EXAMPLES:
        raise ValueError(f"unknown directed-modulation condition label {label!r}")
    fixture_info = fixture["_fixture"]
    meta = fixture.get("meta")
    provenance = fixture_info.get("provenance")
    if not isinstance(meta, Mapping) or meta.get("model") != NEMOTRON.model_id:
        raise ValueError(f"condition {label!r} is not the pinned Nemotron model")
    if not isinstance(provenance, Mapping):
        raise ValueError(f"condition {label!r} is missing required provenance")
    model = provenance.get("model")
    tokenizer = provenance.get("tokenizer")
    lens = provenance.get("lens")
    prompt = provenance.get("prompt")
    exporter = provenance.get("exporter")
    if (
        not isinstance(model, Mapping)
        or model.get("id") != NEMOTRON.model_id
        or not isinstance(tokenizer, Mapping)
        or tokenizer.get("id") != NEMOTRON.model_id
    ):
        raise ValueError(f"condition {label!r} has invalid model/tokenizer provenance")
    if (
        not isinstance(lens, Mapping)
        or not isinstance(prompt, Mapping)
        or not isinstance(exporter, Mapping)
    ):
        raise ValueError(
            f"condition {label!r} has incomplete lens/prompt/exporter provenance"
        )

    model_revision = _provenance_value(
        fixture,
        name="model revision",
        paths=(
            ("_fixture", "model_revision"),
            ("_fixture", "tokenizer", "revision"),
            ("_fixture", "provenance", "model", "revision"),
            ("_fixture", "provenance", "tokenizer", "revision"),
        ),
    )
    lens_sha = require_sha256(
        _provenance_value(
            fixture,
            name="lens SHA-256",
            paths=(
                ("_fixture", "lens_sha256"),
                ("_fixture", "lens", "sha256"),
                ("_fixture", "provenance", "lens", "sha256"),
            ),
        ),
        field=f"condition {label} lens SHA-256",
    )
    prompt_sha = require_sha256(
        _provenance_value(
            fixture,
            name="prompt SHA-256",
            paths=(
                ("_fixture", "prompt_sha256"),
                ("_fixture", "provenance", "prompt", "sha256"),
            ),
        ),
        field=f"condition {label} prompt SHA-256",
    )
    if model_revision != NEMOTRON.revision:
        raise ValueError(f"condition {label!r} is not the pinned model revision")

    expected_example, required_text = _EXPECTED_EXAMPLES[label]
    if (
        prompt.get("example") != expected_example
        or fixture_info.get("example") != expected_example
    ):
        raise ValueError(f"condition {label!r} has the wrong prompt condition identity")
    prompt_text = fixture_info.get("prompt")
    if not isinstance(prompt_text, str) or required_text not in prompt_text:
        raise ValueError(f"condition {label!r} does not contain its registered instruction")
    if hashlib.sha256(prompt_text.encode()).hexdigest() != prompt_sha:
        raise ValueError(f"condition {label!r} prompt hash does not match its prompt text")
    corpus_sha = require_sha256(
        lens.get("corpus_manifest_sha256"), field=f"condition {label} corpus SHA-256"
    )
    lens_source = require_sha256(
        lens.get("adaptation_source_sha256"), field=f"condition {label} lens source SHA-256"
    )
    require_sha256(
        exporter.get("adaptation_source_sha256"),
        field=f"condition {label} exporter source SHA-256",
    )
    if (
        lens.get("upstream_jlens_commit") != UPSTREAM_JLENS_COMMIT
        or exporter.get("upstream_jlens_commit") != UPSTREAM_JLENS_COMMIT
    ):
        raise ValueError(f"condition {label!r} has the wrong upstream jlens commit")
    if not isinstance(lens.get("n_prompts"), int) or lens["n_prompts"] <= 0:
        raise ValueError(f"condition {label!r} has invalid lens prompt provenance")
    if not isinstance(lens.get("source_layers"), list) or not lens["source_layers"]:
        raise ValueError(f"condition {label!r} has invalid lens layer provenance")
    return {
        "model_revision": model_revision,
        "lens_sha256": lens_sha,
        "corpus_manifest_sha256": corpus_sha,
        "lens_adaptation_source_sha256": lens_source,
        "example": expected_example,
        "prompt_sha256": prompt_sha,
    }


def _condition(label: str, path: str | Path) -> dict[str, Any]:
    fixture_path = Path(path)
    with fixture_path.open(encoding="utf-8") as handle:
        fixture = json.load(handle)
    if not isinstance(fixture, Mapping):
        raise ValueError(f"fixture {fixture_path} must contain a JSON object")
    fixture_info = fixture.get("_fixture")
    if not isinstance(fixture_info, Mapping) or fixture_info.get("mode") != "recorded":
        raise ValueError(f"condition {label!r} is not a recorded fixture")

    identity = _validated_condition_identity(fixture, label)
    model_revision = identity["model_revision"]
    lens_sha256 = identity["lens_sha256"]
    layers_by_type = _layers_by_type(fixture)
    tokens, observations = _measurement_rows(fixture, layers_by_type)
    summaries = {
        readout: {key: _summarize(rows) for key, rows in tokens_by_key.items()}
        for readout, tokens_by_key in observations.items()
    }
    meta = fixture.get("meta")
    return {
        "label": label,
        "fixture": {
            "path": str(fixture_path.resolve()),
            "sha256": _sha256_file(fixture_path),
            "model": meta.get("model") if isinstance(meta, Mapping) else None,
            "model_revision": model_revision,
            "lens_sha256": lens_sha256,
            "corpus_manifest_sha256": identity["corpus_manifest_sha256"],
            "lens_adaptation_source_sha256": identity["lens_adaptation_source_sha256"],
            "example": identity["example"],
            "prompt_sha256": identity["prompt_sha256"],
        },
        "assistant": {
            "positions": [row["position"] for row in tokens],
            "response_relative_positions": list(range(len(tokens))),
            "tokens": [row.get("token") for row in tokens],
            "token_ids": [row.get("id") for row in tokens],
        },
        "layers_by_type": {key: list(value) for key, value in layers_by_type.items()},
        "readouts": summaries,
    }


def _compatibility_key(condition: Mapping[str, Any]) -> tuple[Any, ...]:
    fixture = condition["fixture"]
    layer_sets = tuple(
        sorted(
            (readout, tuple(sorted(layers)))
            for readout, layers in condition["layers_by_type"].items()
        )
    )
    return fixture["model_revision"], fixture["lens_sha256"], layer_sets


def _check_compatible(focus: Mapping[str, Any], control: Mapping[str, Any]) -> None:
    focus_fixture = focus["fixture"]
    control_fixture = control["fixture"]
    if focus_fixture["model_revision"] != control_fixture["model_revision"]:
        raise ValueError(
            f"condition {control['label']!r} has a different model revision"
        )
    if focus_fixture["lens_sha256"] != control_fixture["lens_sha256"]:
        raise ValueError(f"condition {control['label']!r} has a different lens SHA-256")
    if _compatibility_key(focus)[2] != _compatibility_key(control)[2]:
        raise ValueError(f"condition {control['label']!r} has different layer sets")
    for field, description in (
        ("corpus_manifest_sha256", "corpus manifest"),
        ("lens_adaptation_source_sha256", "lens adaptation source"),
    ):
        if focus_fixture[field] != control_fixture[field]:
            raise ValueError(f"condition {control['label']!r} has a different {description}")
    if (
        focus["assistant"]["tokens"] != control["assistant"]["tokens"]
        or focus["assistant"]["token_ids"] != control["assistant"]["token_ids"]
    ):
        raise ValueError(f"condition {control['label']!r} has a different assistant sequence")

    if set(focus["readouts"]) != set(control["readouts"]):
        raise ValueError(f"condition {control['label']!r} has different readout types")
    for readout, focus_tokens in focus["readouts"].items():
        control_tokens = control["readouts"][readout]
        if set(focus_tokens) != set(control_tokens):
            raise ValueError(
                f"condition {control['label']!r} has different tracked tokens for {readout}"
            )
        for key, focus_summary in focus_tokens.items():
            control_summary = control_tokens[key]
            left_id = focus_summary["token_id"]
            right_id = control_summary["token_id"]
            if left_id is not None and right_id is not None and left_id != right_id:
                raise ValueError(
                    f"condition {control['label']!r} changes token id for "
                    f"{focus_summary['token']!r} in {readout}"
                )


def _contrasts(focus: Mapping[str, Any], control: Mapping[str, Any]) -> dict[str, Any]:
    readouts: dict[str, Any] = {}
    for readout, focus_tokens in focus["readouts"].items():
        rows: dict[str, Any] = {}
        for key, focus_summary in focus_tokens.items():
            control_summary = control["readouts"][readout][key]
            label = focus_summary["token"]
            rows[label] = {
                "focus_minus_control_max_probability": (
                    focus_summary["max_probability"]
                    - control_summary["max_probability"]
                ),
                "focus_minus_control_reciprocal_best_rank": (
                    1.0 / focus_summary["best_rank"]
                    - 1.0 / control_summary["best_rank"]
                ),
                "median_rank_improvement": (
                    control_summary["median_rank"] - focus_summary["median_rank"]
                ),
            }
        readouts[readout] = rows
    return {
        "focus": focus["label"],
        "control": control["label"],
        "readouts": readouts,
    }


def compare_fixtures(
    focus_path: str | Path,
    controls: Mapping[str, str | Path],
    *,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    """Build an auditable comparison from recorded full-vocabulary rows."""
    if not controls:
        raise ValueError("at least one control fixture is required")
    if "focus" in controls:
        raise ValueError("control label 'focus' is reserved")

    focus = _condition("focus", focus_path)
    control_conditions = {
        label: _condition(label, path) for label, path in controls.items()
    }
    for control in control_conditions.values():
        _check_compatible(focus, control)

    compatibility = {
        "model_revision": focus["fixture"]["model_revision"],
        "lens_sha256": focus["fixture"]["lens_sha256"],
        "layers_by_type": {
            readout: sorted(layers)
            for readout, layers in focus["layers_by_type"].items()
        },
    }
    artifact = {
        "schema": COMPARISON_SCHEMA,
        "metric_definitions": {
            "scope": "tokens with is_generated=true",
            "best": ("lowest full-vocabulary rank; highest probability breaks ties"),
            "probability_delta": "focus max probability minus control max probability",
            "reciprocal_rank_delta": ("1/focus best rank minus 1/control best rank"),
            "median_rank_improvement": (
                "control median rank minus focus median rank; positive favors focus"
            ),
        },
        "compatibility": compatibility,
        "conditions": {"focus": focus, **control_conditions},
        "contrasts": {
            label: _contrasts(focus, control)
            for label, control in control_conditions.items()
        },
    }
    if output_path is not None:
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(artifact, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
            encoding="utf-8",
        )
    return artifact
