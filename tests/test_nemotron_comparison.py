from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

import pytest

from nemotron_jlens.cli import _run, build_parser
from nemotron_jlens.comparison import COMPARISON_SCHEMA, compare_fixtures
from nemotron_jlens.config import NEMOTRON, UPSTREAM_JLENS_COMMIT

JACOBIAN = "JACOBIAN_LENS"
LOGIT = "LOGIT_LENS"


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


LENS_SHA = _sha256("recorded-lens")
OTHER_LENS_SHA = _sha256("different-recorded-lens")
CORPUS_SHA = _sha256("recorded-corpus")
SOURCE_SHA = _sha256("recording-source")


def _tracked(
    ranks: list[int], probs: list[float], *, legacy: bool
) -> list[dict[str, object]] | dict[str, dict[str, object]]:
    entry: dict[str, object] = {"id": 7, "ranks": ranks, "probs": probs}
    if legacy:
        return {"fish": entry}
    return [{"token": " fish", **entry}]


def _fixture(
    *,
    focus: bool,
    legacy: bool = False,
    mode: str = "recorded",
    revision: str = NEMOTRON.revision,
    lens_sha: str = LENS_SHA,
    layers: list[int] | None = None,
) -> dict[str, object]:
    layers = layers or [0, 2]
    if focus:
        jacobian_rows = [([5, 2], [0.1, 0.4]), ([3, 4], [0.2, 0.3])]
        logit_rows = [([20, 10], [0.01, 0.02]), ([15, 12], [0.03, 0.04])]
        positions = [10, 11]
    else:
        jacobian_rows = [([10, 4], [0.05, 0.1]), ([8, 6], [0.08, 0.07])]
        logit_rows = [([25, 20], [0.005, 0.01]), ([22, 18], [0.02, 0.015])]
        positions = [20, 21]

    if focus:
        example = "modulation-topic"
        prompt = (
            "Write the carrier sentence. Concentrate on ocean creatures while you "
            "write the sentence."
        )
    else:
        example = "modulation-topic-neutral"
        prompt = "Write the carrier sentence. Don't write anything else."
    prompt_sha = _sha256(prompt)

    def results(
        jacobian: tuple[list[int], list[float]],
        logit: tuple[list[int], list[float]],
    ) -> list[dict[str, object]]:
        return [
            {
                "type": JACOBIAN,
                "tracked": _tracked(*jacobian, legacy=legacy),
            },
            {"type": LOGIT, "tracked": _tracked(*logit, legacy=legacy)},
        ]

    tokens = [
        {
            "kind": "token",
            "position": 0,
            "token": "prompt",
            "id": 1,
            "is_generated": False,
            "results": results(([1, 1], [0.99, 0.99]), ([1, 1], [0.99, 0.99])),
        }
    ]
    for index, position in enumerate(positions):
        tokens.append(
            {
                "kind": "token",
                "position": position,
                "token": f" response-{index}",
                "id": 100 + index,
                "is_generated": True,
                "results": results(jacobian_rows[index], logit_rows[index]),
            }
        )
    lens_provenance = {
        "sha256": lens_sha,
        "n_prompts": 1000,
        "source_layers": layers,
        "target_layer": 51,
        "corpus_manifest_sha256": CORPUS_SHA,
        "adaptation_source_sha256": SOURCE_SHA,
        "upstream_jlens_commit": UPSTREAM_JLENS_COMMIT,
    }
    tokenizer = {"id": NEMOTRON.model_id, "revision": revision}
    return {
        "_fixture": {
            "mode": mode,
            "example": example,
            "prompt": prompt,
            "prompt_sha256": prompt_sha,
            "model_revision": revision,
            "tokenizer": dict(tokenizer),
            "lens_sha256": lens_sha,
            "lens": dict(lens_provenance),
            "provenance": {
                "model": {"id": NEMOTRON.model_id, "revision": revision},
                "tokenizer": dict(tokenizer),
                "lens": dict(lens_provenance),
                "prompt": {"example": example, "sha256": prompt_sha},
                "exporter": {
                    "upstream_jlens_commit": UPSTREAM_JLENS_COMMIT,
                    "adaptation_source_sha256": SOURCE_SHA,
                },
            },
        },
        "meta": {
            "kind": "meta",
            "model": NEMOTRON.model_id,
            "layers_by_type": {JACOBIAN: layers, LOGIT: layers},
        },
        "tokens": tokens,
        "done": {"kind": "done"},
    }


def _write(path: Path, fixture: dict[str, object]) -> Path:
    path.write_text(json.dumps(fixture), encoding="utf-8")
    return path


def test_compare_recorded_fixtures_uses_generated_exact_tracked_rows(tmp_path: Path):
    focus = _write(tmp_path / "focus.json", _fixture(focus=True))
    control = _write(tmp_path / "neutral.json", _fixture(focus=False, legacy=True))
    output = tmp_path / "comparison.json"

    artifact = compare_fixtures(
        focus,
        {"neutral": control},
        output_path=output,
    )

    assert artifact["schema"] == COMPARISON_SCHEMA
    assert artifact["compatibility"] == {
        "model_revision": NEMOTRON.revision,
        "lens_sha256": LENS_SHA,
        "layers_by_type": {JACOBIAN: [0, 2], LOGIT: [0, 2]},
    }
    focus_summary = artifact["conditions"]["focus"]["readouts"][JACOBIAN]["fish"]
    assert focus_summary["best_rank"] == 2
    assert focus_summary["median_rank"] == 3.5
    assert focus_summary["max_probability"] == pytest.approx(0.4)
    assert focus_summary["observation_count"] == 4
    assert focus_summary["best_layer"] == 2
    assert focus_summary["best_absolute_position"] == 10
    assert focus_summary["best_response_relative_position"] == 0
    assert focus_summary["best"] == {
        "layer": 2,
        "absolute_position": 10,
        "response_relative_position": 0,
        "rank": 2,
        "probability": 0.4,
    }
    assert artifact["conditions"]["neutral"]["assistant"]["positions"] == [20, 21]

    contrast = artifact["contrasts"]["neutral"]["readouts"][JACOBIAN][" fish"]
    assert contrast["focus_minus_control_max_probability"] == pytest.approx(0.3)
    assert contrast["focus_minus_control_reciprocal_best_rank"] == pytest.approx(0.25)
    assert contrast["median_rank_improvement"] == pytest.approx(3.5)

    # The prompt's deliberately stronger values are excluded by is_generated.
    assert focus_summary["best_rank"] != 1
    assert json.loads(output.read_text(encoding="utf-8")) == artifact


@pytest.mark.parametrize(
    ("mutator", "message"),
    [
        (lambda value: value["_fixture"].update(mode="illustrative"), "not a recorded"),
        (
            lambda value: value["_fixture"].update(model_revision="other"),
            "inconsistent model revision",
        ),
        (
            lambda value: value["_fixture"].update(lens_sha256="other"),
            "inconsistent lens SHA-256",
        ),
        (
            lambda value: value["_fixture"].update(prompt_sha256="other"),
            "inconsistent prompt SHA-256",
        ),
        (
            lambda value: value["meta"]["layers_by_type"].update(
                {JACOBIAN: [0, 3], LOGIT: [0, 3]}
            ),
            "different layer sets",
        ),
    ],
)
def test_compare_rejects_incompatible_or_unrecorded_controls(
    tmp_path: Path, mutator, message: str
):
    focus_fixture = _fixture(focus=True)
    control_fixture = deepcopy(_fixture(focus=False))
    mutator(control_fixture)
    focus = _write(tmp_path / "focus.json", focus_fixture)
    control = _write(tmp_path / "control.json", control_fixture)

    with pytest.raises(ValueError, match=message):
        compare_fixtures(focus, {"neutral": control})


def test_compare_rejects_cross_condition_lens_sha_mismatch(tmp_path: Path):
    focus = _write(tmp_path / "focus.json", _fixture(focus=True))
    control_fixture = _fixture(focus=False, lens_sha=OTHER_LENS_SHA)
    control = _write(tmp_path / "neutral.json", control_fixture)

    with pytest.raises(ValueError, match="different lens SHA-256"):
        compare_fixtures(focus, {"neutral": control})


def test_compare_cli_parses_repeatable_controls_and_output():
    args = build_parser().parse_args(
        [
            "compare-fixtures",
            "focus.json",
            "--control",
            "neutral=neutral.json",
            "--control",
            "suppress=suppress.json",
            "--output",
            "comparison.json",
        ]
    )
    with patch(
        "nemotron_jlens.cli.compare_fixtures", return_value={"ok": True}
    ) as compare:
        assert _run(args) == {"ok": True}
    compare.assert_called_once_with(
        "focus.json",
        {"neutral": "neutral.json", "suppress": "suppress.json"},
        output_path="comparison.json",
    )
