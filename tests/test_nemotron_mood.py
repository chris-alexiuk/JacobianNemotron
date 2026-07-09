import io
import json

import pytest
import torch

import nemotron_mood.backend as mood_backend
import nemotron_mood.cli as cli
from nemotron_mood.analysis import (
    EMOTIONS,
    EmotionTrace,
    baseline_from_traces,
    calibrate,
    sentence_spans,
    summarize_mood,
    summarize_tokens,
)
from nemotron_mood.anchors import anchor_metadata, anchor_token_ids
from nemotron_mood.render import render_mood_response, sanitize
from nemotron_mood.requests import (
    DEFAULT_CHUNK_TOKENS,
    DEFAULT_MOOD_LAYERS,
    MoodRequestError,
    parse_mood_request,
)
from tests.test_steering_backend import make_tiny_backend


class FakeTokenizer:
    def __init__(self, vocab):
        self.vocab = vocab

    def encode(self, text, *, add_special_tokens=False):
        assert add_special_tokens is False
        return [self.vocab[text]] if text in self.vocab else [0, 1]


def _row(emotion: str, value: float) -> list[float]:
    row = [0.0] * len(EMOTIONS)
    row[EMOTIONS.index(emotion)] = value
    return row


def _trace(rows, tokens=None, token_ids=None, layer=13):
    tokens = tokens or [f" t{i}" for i in range(len(rows))]
    token_ids = token_ids or list(range(100, 100 + len(rows)))
    return EmotionTrace(
        tokens=tokens,
        token_ids=token_ids,
        per_layer={layer: torch.tensor(rows, dtype=torch.float32)},
        dropped={},
    )


def test_anchor_token_ids_keeps_variants_dedupes_and_reports_drops():
    tokenizer = FakeTokenizer({" joy": 7, " Joy": 8, " glad": 9, " Glad": 9})
    ids, dropped = anchor_token_ids(tokenizer, ["joy", "glad", "heartbreak"])
    assert ids == [7, 8, 9]
    assert dropped == ["heartbreak"]
    assert anchor_metadata({emotion: [index] for index, emotion in enumerate(EMOTIONS)}, {})[
        "joy"
    ] == {"token_ids": [2], "dropped_words": []}


def test_emotion_trace_validates_alignment_shape_and_finiteness():
    trace = _trace([_row("joy", 3.0)], tokens=[" hi"], token_ids=[17])
    assert trace.mean.shape == (1, len(EMOTIONS))
    with pytest.raises(ValueError, match="align"):
        EmotionTrace([" hi"], [], {13: torch.zeros(1, len(EMOTIONS))}, {})
    with pytest.raises(ValueError, match="shape"):
        EmotionTrace([" hi"], [17], {13: torch.zeros(1, 2)}, {})


def test_calibrate_subtracts_baseline_and_normalizes():
    scores = torch.tensor([_row("joy", 3.0)])
    baseline = torch.tensor(_row("joy", 2.0))
    calibrated, shares = calibrate(scores, baseline)
    assert calibrated[0, EMOTIONS.index("joy")] == 1.0
    assert torch.allclose(shares.sum(dim=-1), torch.tensor([1.0]))


def test_sentence_spans_drops_trailing_whitespace():
    tokens = [" Good", " day", ".", " Then", " rain", ".", "\n"]
    assert sentence_spans(tokens) == [(0, 3), (3, 6)]


def test_summarize_mood_has_exact_sentence_fields_and_explicit_gate():
    trace = _trace(
        [_row("joy", 4.0)] * 3 + [_row("fear", 6.0)] * 3,
        tokens=[" What", " a", ".", " Then", " fire", "."],
    )
    summary = summarize_mood(trace, torch.zeros(len(EMOTIONS)), threshold=2.0)
    assert summary.mood == "fear"
    assert [sentence.mood for sentence in summary.sentences] == ["joy", "fear"]
    assert summary.sentences[1].to_dict().keys() == {
        "text",
        "mood",
        "strongest",
        "intensity",
        "shares",
        "start",
        "end",
    }

    gated = summarize_mood(
        _trace([_row("anger", 1.5)]),
        torch.zeros(len(EMOTIONS)),
        threshold=2.0,
    )
    assert gated.mood == "neutral"
    assert gated.strongest == "anger"


def test_summarize_tokens_preserves_numeric_identity_and_positions():
    trace = _trace(
        [_row("curiosity", 3.0), _row("neutral", 4.0)],
        tokens=[" Spider", "Spider"],
        token_ids=[44, 45],
    )
    rows = summarize_tokens(trace, torch.zeros(len(EMOTIONS)), threshold=2.0)
    assert [row.to_dict()["id"] for row in rows] == [44, 45]
    assert [row.position for row in rows] == [0, 1]
    assert rows[1].mood == "neutral" and rows[1].strongest == "neutral"


def test_baseline_is_uniform_over_reference_text_means():
    short = _trace([_row("joy", 2.0)])
    long = _trace([_row("joy", 4.0), _row("joy", 4.0)])
    baseline = baseline_from_traces([short, long])
    assert baseline[EMOTIONS.index("joy")] == 3.0


def test_parse_mood_request_defaults_and_sorts_layers():
    default = parse_mood_request({"text": "A quiet room."})
    assert default.layers == DEFAULT_MOOD_LAYERS
    assert default.chunk_tokens == DEFAULT_CHUNK_TOKENS
    custom = parse_mood_request(
        {"text": "A quiet room.", "layers": [50, 13], "chunk_tokens": 64}
    )
    assert custom.layers == (13, 50)
    assert custom.to_payload()["layers"] == [13, 50]


@pytest.mark.parametrize(
    "body,message",
    [
        ({"text": " "}, "non-empty"),
        ({"text": "x", "layers": [51]}, "0-50"),
        ({"text": "x", "layers": [13, 13]}, "duplicates"),
        ({"text": "x", "chunk_tokens": 0}, "chunk_tokens"),
        ({"text": "x", "extra": True}, "unknown"),
    ],
)
def test_parse_mood_request_rejects_invalid_values(body, message):
    with pytest.raises(MoodRequestError, match=message):
        parse_mood_request(body)


def _api_response():
    shares = {emotion: 0.05 for emotion in EMOTIONS}
    shares["fear"] = 0.65
    return {
        "schema_version": "nemotron-mood/v1",
        "status": "complete",
        "mood": "fear",
        "strongest": "fear",
        "intensity": 3.2,
        "threshold": 2.0,
        "shares": shares,
        "sentences": [],
        "tokens": [
            {
                "position": 0,
                "id": 42,
                "text": " danger",
                "mood": "fear",
                "strongest": "fear",
                "intensity": 3.2,
                "shares": shares,
            }
        ],
    }


def test_render_response_includes_tokens_and_sanitizes_controls():
    response = _api_response()
    response["tokens"][0]["text"] = " danger\x1b]0;bad\x07"
    rendered = render_mood_response(response, include_tokens=True)
    assert "mood: fear" in rendered
    assert "danger" in rendered
    assert "\x1b" not in rendered
    assert "\x1b" not in sanitize("bad\x1b text")


def test_mood_endpoint_accepts_base_or_full_endpoint():
    assert cli.mood_endpoint("http://127.0.0.1:8000") == (
        "http://127.0.0.1:8000/api/mood"
    )
    assert cli.mood_endpoint("https://example.test/api/mood/") == (
        "https://example.test/api/mood"
    )
    with pytest.raises(cli.MoodClientError, match="absolute"):
        cli.mood_endpoint("localhost:8000")


def test_request_mood_posts_json_with_stdlib(monkeypatch):
    captured = {}

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self):
            return json.dumps(_api_response()).encode()

    def fake_open(request, *, timeout):
        captured["url"] = request.full_url
        captured["body"] = json.loads(request.data)
        captured["timeout"] = timeout
        captured["authorization"] = request.get_header("Authorization")
        return Response()

    monkeypatch.setattr(cli.urllib.request, "urlopen", fake_open)
    request = parse_mood_request({"text": "Test", "layers": [13]})
    response = cli.request_mood("http://localhost:8000", request, timeout=12.0)
    assert captured == {
        "url": "http://localhost:8000/api/mood",
        "body": {"text": "Test", "layers": [13], "chunk_tokens": 128},
        "timeout": 12.0,
        "authorization": None,
    }
    assert response["mood"] == "fear"


def test_cli_json_mode_reads_stdin(monkeypatch, capsys):
    monkeypatch.setattr(cli.sys, "stdin", io.StringIO("A sudden noise."))
    monkeypatch.setattr(cli, "request_mood", lambda *_args, **_kwargs: _api_response())
    assert cli.main(["--json", "--layers", "13,50"]) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["status"] == "complete"


def test_cli_uses_pretty_renderer_automatically_on_a_terminal(monkeypatch):
    import nemotron_mood.tui as tui

    observed = {}
    monkeypatch.setattr(cli.sys.stdout, "isatty", lambda: True)
    monkeypatch.setattr(cli, "request_mood", lambda *_args, **_kwargs: _api_response())

    def render(response, *, include_tokens, force_terminal):
        observed.update(
            response=response,
            include_tokens=include_tokens,
            force_terminal=force_terminal,
        )

    monkeypatch.setattr(tui, "render_mood_tui", render)

    assert cli.main(["A sudden noise.", "--tokens"]) == 0
    assert observed["response"]["mood"] == "fear"
    assert observed["include_tokens"] is True
    assert observed["force_terminal"] is None


def test_cli_pretty_forces_color_when_stdout_is_not_a_terminal(monkeypatch):
    import nemotron_mood.tui as tui

    observed = {}
    monkeypatch.setattr(cli.sys.stdout, "isatty", lambda: False)
    monkeypatch.setattr(cli, "request_mood", lambda *_args, **_kwargs: _api_response())
    monkeypatch.setattr(
        tui,
        "render_mood_tui",
        lambda _response, **kwargs: observed.update(kwargs),
    )

    assert cli.main(["A sudden noise.", "--pretty"]) == 0
    assert observed == {"include_tokens": False, "force_terminal": True}


def test_cli_output_modes_are_mutually_exclusive():
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(["text", "--json", "--pretty"])


def test_nano_mood_analyzer_uses_raw_ids_selected_logits_and_cached_calibration(
    monkeypatch,
):
    backend = make_tiny_backend()
    anchor_words = {
        emotion: (f"axis{index}",)
        for index, emotion in enumerate(EMOTIONS)
    }
    anchor_ids = {
        f" {word[0]}": index + 2
        for index, word in enumerate(anchor_words.values())
    }
    anchor_ids.update(
        {
            f" {word[0].capitalize()}": index + 2
            for index, word in enumerate(anchor_words.values())
        }
    )
    original_encode = backend.tokenizer.encode

    def encode(text, *, add_special_tokens=False):
        assert add_special_tokens is False
        if text in anchor_ids:
            return [anchor_ids[text]]
        return original_encode(text, add_special_tokens=add_special_tokens)

    monkeypatch.setattr(mood_backend, "EMOTION_ANCHORS", anchor_words)
    monkeypatch.setattr(backend.tokenizer, "encode", encode)
    analyzer = mood_backend.MoodAnalyzer(backend)
    request = mood_backend.MoodRequest(
        text="alpha beta", layers=(0, 1), chunk_tokens=8
    )

    first = analyzer.analyze(request)
    calls_after_first = backend.backbone.call_count
    second = analyzer.analyze(request)

    assert [token["id"] for token in first["tokens"]] == [2, 3]
    assert first["provenance"]["prompt_format"] == "raw"
    assert first["provenance"]["chat_template_applied"] is False
    assert first["layers"] == [0, 1]
    assert first["calibration"]["reference_count"] == 5
    assert first["diagnostics"]["hooks_before"] == 0
    assert first["diagnostics"]["hooks_after"] == 0
    assert first["diagnostics"]["selected_logit_parity_max_abs"] == pytest.approx(0)
    assert backend.backbone.call_count == calls_after_first + 1
    assert first["calibration"] == second["calibration"]
