"""FastAPI boundary tests for exact steering token identities."""

from __future__ import annotations

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient

from nemotron_steering.server import create_app
from nemotron_steering.service import InferenceService
from tests.test_steering_backend import make_tiny_backend


def test_fastapi_preserves_numeric_ids_and_leading_space_tokens(tmp_path) -> None:
    static_dir = tmp_path / "steering_demo"
    static_dir.mkdir()
    (static_dir / "index.html").write_text("<!doctype html><title>test</title>")
    backend = make_tiny_backend()
    client = TestClient(create_app(InferenceService(backend), static_dir=static_dir))

    response = client.post(
        "/api/baseline",
        headers={"X-Request-ID": "token_ids_01"},
        json={
            "prompt": "alpha beta",
            "layers": [0],
            "top_k": 3,
            "max_new_tokens": 1,
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert [token["id"] for token in body["clean"]["tokens"][:3]] == [1, 2, 3]
    assert [token["text"] for token in body["clean"]["tokens"][:3]] == [
        "<BOS>",
        "alpha",
        " beta",
    ]
    assert body["provenance"]["formatted_prompt_token_ids"] == [1, 2, 3]
    assert body["provenance"]["formatted_prompt"] == "<BOS>alpha beta"

    tokenized = client.post("/api/tokenize", json={"text": " beta"})
    assert tokenized.status_code == 200
    assert tokenized.json()["token_ids"] == [3]
    assert tokenized.json()["pieces"] == [{"id": 3, "text": " beta"}]


def test_mood_endpoint_parses_raw_text_and_uses_shared_service(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    static_dir = tmp_path / "steering_demo"
    static_dir.mkdir()
    (static_dir / "index.html").write_text("<!doctype html><title>test</title>")
    service = InferenceService(make_tiny_backend())
    observed = {}

    def run_mood(request, *, request_id):
        observed["request"] = request
        observed["request_id"] = request_id
        return {
            "schema_version": "nemotron-jlens-mood/v1",
            "status": "complete",
            "mood": "neutral",
            "request_id": request_id,
        }

    monkeypatch.setattr(service, "run_mood", run_mood)
    client = TestClient(create_app(service, static_dir=static_dir))

    response = client.post(
        "/api/mood",
        headers={"X-Request-ID": "mood_raw_01"},
        json={"text": "The train arrived.", "layers": [13, 20], "chunk_tokens": 64},
    )

    assert response.status_code == 200
    assert response.json()["request_id"] == "mood_raw_01"
    assert observed["request_id"] == "mood_raw_01"
    assert observed["request"].text == "The train arrived."
    assert observed["request"].layers == (13, 20)
    assert observed["request"].chunk_tokens == 64


def test_mood_endpoint_rejects_unknown_fields(tmp_path) -> None:
    static_dir = tmp_path / "steering_demo"
    static_dir.mkdir()
    (static_dir / "index.html").write_text("<!doctype html><title>test</title>")
    client = TestClient(
        create_app(InferenceService(make_tiny_backend()), static_dir=static_dir)
    )

    response = client.post(
        "/api/mood", json={"text": "plain text", "authorization": "not-used"}
    )

    assert response.status_code == 422
    assert "unknown" in response.json()["error"].lower()
