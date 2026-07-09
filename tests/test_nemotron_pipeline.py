from types import SimpleNamespace

import torch

import nemotron_jlens.pipeline as pipeline
from jlens import JacobianLens
from nemotron_jlens.config import (
    ARTIFACT_STORAGE_DTYPE,
    NEMOTRON,
    REQUIRED_TRANSFORMERS_VERSION,
)
from nemotron_jlens.corpus import CorpusRecord, sha256_text


def test_fit_shard_routes_fp32_artifact_through_post_save_validation(
    tmp_path, monkeypatch
):
    text = "A deterministic test prompt."
    record = CorpusRecord(index=0, text=text, sha256=sha256_text(text))
    corpus_metadata = {
        "manifest_sha256": sha256_text("smoke-manifest"),
        "dataset": {
            "id": "Salesforce/wikitext",
            "config": "wikitext-103-raw-v1",
            "revision": "b08601e04326c79dfdd32d625aee71d232d685c3",
            "split": "train",
            "text_field": "text",
        },
    }
    monkeypatch.setattr(
        pipeline, "load_corpus", lambda _: ([record], corpus_metadata)
    )
    monkeypatch.setattr(
        pipeline,
        "load_nemotron",
        lambda **_: SimpleNamespace(
            model_id=NEMOTRON.model_id,
            revision=NEMOTRON.revision,
            lens_model=object(),
            mamba_backend="fused-or-auto",
            patched_mamba_layers=0,
            runtime_identity={
                "python": "3.12.0",
                "torch": "2.9.0+cu130",
                "cuda_runtime": "13.0",
                "packages": {
                    "transformers": REQUIRED_TRANSFORMERS_VERSION,
                    "mamba-ssm": "2.2.5",
                    "causal-conv1d": "1.5.3",
                    "accelerate": "1.12.0",
                    "datasets": "3.1.0",
                    "huggingface-hub": "0.36.0",
                },
            },
        ),
    )
    lens = JacobianLens(
        jacobians={0: torch.eye(3)},
        n_prompts=1,
        d_model=3,
    )
    monkeypatch.setattr(pipeline.jlens, "fit", lambda *_, **__: lens)
    monkeypatch.setattr(pipeline, "validate_lens", lambda *_, **__: {"ok": True})

    captured = {}

    def fake_save(fitted, output, metadata, *, expected_kind):
        captured.update(
            fitted=fitted,
            output=output,
            metadata=metadata,
            expected_kind=expected_kind,
        )
        return {**metadata, "artifact_sha256": sha256_text("artifact")}

    monkeypatch.setattr(pipeline, "save_validated_artifact", fake_save)
    output = tmp_path / "shard.pt"
    result = pipeline.fit_shard(
        corpus_path=str(tmp_path / "corpus.jsonl"),
        output_path=str(output),
        source_layer_spec="0",
        device_map="cuda",
    )

    assert captured["fitted"] is lens
    assert captured["output"] == output
    assert captured["expected_kind"] == "shard"
    assert captured["metadata"]["storage_dtype"] == ARTIFACT_STORAGE_DTYPE
    assert (
        captured["metadata"]["runtime"]["packages"]["transformers"]
        == REQUIRED_TRANSFORMERS_VERSION
    )
    assert captured["metadata"]["environment"]["packages"] == captured["metadata"][
        "runtime"
    ]["packages"]
    assert captured["metadata"]["acceptance"]["tier"] == "smoke"
    assert captured["metadata"]["acceptance"]["status"] == "non-final"
    assert result["artifact_sha256"] == sha256_text("artifact")
