from __future__ import annotations

import importlib.metadata

import pytest
import torch

from nemotron_jlens.config import (
    REQUIRED_TRANSFORMERS_VERSION,
    RUNTIME_DISTRIBUTIONS,
)
from nemotron_jlens.loading import load_nemotron
from nemotron_jlens.runtime import (
    require_pinned_transformers,
    runtime_identity,
    validate_runtime_identity,
)


def _versions(*, transformers: str = REQUIRED_TRANSFORMERS_VERSION) -> dict[str, str]:
    return {
        "transformers": transformers,
        "mamba-ssm": "2.2.5",
        "causal-conv1d": "1.5.3",
        "accelerate": "1.12.0",
        "datasets": "3.1.0",
        "huggingface-hub": "0.36.0",
    }


def test_load_rejects_transformers_drift_before_cuda_or_model_code(monkeypatch):
    monkeypatch.setattr(
        "nemotron_jlens.runtime.importlib.metadata.version",
        lambda _: "4.58.0",
    )
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    with pytest.raises(RuntimeError, match="exact Transformers 4.57.3.*4.58.0"):
        load_nemotron()


def test_runtime_identity_records_every_scientific_distribution(monkeypatch):
    versions = _versions()
    monkeypatch.setattr(
        "nemotron_jlens.runtime.importlib.metadata.version",
        versions.__getitem__,
    )
    monkeypatch.setattr(torch.version, "cuda", "13.0")

    identity = runtime_identity()

    assert identity["packages"] == versions
    assert tuple(identity["packages"]) == RUNTIME_DISTRIBUTIONS
    assert identity["cuda_runtime"] == "13.0"
    assert identity["python"]
    assert identity["torch"]
    assert validate_runtime_identity(identity, field="test runtime") == identity


def test_runtime_identity_fails_closed_on_missing_distribution(monkeypatch):
    versions = _versions()

    def version(distribution: str) -> str:
        if distribution == "mamba-ssm":
            raise importlib.metadata.PackageNotFoundError(distribution)
        return versions[distribution]

    monkeypatch.setattr(
        "nemotron_jlens.runtime.importlib.metadata.version",
        version,
    )
    monkeypatch.setattr(torch.version, "cuda", "13.0")

    with pytest.raises(RuntimeError, match="mamba-ssm.*not installed"):
        runtime_identity()


def test_runtime_validator_enforces_exact_transformers():
    value = {
        "python": "3.12.0",
        "torch": "2.9.0+cu130",
        "cuda_runtime": "13.0",
        "packages": _versions(transformers="4.57.2"),
    }
    with pytest.raises(ValueError, match="exact version 4.57.3"):
        validate_runtime_identity(value, field="test runtime")

    assert require_pinned_transformers() == REQUIRED_TRANSFORMERS_VERSION
