"""Fail-closed scientific runtime identity for Nemotron model execution."""

from __future__ import annotations

import importlib.metadata
import platform
from collections.abc import Mapping
from typing import Any

import torch

from nemotron_jlens.config import (
    REQUIRED_TRANSFORMERS_VERSION,
    RUNTIME_DISTRIBUTIONS,
)


def _distribution_version(distribution: str) -> str:
    try:
        version = importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError as exc:
        raise RuntimeError(
            f"required Nemotron runtime distribution {distribution!r} is not installed"
        ) from exc
    if not version:
        raise RuntimeError(
            f"required Nemotron runtime distribution {distribution!r} has no version"
        )
    return version


def require_pinned_transformers() -> str:
    """Reject any Transformers release other than the vendor-tested pin."""
    observed = _distribution_version("transformers")
    if observed != REQUIRED_TRANSFORMERS_VERSION:
        raise RuntimeError(
            "Nemotron 3 Nano requires exact Transformers "
            f"{REQUIRED_TRANSFORMERS_VERSION}; found {observed}"
        )
    return observed


def runtime_identity() -> dict[str, Any]:
    """Return the complete software identity that all fit shards must share."""
    require_pinned_transformers()
    packages = {
        distribution: _distribution_version(distribution)
        for distribution in RUNTIME_DISTRIBUTIONS
    }
    cuda_runtime = torch.version.cuda
    if not isinstance(cuda_runtime, str) or not cuda_runtime:
        raise RuntimeError("the Nemotron scientific runtime requires a CUDA build of torch")
    return {
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "cuda_runtime": cuda_runtime,
        "packages": packages,
    }


def validate_runtime_identity(value: object, *, field: str) -> dict[str, Any]:
    """Validate a serialized runtime identity and its exact Transformers pin."""
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be an object")
    for key in ("python", "torch", "cuda_runtime"):
        if not isinstance(value.get(key), str) or not value[key]:
            raise ValueError(f"{field}.{key} must be a non-empty string")
    packages = value.get("packages")
    if not isinstance(packages, Mapping):
        raise ValueError(f"{field}.packages must be an object")
    for distribution in RUNTIME_DISTRIBUTIONS:
        version = packages.get(distribution)
        if not isinstance(version, str) or not version:
            raise ValueError(
                f"{field}.packages.{distribution} must be a non-empty string"
            )
    observed_transformers = packages["transformers"]
    if observed_transformers != REQUIRED_TRANSFORMERS_VERSION:
        raise ValueError(
            f"{field}.packages.transformers must be exact version "
            f"{REQUIRED_TRANSFORMERS_VERSION}; found {observed_transformers}"
        )
    return {
        "python": value["python"],
        "torch": value["torch"],
        "cuda_runtime": value["cuda_runtime"],
        "packages": dict(packages),
    }
