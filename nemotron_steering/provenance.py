"""Fail-closed artifact/runtime validation and live application fingerprints."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from jlens import JacobianLens
from nemotron_jlens.artifacts import (
    load_validated_artifact,
    read_metadata,
)
from nemotron_jlens.provenance import adaptation_source_sha256
from nemotron_jlens.runtime import runtime_identity
from nemotron_steering.constants import (
    D_MODEL,
    DEFAULT_READOUT_LAYERS,
    FINAL_LAYER,
    FIT_SOURCE_SHA256,
    LENS_SHA256,
    MAX_INTERVENTION_LAYERS,
    MAX_READOUT_LAYERS,
    MODEL_ID,
    MODEL_REVISION,
    N_LAYERS,
    NEMO_IMAGE,
    NEURONPEDIA_COMMIT,
    PILOT_DISCLOSURE,
    SOURCE_LAYERS,
    VOCAB_SIZE,
)
from nemotron_steering.errors import ValidationError


@dataclass(frozen=True)
class ValidatedLens:
    lens: JacobianLens
    metadata: dict[str, Any]
    validation: dict[str, Any]
    application_source_sha256: str


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def application_source_sha256(root: str | Path | None = None) -> str:
    """Fingerprint the isolated live backend and browser implementation."""
    repository = (
        Path(root).resolve()
        if root is not None
        else Path(__file__).resolve().parents[1]
    )
    roots = [
        repository / "nemotron_mood",
        repository / "nemotron_steering",
        repository / "steering_demo",
    ]
    files = sorted(
        path
        for source_root in roots
        if source_root.exists()
        for path in source_root.rglob("*")
        if path.is_file()
        and "__pycache__" not in path.parts
        and path.suffix in {".py", ".html", ".css", ".js"}
    )
    if not files:
        raise ValidationError("live application source fingerprint has no files")
    digest = hashlib.sha256()
    for path in files:
        relative = path.relative_to(repository).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _require_equal(metadata: Mapping[str, Any], key: str, expected: object) -> None:
    observed = metadata.get(key)
    if observed != expected:
        raise ValidationError(
            f"lens sidecar {key} drifted: expected {expected!r}, found {observed!r}"
        )


def validate_lens_before_model_load(
    lens_path: str | Path,
    *,
    observed_runtime: Mapping[str, Any] | None = None,
    source_root: str | Path | None = None,
) -> ValidatedLens:
    """Validate bytes, sidecar, fit source, matrices, and exact runtime first."""
    path = Path(lens_path).resolve()
    if not path.is_file():
        raise ValidationError(f"lens file does not exist: {path}")
    actual_sha = sha256_file(path)
    if actual_sha != LENS_SHA256:
        raise ValidationError(
            f"lens SHA-256 mismatch: expected {LENS_SHA256}, found {actual_sha}"
        )
    try:
        metadata = read_metadata(path)
        lens, validation = load_validated_artifact(
            path, metadata, expected_kind="merged"
        )
    except (KeyError, OSError, RuntimeError, TypeError, ValueError) as exc:
        raise ValidationError(f"lens artifact validation failed: {exc}") from exc

    _require_equal(metadata, "artifact_sha256", LENS_SHA256)
    _require_equal(metadata, "model_id", MODEL_ID)
    _require_equal(metadata, "model_revision", MODEL_REVISION)
    _require_equal(metadata, "adaptation_source_sha256", FIT_SOURCE_SHA256)
    _require_equal(metadata, "source_layers", list(SOURCE_LAYERS))
    _require_equal(metadata, "target_layer", FINAL_LAYER)
    _require_equal(metadata, "dtype", "bfloat16")
    _require_equal(metadata, "storage_dtype", "float32")
    _require_equal(metadata, "mamba_backend", "fused-or-auto")
    _require_equal(metadata, "disable_mamba_kernels", False)
    _require_equal(metadata, "n_prompts", 100)

    architecture = metadata.get("architecture")
    expected_architecture = {
        "model_id": MODEL_ID,
        "revision": MODEL_REVISION,
        "model_type": "nemotron_h",
        "n_layers": N_LAYERS,
        "d_model": D_MODEL,
        "vocab_size": VOCAB_SIZE,
    }
    if not isinstance(architecture, Mapping):
        raise ValidationError("lens sidecar architecture must be an object")
    drift = {
        key: {"expected": expected, "observed": architecture.get(key)}
        for key, expected in expected_architecture.items()
        if architecture.get(key) != expected
    }
    if drift:
        raise ValidationError(f"lens architecture drifted: {drift}")

    acceptance = metadata.get("acceptance")
    if not isinstance(acceptance, Mapping) or (
        acceptance.get("tier"),
        acceptance.get("status"),
        acceptance.get("is_final"),
    ) != ("pilot", "accepted", False):
        raise ValidationError(
            "lens acceptance must be accepted pilot with is_final=false"
        )

    live_fit_hash = adaptation_source_sha256()
    if live_fit_hash != FIT_SOURCE_SHA256:
        raise ValidationError(
            "fit-source provenance changed: "
            f"expected {FIT_SOURCE_SHA256}, found {live_fit_hash}"
        )
    current_runtime = dict(observed_runtime or runtime_identity())
    sidecar_runtime = metadata.get("runtime")
    if current_runtime != sidecar_runtime:
        raise ValidationError(
            "scientific runtime differs from the fitted lens sidecar: "
            f"expected {sidecar_runtime!r}, found {current_runtime!r}"
        )
    return ValidatedLens(
        lens=lens,
        metadata=dict(metadata),
        validation=validation,
        application_source_sha256=application_source_sha256(source_root),
    )


def immutable_info(
    bundle: ValidatedLens, *, loaded: Any | None = None
) -> dict[str, Any]:
    metadata = bundle.metadata
    model = {
        "id": MODEL_ID,
        "revision": MODEL_REVISION,
        "dtype": "bfloat16",
        "n_layers": N_LAYERS,
        "d_model": D_MODEL,
        "vocab_size": VOCAB_SIZE,
    }
    if loaded is not None:
        model["mamba_backend"] = loaded.mamba_backend
        model["patched_mamba_layers"] = loaded.patched_mamba_layers
    return {
        "disclosure": PILOT_DISCLOSURE,
        "model": model,
        "lens": {
            "sha256": LENS_SHA256,
            "source_layers": list(SOURCE_LAYERS),
            "target_layer": FINAL_LAYER,
            "storage_dtype": "float32",
            "acceptance": metadata["acceptance"],
            "prompt_count": metadata["n_prompts"],
            "fit_source_sha256": metadata["adaptation_source_sha256"],
        },
        "runtime": metadata["runtime"],
        "container": NEMO_IMAGE,
        "neuronpedia_reference_commit": NEURONPEDIA_COMMIT,
        "chat_template": {
            "source": "pinned-tokenizer",
            "add_generation_prompt": True,
            "default_enable_thinking": False,
        },
        "layer_policy": {
            "readout": {
                "min": 0,
                "max": N_LAYERS - 1,
                "max_selected": MAX_READOUT_LAYERS,
                "default": list(DEFAULT_READOUT_LAYERS),
            },
            "intervention": {
                "min": SOURCE_LAYERS[0],
                "max": SOURCE_LAYERS[-1],
                "max_selected": MAX_INTERVENTION_LAYERS,
                "read_only": [FINAL_LAYER],
            },
        },
        "live_application_source_sha256": bundle.application_source_sha256,
    }
