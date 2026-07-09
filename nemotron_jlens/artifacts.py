"""Lens artifact validation, metadata, and exact prompt-shard merging."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

import torch

from jlens import JacobianLens
from nemotron_jlens.config import (
    ACCEPTANCE_SCHEMA,
    ARTIFACT_STORAGE_DTYPE,
    DEFAULT_DATASET_CONFIG,
    DEFAULT_DATASET_ID,
    DEFAULT_DATASET_REVISION,
    DEFAULT_DATASET_SPLIT,
    NEMOTRON,
    PAPER_SCALE_CORPUS_MANIFEST_SHA256,
    PAPER_SCALE_N_PROMPTS,
    PILOT_CORPUS_MANIFEST_SHA256,
    PILOT_N_PROMPTS,
    REPRODUCTION_FIT_SETTINGS,
    REQUIRED_TRANSFORMERS_VERSION,
)
from nemotron_jlens.corpus import sha256_file
from nemotron_jlens.provenance import require_sha256
from nemotron_jlens.runtime import validate_runtime_identity


def metadata_path(lens_path: str | Path) -> Path:
    return Path(f"{lens_path}.meta.json")


def write_metadata(lens_path: str | Path, metadata: dict[str, Any]) -> Path:
    lens_path = Path(lens_path)
    metadata = dict(metadata)
    metadata["artifact_sha256"] = sha256_file(lens_path)
    metadata.setdefault("created_at", datetime.now(timezone.utc).isoformat())
    path = metadata_path(lens_path)
    temp = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    temp.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temp.replace(path)
    return path


def read_metadata(lens_path: str | Path, *, verify_hash: bool = True) -> dict[str, Any]:
    metadata = json.loads(metadata_path(lens_path).read_text(encoding="utf-8"))
    if verify_hash:
        expected = metadata.get("artifact_sha256")
        if not isinstance(expected, str) or not expected:
            raise ValueError(f"metadata for {lens_path} has no artifact_sha256")
        actual = sha256_file(lens_path)
        if actual != expected:
            raise ValueError(
                f"artifact checksum mismatch for {lens_path}: {actual} != {expected}"
            )
        expected_storage = metadata.get("storage_dtype")
        if expected_storage != ARTIFACT_STORAGE_DTYPE:
            raise ValueError(
                "artifact metadata does not declare the required serialized "
                f"storage dtype {ARTIFACT_STORAGE_DTYPE!r}"
            )
        observed_storage = serialized_storage_dtype(lens_path)
        if observed_storage != expected_storage:
            raise ValueError(
                f"serialized storage dtype mismatch for {lens_path}: "
                f"{observed_storage!r} != {expected_storage!r}"
            )
    return metadata


def classify_reproduction(metadata: Mapping[str, Any]) -> dict[str, Any]:
    """Classify a valid artifact against the public reproduction contract.

    Pilot and paper-scale acceptance are exact allowlists, not prompt-count
    heuristics.  Any other scientifically valid artifact remains exportable as
    a non-final smoke result.
    """
    architecture = metadata.get("architecture")
    pinned_model = (
        metadata.get("model_id") == NEMOTRON.model_id
        and metadata.get("model_revision") == NEMOTRON.revision
        and architecture == NEMOTRON.to_dict()
    )

    dataset = metadata.get("dataset")
    pinned_dataset_values = {
        "id": DEFAULT_DATASET_ID,
        "config": DEFAULT_DATASET_CONFIG,
        "revision": DEFAULT_DATASET_REVISION,
        "split": DEFAULT_DATASET_SPLIT,
    }
    pinned_dataset = isinstance(dataset, Mapping) and all(
        dataset.get(key) == value for key, value in pinned_dataset_values.items()
    )

    manifest = metadata.get("corpus_manifest_sha256")
    profile_spec: tuple[str, int] | None = None
    if manifest == PAPER_SCALE_CORPUS_MANIFEST_SHA256:
        profile_spec = ("paper-scale", PAPER_SCALE_N_PROMPTS)
    elif manifest == PILOT_CORPUS_MANIFEST_SHA256:
        profile_spec = ("pilot", PILOT_N_PROMPTS)

    prompt_indices = metadata.get("prompt_indices")
    complete_prompt_set = (
        profile_spec is not None
        and metadata.get("n_prompts") == profile_spec[1]
        and prompt_indices == list(range(profile_spec[1]))
    )
    full_layer_coverage = (
        metadata.get("target_layer") == NEMOTRON.n_layers - 1
        and metadata.get("source_layers") == list(range(NEMOTRON.n_layers - 1))
    )
    fixed_fit_settings = all(
        metadata.get(key) == value
        for key, value in REPRODUCTION_FIT_SETTINGS.items()
    )
    runtime = metadata.get("runtime")
    packages = runtime.get("packages") if isinstance(runtime, Mapping) else None
    pinned_transformers = (
        isinstance(packages, Mapping)
        and packages.get("transformers") == REQUIRED_TRANSFORMERS_VERSION
    )
    fp32_storage = metadata.get("storage_dtype") == ARTIFACT_STORAGE_DTYPE
    checks = {
        "pinned_model": pinned_model,
        "pinned_dataset": pinned_dataset,
        "known_manifest": profile_spec is not None,
        "complete_prompt_set": complete_prompt_set,
        "full_layer_coverage": full_layer_coverage,
        "fixed_fit_settings": fixed_fit_settings,
        "pinned_transformers": pinned_transformers,
        "fp32_storage": fp32_storage,
    }

    all_checks_pass = all(checks.values())
    if all_checks_pass:
        assert profile_spec is not None
        tier = profile_spec[0]
    else:
        tier = "smoke"

    reasons: list[str] = []
    if tier == "pilot":
        reasons.append(
            "Pilot acceptance is non-final; paper-scale acceptance requires the "
            "canonical 1,000-prompt manifest."
        )
    elif tier == "smoke":
        failure_reasons = {
            "pinned_model": "The artifact does not use the exact pinned Nemotron model.",
            "pinned_dataset": "The artifact does not use the exact pinned dataset.",
            "known_manifest": (
                "The corpus is not the canonical 100-prompt or 1,000-prompt manifest."
            ),
            "complete_prompt_set": (
                "The artifact does not contain the complete canonical prompt index set."
            ),
            "full_layer_coverage": (
                "The artifact does not fit source layers 0..50 to target layer 51."
            ),
            "fixed_fit_settings": (
                "The artifact does not use every fixed reproduction fit setting."
            ),
            "pinned_transformers": (
                "The artifact was not fitted with exact Transformers "
                f"{REQUIRED_TRANSFORMERS_VERSION}."
            ),
            "fp32_storage": "The fitted means are not serialized in float32.",
        }
        reasons.extend(
            message for key, message in failure_reasons.items() if not checks[key]
        )

    is_final = tier == "paper-scale"
    return {
        "schema": ACCEPTANCE_SCHEMA,
        "tier": tier,
        "status": "accepted" if tier in {"pilot", "paper-scale"} else "non-final",
        "is_final": is_final,
        "exportable": True,
        "checks": checks,
        "reasons": reasons,
    }


def validate_lens(
    lens: JacobianLens,
    *,
    expected_d_model: int | None = None,
    expected_layers: Sequence[int] | None = None,
    expected_dtype: torch.dtype | None = None,
) -> dict[str, Any]:
    """Fail closed on malformed, non-finite, or suspicious lens tensors."""
    if expected_d_model is not None and lens.d_model != expected_d_model:
        raise ValueError(f"d_model={lens.d_model}, expected {expected_d_model}")
    if expected_layers is not None and lens.source_layers != list(expected_layers):
        raise ValueError(
            f"source_layers={lens.source_layers}, expected {list(expected_layers)}"
        )
    if lens.n_prompts <= 0:
        raise ValueError("lens has no fitted prompts")
    if not lens.source_layers:
        raise ValueError("lens has no source layers")

    summaries: dict[str, Any] = {}
    for layer in lens.source_layers:
        matrix = lens.jacobians[layer]
        if matrix.shape != (lens.d_model, lens.d_model):
            raise ValueError(f"layer {layer} has invalid shape {tuple(matrix.shape)}")
        if expected_dtype is not None and matrix.dtype != expected_dtype:
            raise ValueError(
                f"layer {layer} has dtype {matrix.dtype}, expected {expected_dtype}"
            )
        if not torch.isfinite(matrix).all():
            raise ValueError(f"layer {layer} contains NaN or Inf")
        frobenius = float(matrix.norm().item())
        if frobenius == 0 or not math.isfinite(frobenius):
            raise ValueError(f"layer {layer} has invalid norm {frobenius}")
        identity_distance_sq = max(
            0.0,
            frobenius**2
            + lens.d_model
            - 2 * float(matrix.diagonal().sum().item()),
        )
        summaries[str(layer)] = {
            "frobenius_norm": frobenius,
            "identity_distance_over_sqrt_d": math.sqrt(
                identity_distance_sq / lens.d_model
            ),
        }
    return {
        "ok": True,
        "d_model": lens.d_model,
        "n_prompts": lens.n_prompts,
        "source_layers": lens.source_layers,
        "dtype": str(next(iter(lens.jacobians.values())).dtype).removeprefix("torch."),
        "layers": summaries,
    }


def serialized_storage_dtype(lens_path: str | Path) -> str:
    """Return the on-disk tensor dtype, rejecting mixed or malformed payloads."""
    payload = torch.load(
        lens_path,
        map_location="cpu",
        weights_only=True,
        mmap=True,
    )
    jacobians = payload.get("J") if isinstance(payload, Mapping) else None
    if not isinstance(jacobians, Mapping) or not jacobians:
        raise ValueError(f"{lens_path} is not a serialized Jacobian lens")
    if not all(isinstance(matrix, torch.Tensor) for matrix in jacobians.values()):
        raise ValueError(f"{lens_path} contains a non-tensor Jacobian matrix")
    dtypes = {matrix.dtype for matrix in jacobians.values()}
    if len(dtypes) != 1:
        readable = sorted(str(dtype) for dtype in dtypes)
        raise ValueError(f"{lens_path} has mixed serialized storage dtypes: {readable}")
    dtype = dtypes.pop()
    names = {torch.float32: "float32"}
    if dtype not in names:
        raise ValueError(f"unsupported serialized storage dtype in {lens_path}: {dtype}")
    return names[dtype]


def load_validated_artifact(
    lens_path: str | Path,
    metadata: Mapping[str, Any],
    *,
    expected_kind: str | None = None,
) -> tuple[JacobianLens, dict[str, Any]]:
    """Reload an artifact and validate both its payload and serialized dtype."""
    expected_storage = metadata.get("storage_dtype")
    observed_storage = serialized_storage_dtype(lens_path)
    if observed_storage != expected_storage:
        raise ValueError(
            f"serialized storage dtype mismatch for {lens_path}: "
            f"{observed_storage!r} != {expected_storage!r}"
        )
    lens = JacobianLens.load(str(lens_path))
    summary = validate_artifact(lens, metadata, expected_kind=expected_kind)
    return lens, summary


def save_validated_artifact(
    lens: JacobianLens,
    lens_path: str | Path,
    metadata: Mapping[str, Any],
    *,
    expected_kind: str,
) -> dict[str, Any]:
    """Serialize, reload, validate, then atomically publish an artifact.

    The final path and checksum sidecar are untouched if serialization or
    post-serialization validation fails.
    """
    output = Path(lens_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temp = output.with_name(f".{output.name}.{uuid4().hex}.tmp")
    saved_metadata = dict(metadata)
    storage_dtype = saved_metadata.get("storage_dtype")
    if storage_dtype != ARTIFACT_STORAGE_DTYPE:
        raise ValueError(
            f"artifact storage_dtype must be {ARTIFACT_STORAGE_DTYPE!r}; "
            f"got {storage_dtype!r}"
        )
    try:
        # A binary file object makes PyTorch use the stable ``archive/`` ZIP
        # member root instead of deriving it from our randomized temporary
        # filename. Canonically merged tensors therefore produce identical
        # artifact bytes and checksums for every input-path permutation.
        with temp.open("wb") as file:
            torch.save(
                {
                    "J": {
                        layer: matrix.to(torch.float32)
                        for layer, matrix in lens.jacobians.items()
                    },
                    "n_prompts": lens.n_prompts,
                    "source_layers": lens.source_layers,
                    "d_model": lens.d_model,
                },
                file,
            )
        _, summary = load_validated_artifact(
            temp, saved_metadata, expected_kind=expected_kind
        )
        saved_metadata["validation"] = summary
        temp.replace(output)
    finally:
        temp.unlink(missing_ok=True)
    write_metadata(output, saved_metadata)
    return read_metadata(output)


_REQUIRED_PROVENANCE_STRINGS = (
    "model_id",
    "model_revision",
    "upstream_jlens_commit",
    "corpus_manifest_sha256",
    "adaptation_source_sha256",
)


def _integer(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _require_mapping(metadata: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = metadata.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"artifact metadata {key!r} must be an object")
    return value


def _validate_provenance(metadata: Mapping[str, Any]) -> None:
    missing = [
        key
        for key in _REQUIRED_PROVENANCE_STRINGS
        if not isinstance(metadata.get(key), str) or not metadata[key]
    ]
    if missing:
        raise ValueError(f"artifact metadata is missing provenance fields: {missing}")
    require_sha256(
        metadata["adaptation_source_sha256"],
        field="artifact metadata adaptation_source_sha256",
    )
    require_sha256(
        metadata["corpus_manifest_sha256"],
        field="artifact metadata corpus_manifest_sha256",
    )
    runtime = validate_runtime_identity(
        metadata.get("runtime"), field="artifact metadata runtime"
    )
    if metadata.get("kind") == "shard":
        environment = validate_runtime_identity(
            metadata.get("environment"), field="artifact metadata environment"
        )
        if environment != runtime:
            raise ValueError(
                "artifact metadata environment disagrees with runtime identity"
            )

    dataset = _require_mapping(metadata, "dataset")
    missing_dataset = [
        key
        for key in ("id", "revision")
        if not isinstance(dataset.get(key), str) or not dataset[key]
    ]
    if missing_dataset:
        raise ValueError(
            f"artifact metadata dataset is missing fields: {missing_dataset}"
        )

    architecture = _require_mapping(metadata, "architecture")
    for key in ("n_layers", "d_model"):
        if not _integer(architecture.get(key)) or architecture[key] <= 0:
            raise ValueError(f"artifact metadata architecture.{key} must be positive")
    for architecture_key, metadata_key in (
        ("model_id", "model_id"),
        ("revision", "model_revision"),
    ):
        if (
            architecture_key in architecture
            and architecture[architecture_key] != metadata[metadata_key]
        ):
            raise ValueError(
                f"artifact metadata {metadata_key} disagrees with "
                f"architecture.{architecture_key}"
            )


def _validate_shard_coordinates(metadata: Mapping[str, Any]) -> None:
    shard_index = metadata.get("shard_index")
    num_shards = metadata.get("num_shards")
    if (
        not _integer(shard_index)
        or not _integer(num_shards)
        or num_shards <= 0
        or not 0 <= shard_index < num_shards
    ):
        raise ValueError(
            "invalid shard coordinates: need num_shards > 0 and "
            f"0 <= shard_index < num_shards; got {shard_index}/{num_shards}"
        )


def validate_artifact(
    lens: JacobianLens,
    metadata: Mapping[str, Any],
    *,
    expected_kind: str | None = None,
) -> dict[str, Any]:
    """Validate a lens and cross-check every identity-bearing sidecar field.

    This is deliberately stricter than :func:`validate_lens`: a numerically valid
    tensor is not a reproducible artifact unless its model, corpus, fit and prompt
    provenance agree with the serialized lens payload.
    """
    storage_dtype = metadata.get("storage_dtype")
    if storage_dtype != ARTIFACT_STORAGE_DTYPE:
        raise ValueError(
            f"artifact metadata storage_dtype must be {ARTIFACT_STORAGE_DTYPE!r}; "
            f"got {storage_dtype!r}"
        )
    summary = validate_lens(lens, expected_dtype=torch.float32)
    _validate_provenance(metadata)

    kind = metadata.get("kind")
    if kind not in {"shard", "merged"}:
        raise ValueError(f"unsupported artifact metadata kind: {kind!r}")
    if expected_kind is not None and kind != expected_kind:
        raise ValueError(f"expected {expected_kind!r} artifact, found {kind!r}")

    n_prompts = metadata.get("n_prompts")
    if not _integer(n_prompts) or n_prompts <= 0:
        raise ValueError("artifact metadata n_prompts must be a positive integer")
    if n_prompts != lens.n_prompts:
        raise ValueError(
            f"artifact metadata n_prompts={n_prompts} disagrees with "
            f"lens n_prompts={lens.n_prompts}"
        )

    source_layers = metadata.get("source_layers")
    if source_layers != lens.source_layers:
        raise ValueError(
            f"artifact metadata source_layers={source_layers} disagrees with "
            f"lens source_layers={lens.source_layers}"
        )

    prompt_hashes = metadata.get("prompt_hashes")
    if not isinstance(prompt_hashes, list) or not all(
        isinstance(value, str) and value for value in prompt_hashes
    ):
        raise ValueError("artifact metadata prompt_hashes must be non-empty strings")
    if len(prompt_hashes) != lens.n_prompts:
        raise ValueError(
            "artifact metadata prompt_hashes has the wrong prompt count: "
            f"{len(prompt_hashes)} != {lens.n_prompts}"
        )
    if len(set(prompt_hashes)) != len(prompt_hashes):
        raise ValueError("artifact metadata contains duplicate prompt hashes")
    for prompt_index, prompt_hash in enumerate(prompt_hashes):
        require_sha256(prompt_hash, field=f"artifact prompt_hashes[{prompt_index}]")

    prompt_indices = metadata.get("prompt_indices")
    if not isinstance(prompt_indices, list) or not all(
        _integer(index) and index >= 0 for index in prompt_indices
    ):
        raise ValueError("artifact metadata prompt_indices must be nonnegative integers")
    if len(prompt_indices) != lens.n_prompts:
        raise ValueError(
            "artifact metadata prompt_indices has the wrong prompt count: "
            f"{len(prompt_indices)} != {lens.n_prompts}"
        )
    if len(set(prompt_indices)) != len(prompt_indices):
        raise ValueError("artifact metadata contains duplicate prompt indices")
    if kind == "shard":
        _validate_shard_coordinates(metadata)
        if any(
            index % metadata["num_shards"] != metadata["shard_index"]
            for index in prompt_indices
        ):
            raise ValueError(
                "artifact metadata prompt_indices do not belong to "
                f"shard {metadata['shard_index']}/{metadata['num_shards']}"
            )
    elif sorted(prompt_indices) != list(range(lens.n_prompts)):
        raise ValueError("merged artifact prompt_indices must exactly cover its corpus")

    architecture = _require_mapping(metadata, "architecture")
    if architecture["d_model"] != lens.d_model:
        raise ValueError(
            f"artifact architecture d_model={architecture['d_model']} disagrees "
            f"with lens d_model={lens.d_model}"
        )
    target_layer = metadata.get("target_layer")
    if not _integer(target_layer):
        raise ValueError("artifact metadata target_layer must be an integer")
    if not 0 <= target_layer < architecture["n_layers"]:
        raise ValueError(
            f"target_layer={target_layer} is outside architecture with "
            f"{architecture['n_layers']} layers"
        )
    if lens.source_layers[0] < 0 or lens.source_layers[-1] >= target_layer:
        raise ValueError(
            f"source_layers must be in [0, {target_layer}); got "
            f"{lens.source_layers[0]}..{lens.source_layers[-1]}"
        )

    if kind == "shard":
        _validate_shard_coordinates(metadata)
    expected_acceptance = classify_reproduction(metadata)
    if metadata.get("acceptance") != expected_acceptance:
        raise ValueError(
            "artifact metadata acceptance classification is missing or incorrect: "
            f"expected {expected_acceptance!r}"
        )
    return summary


_MERGE_INVARIANTS = (
    "model_id",
    "model_revision",
    "upstream_jlens_commit",
    "adaptation_source_sha256",
    "architecture",
    "dataset",
    "corpus_manifest_sha256",
    "source_layers",
    "target_layer",
    "dim_batch",
    "max_seq_len",
    "skip_first",
    "dtype",
    "storage_dtype",
    "compile_blocks",
    "disable_mamba_kernels",
    "runtime",
)

_OPTIONAL_MERGE_INVARIANTS = (
    "mamba_backend",
    "patched_mamba_layers",
)

_MERGED_PROVENANCE_FIELDS = (
    *_MERGE_INVARIANTS,
    *_OPTIONAL_MERGE_INVARIANTS,
    "corpus_path",
)


def _check_merge_invariants(
    first: Mapping[str, Any],
    current: Mapping[str, Any],
    path: str | Path,
) -> None:
    sentinel = object()
    keys = list(_MERGE_INVARIANTS)
    keys.extend(
        key
        for key in _OPTIONAL_MERGE_INVARIANTS
        if key in first or key in current
    )
    drift = {
        key: (first.get(key, sentinel), current.get(key, sentinel))
        for key in keys
        if first.get(key, sentinel) != current.get(key, sentinel)
    }
    readable_drift = {
        key: (
            "<missing>" if left is sentinel else left,
            "<missing>" if right is sentinel else right,
        )
        for key, (left, right) in drift.items()
    }
    if readable_drift:
        raise ValueError(f"shard metadata mismatch in {path}: {readable_drift}")


def _validate_complete_partition(metadata: Sequence[Mapping[str, Any]]) -> int:
    for current in metadata:
        if current.get("kind") != "shard":
            raise ValueError("merge inputs must be direct shard artifacts")
        _validate_shard_coordinates(current)

    num_shards = metadata[0]["num_shards"]
    if any(current["num_shards"] != num_shards for current in metadata[1:]):
        raise ValueError("shard metadata mismatch: num_shards values disagree")
    shard_indices = [current["shard_index"] for current in metadata]
    if len(set(shard_indices)) != len(shard_indices):
        raise ValueError(f"duplicate shard indices: {shard_indices}")
    expected = set(range(num_shards))
    actual = set(shard_indices)
    if actual != expected:
        raise ValueError(
            "incomplete shard partition: "
            f"expected indices {sorted(expected)}, found {sorted(actual)}"
        )
    return num_shards


def merge_shards(inputs: Sequence[str | Path], output: str | Path) -> dict[str, Any]:
    """Merge a complete partition while holding at most one input lens in RAM."""
    if not inputs:
        raise ValueError("at least one input shard is required")
    paths = [Path(path) for path in inputs]
    metadata = [read_metadata(path) for path in paths]
    num_shards = _validate_complete_partition(metadata)
    ordered_inputs = sorted(
        zip(paths, metadata, strict=True),
        key=lambda item: item[1]["shard_index"],
    )
    paths = [path for path, _ in ordered_inputs]
    metadata = [current for _, current in ordered_inputs]
    first = metadata[0]
    _validate_provenance(first)
    for path, current in zip(paths[1:], metadata[1:], strict=True):
        _validate_provenance(current)
        _check_merge_invariants(first, current, path)

    prompt_hashes: list[str] = []
    indexed_hashes: list[tuple[int, str]] = []
    prompt_indices: list[int] = []
    index_presence = [
        isinstance(current.get("prompt_indices"), list) for current in metadata
    ]
    if any(index_presence) and not all(index_presence):
        raise ValueError("shard metadata mismatch: prompt_indices missing from a shard")
    every_shard_has_indices = all(index_presence)
    seen_hashes: set[str] = set()
    seen_indices: set[int] = set()
    weighted_sums: dict[int, torch.Tensor] | None = None
    d_model: int | None = None
    n_total = 0

    # Input lenses are intentionally loaded and released one at a time. On the
    # first input, its fp32 matrices become the accumulator in place.
    for path, current in zip(paths, metadata, strict=True):
        lens, _ = load_validated_artifact(path, current, expected_kind="shard")

        hashes = current["prompt_hashes"]
        overlap = seen_hashes.intersection(hashes)
        if overlap:
            raise ValueError(f"prompt shards overlap ({len(overlap)} duplicate hashes)")
        seen_hashes.update(hashes)
        prompt_hashes.extend(hashes)

        if every_shard_has_indices:
            indices = current["prompt_indices"]
            if len(indices) != lens.n_prompts or not all(
                _integer(index) and index >= 0 for index in indices
            ):
                raise ValueError(f"{path} metadata has invalid prompt_indices")
            overlap_indices = seen_indices.intersection(indices)
            if overlap_indices:
                raise ValueError(
                    "prompt shards overlap by corpus index "
                    f"({len(overlap_indices)} duplicates)"
                )
            if any(index % num_shards != current["shard_index"] for index in indices):
                raise ValueError(
                    f"{path} prompt_indices do not belong to shard "
                    f"{current['shard_index']}/{num_shards}"
                )
            seen_indices.update(indices)
            prompt_indices.extend(indices)
            indexed_hashes.extend(zip(indices, hashes, strict=True))

        if weighted_sums is None:
            weighted_sums = lens.jacobians
            d_model = lens.d_model
            for matrix in weighted_sums.values():
                matrix.mul_(lens.n_prompts)
            del matrix
        else:
            for layer in lens.source_layers:
                weighted_sums[layer].add_(
                    lens.jacobians[layer], alpha=lens.n_prompts
                )
        n_total += lens.n_prompts
        del lens

    assert weighted_sums is not None and d_model is not None
    if every_shard_has_indices and sorted(prompt_indices) != list(range(n_total)):
        raise ValueError(
            "incomplete prompt partition: combined prompt_indices are not contiguous"
        )
    for matrix in weighted_sums.values():
        matrix.div_(n_total)
    del matrix
    merged = JacobianLens(
        jacobians=weighted_sums,
        n_prompts=n_total,
        d_model=d_model,
    )

    ordered_hashes = (
        [value for _, value in sorted(indexed_hashes)]
        if every_shard_has_indices
        else sorted(prompt_hashes)
    )
    merged_metadata: dict[str, Any] = {
        key: first[key] for key in _MERGED_PROVENANCE_FIELDS if key in first
    }
    merged_metadata.update(
        {
            "schema_version": 1,
            "kind": "merged",
            "num_shards": num_shards,
            "n_prompts": merged.n_prompts,
            "prompt_hashes": ordered_hashes,
            "input_shards": [
                {
                    "path": str(path),
                    "sha256": current["artifact_sha256"],
                    "shard_index": current["shard_index"],
                    "num_shards": current["num_shards"],
                    "n_prompts": current["n_prompts"],
                    "prompt_indices": current.get("prompt_indices"),
                    "prompt_hashes": current["prompt_hashes"],
                    "created_at": current.get("created_at"),
                    "device_map": current.get("device_map"),
                    "environment": current.get("environment"),
                    "runtime": current["runtime"],
                    "validation": current.get("validation"),
                }
                for path, current in zip(paths, metadata, strict=True)
            ],
        }
    )
    if every_shard_has_indices:
        merged_metadata["prompt_indices"] = sorted(prompt_indices)
    merged_metadata["acceptance"] = classify_reproduction(merged_metadata)

    output_path = Path(output)
    return save_validated_artifact(
        merged,
        output_path,
        merged_metadata,
        expected_kind="merged",
    )
