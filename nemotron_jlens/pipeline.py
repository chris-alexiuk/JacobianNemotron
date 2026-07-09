"""High-level fit, merge, validation, and static-demo operations."""

from __future__ import annotations

import json
import os
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

import jlens
from jlens.examples import EXAMPLES, resolve_prompt
from jlens.vis import build_page, compute_slice
from nemotron_jlens.artifacts import (
    classify_reproduction,
    metadata_path,
    read_metadata,
    save_validated_artifact,
    validate_artifact,
    validate_lens,
)
from nemotron_jlens.config import (
    ARTIFACT_STORAGE_DTYPE,
    DEFAULT_DIM_BATCH,
    DEFAULT_MAX_SEQ_LEN,
    DEFAULT_SKIP_FIRST,
    NEMOTRON,
    UPSTREAM_JLENS_COMMIT,
)
from nemotron_jlens.corpus import load_corpus, select_shard, sha256_text
from nemotron_jlens.loading import load_nemotron
from nemotron_jlens.provenance import adaptation_source_sha256
from nemotron_jlens.runtime import runtime_identity


def resolve_layer(index: int, n_layers: int) -> int:
    resolved = index + n_layers if index < 0 else index
    if not 0 <= resolved < n_layers:
        raise ValueError(f"layer {index} is out of range for {n_layers} layers")
    return resolved


def parse_source_layers(spec: str, *, n_layers: int, target_layer: int) -> list[int]:
    """Parse ``all``, comma-separated indices, or Python-style ranges.

    Examples: ``all``, ``0,8,16,24``, ``0:51:4``. Negative indices are
    resolved relative to ``n_layers``.
    """
    if spec.strip().lower() == "all":
        return list(range(target_layer))
    layers: list[int] = []
    for raw_part in spec.split(","):
        part = raw_part.strip()
        if not part:
            continue
        if ":" in part:
            fields = part.split(":")
            if not 2 <= len(fields) <= 3:
                raise ValueError(f"invalid layer range {part!r}")
            start = int(fields[0]) if fields[0] else 0
            stop = int(fields[1]) if fields[1] else target_layer
            step = int(fields[2]) if len(fields) == 3 and fields[2] else 1
            if start < 0:
                start += n_layers
            if stop < 0:
                stop += n_layers
            layers.extend(range(start, stop, step))
        else:
            layers.append(resolve_layer(int(part), n_layers))
    layers = sorted(set(layers))
    if not layers:
        raise ValueError("source layer selection is empty")
    if layers[0] < 0 or layers[-1] >= target_layer:
        raise ValueError(
            f"source layers must be in [0, {target_layer}); got "
            f"{layers[0]}..{layers[-1]}"
        )
    return layers


def environment_metadata(
    scientific_runtime: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Record scientific software separately from hardware placement."""
    if scientific_runtime is None:
        scientific_runtime = runtime_identity()
    cuda_devices = []
    if torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(index)
            cuda_devices.append(
                {
                    "index": index,
                    "name": props.name,
                    "total_memory_gib": props.total_memory / 1024**3,
                }
            )
    return {
        **scientific_runtime,
        "platform": platform.platform(),
        "cuda_devices": cuda_devices,
    }


def _checkpoint_sidecar(checkpoint: Path) -> Path:
    return Path(f"{checkpoint}.meta.json")


def _guard_checkpoint(checkpoint: Path, signature: dict[str, Any]) -> Path:
    """Refuse to resume a checkpoint produced by different inputs/settings."""
    sidecar = _checkpoint_sidecar(checkpoint)
    if checkpoint.exists():
        if not sidecar.exists():
            raise ValueError(
                f"{checkpoint} has no provenance sidecar; refusing an unsafe resume"
            )
        observed = json.loads(sidecar.read_text(encoding="utf-8"))
        if observed != signature:
            keys = sorted(set(observed) | set(signature))
            drift = {
                key: (observed.get(key), signature.get(key))
                for key in keys
                if observed.get(key) != signature.get(key)
            }
            raise ValueError(f"checkpoint provenance mismatch: {drift}")
    temp = sidecar.with_name(f".{sidecar.name}.tmp")
    temp.write_text(
        json.dumps(signature, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temp.replace(sidecar)
    return sidecar


def fit_shard(
    *,
    corpus_path: str,
    output_path: str,
    shard_index: int = 0,
    num_shards: int = 1,
    source_layer_spec: str = "all",
    target_layer: int = -1,
    dim_batch: int = DEFAULT_DIM_BATCH,
    max_seq_len: int = DEFAULT_MAX_SEQ_LEN,
    skip_first: int = DEFAULT_SKIP_FIRST,
    checkpoint_every: int = 10,
    keep_checkpoint: bool = False,
    dtype: str = "bfloat16",
    device_map: str = "auto",
    compile_blocks: bool = False,
    disable_mamba_kernels: bool = False,
    cache_dir: str | None = None,
) -> dict[str, Any]:
    """Fit one exact prompt shard and save a mergeable artifact + sidecar."""
    if dim_batch <= 0:
        raise ValueError("dim_batch must be positive")
    if max_seq_len <= skip_first + 1:
        raise ValueError("max_seq_len must leave a valid position after skip_first")
    if checkpoint_every <= 0:
        raise ValueError("checkpoint_every must be positive")
    records, corpus_metadata = load_corpus(corpus_path)
    shard = select_shard(records, shard_index=shard_index, num_shards=num_shards)
    if not shard:
        raise ValueError(f"shard {shard_index}/{num_shards} contains no prompts")

    target = resolve_layer(target_layer, NEMOTRON.n_layers)
    source_layers = parse_source_layers(
        source_layer_spec, n_layers=NEMOTRON.n_layers, target_layer=target
    )
    loaded = load_nemotron(
        dtype=dtype,
        device_map=device_map,
        compile_blocks=compile_blocks,
        disable_mamba_kernels=disable_mamba_kernels,
        cache_dir=cache_dir,
    )

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = Path(f"{output}.checkpoint.pt")
    checkpoint_signature = {
        "schema_version": 1,
        "model_id": loaded.model_id,
        "model_revision": loaded.revision,
        "corpus_manifest_sha256": corpus_metadata["manifest_sha256"],
        "shard_index": shard_index,
        "num_shards": num_shards,
        "prompt_hashes": [record.sha256 for record in shard],
        "source_layers": source_layers,
        "target_layer": target,
        "dim_batch": dim_batch,
        "max_seq_len": max_seq_len,
        "skip_first": skip_first,
        "dtype": dtype,
        "storage_dtype": ARTIFACT_STORAGE_DTYPE,
        "device_map": device_map,
        "compile_blocks": compile_blocks,
        "disable_mamba_kernels": disable_mamba_kernels,
        "mamba_backend": loaded.mamba_backend,
        "patched_mamba_layers": loaded.patched_mamba_layers,
        "upstream_jlens_commit": UPSTREAM_JLENS_COMMIT,
        "adaptation_source_sha256": adaptation_source_sha256(),
        "runtime": loaded.runtime_identity,
    }
    checkpoint_sidecar = _guard_checkpoint(checkpoint, checkpoint_signature)
    lens = jlens.fit(
        loaded.lens_model,
        prompts=[record.text for record in shard],
        source_layers=source_layers,
        target_layer=target,
        dim_batch=dim_batch,
        max_seq_len=max_seq_len,
        skip_first=skip_first,
        checkpoint_path=str(checkpoint),
        checkpoint_every=checkpoint_every,
        resume=True,
    )
    if lens.n_prompts != len(shard):
        raise RuntimeError(
            f"fit accepted {lens.n_prompts}/{len(shard)} prompts; refusing to attach "
            "incorrect prompt provenance (fix the corpus and remove the checkpoint)"
        )
    validate_lens(
        lens, expected_d_model=NEMOTRON.d_model, expected_layers=source_layers
    )
    metadata: dict[str, Any] = {
        "schema_version": 1,
        "kind": "shard",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "upstream_jlens_commit": UPSTREAM_JLENS_COMMIT,
        "adaptation_source_sha256": adaptation_source_sha256(),
        "model_id": loaded.model_id,
        "model_revision": loaded.revision,
        "architecture": NEMOTRON.to_dict(),
        "dataset": corpus_metadata["dataset"],
        "corpus_manifest_sha256": corpus_metadata["manifest_sha256"],
        "corpus_path": os.path.abspath(corpus_path),
        "shard_index": shard_index,
        "num_shards": num_shards,
        "n_prompts": lens.n_prompts,
        "prompt_indices": [record.index for record in shard],
        "prompt_hashes": [record.sha256 for record in shard],
        "source_layers": source_layers,
        "target_layer": target,
        "dim_batch": dim_batch,
        "max_seq_len": max_seq_len,
        "skip_first": skip_first,
        "dtype": dtype,
        "storage_dtype": ARTIFACT_STORAGE_DTYPE,
        "device_map": device_map,
        "compile_blocks": compile_blocks,
        "disable_mamba_kernels": disable_mamba_kernels,
        "mamba_backend": loaded.mamba_backend,
        "patched_mamba_layers": loaded.patched_mamba_layers,
        "runtime": loaded.runtime_identity,
        "environment": environment_metadata(loaded.runtime_identity),
    }
    metadata["acceptance"] = classify_reproduction(metadata)
    metadata = save_validated_artifact(
        lens,
        output,
        metadata,
        expected_kind="shard",
    )
    if not keep_checkpoint:
        checkpoint.unlink(missing_ok=True)
        checkpoint_sidecar.unlink(missing_ok=True)
    return metadata


def _example_by_slug(slug: str):
    for example in EXAMPLES:
        if example.slug == slug:
            return example
    raise ValueError(f"unknown example {slug!r}; have {[e.slug for e in EXAMPLES]}")


def render_static_demo(
    *,
    lens_path: str,
    output_dir: str,
    example_slug: str = "modulation-topic",
    prompt: str | None = None,
    top_n: int = 8,
    layer_stride: int = 1,
    last_n_tokens: int | None = None,
    max_seq_len: int = 512,
    max_tracked: int = 512,
    dtype: str = "bfloat16",
    device_map: str = "auto",
    compile_blocks: bool = False,
    disable_mamba_kernels: bool = False,
    cache_dir: str | None = None,
) -> dict[str, Any]:
    """Render the official interactive slice viewer for a fitted Nemotron lens."""
    lens = jlens.JacobianLens.load(lens_path)
    validate_lens(lens, expected_d_model=NEMOTRON.d_model)
    if not metadata_path(lens_path).exists():
        raise ValueError("static Nemotron render requires a lens metadata sidecar")
    lens_metadata = read_metadata(lens_path)
    validate_artifact(lens, lens_metadata)
    if (
        lens_metadata["model_id"] != NEMOTRON.model_id
        or lens_metadata["model_revision"] != NEMOTRON.revision
    ):
        raise ValueError(
            "lens sidecar does not identify the pinned Nemotron checkpoint: "
            f"{lens_metadata['model_id']}@{lens_metadata['model_revision']}"
        )
    loaded = load_nemotron(
        dtype=dtype,
        device_map=device_map,
        compile_blocks=compile_blocks,
        disable_mamba_kernels=disable_mamba_kernels,
        cache_dir=cache_dir,
    )
    if lens_metadata.get("dtype") != dtype:
        raise ValueError(
            f"render dtype {dtype!r} differs from lens dtype "
            f"{lens_metadata.get('dtype')!r}"
        )
    if lens_metadata.get("mamba_backend") != loaded.mamba_backend:
        raise ValueError("render Mamba backend differs from the fitted lens backend")
    if lens_metadata.get("runtime") != loaded.runtime_identity:
        raise ValueError("render runtime identity differs from the fitted lens runtime")

    if prompt is None:
        example = _example_by_slug(example_slug)
        prompt = resolve_prompt(example, loaded.tokenizer)
        title = f"Nemotron 3 Nano · {example.section}"
        description = example.description
    else:
        title = "Nemotron 3 Nano · Jacobian lens"
        description = "User-supplied prompt"

    pinned: set[int] = set()
    if example_slug == "modulation-topic":
        for word in (" fish", " whale", " coral", " lobster"):
            ids = loaded.tokenizer.encode(word, add_special_tokens=False)
            if len(ids) == 1:
                pinned.add(int(ids[0]))

    slice_data = compute_slice(
        loaded.lens_model,
        lens,
        prompt,
        top_n=top_n,
        max_tracked=max_tracked,
        pinned_token_ids=pinned,
        layer_stride=layer_stride,
        last_n_tokens=last_n_tokens,
        max_seq_len=max_seq_len,
        mask_display=True,
    )
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    page, raw_bytes, payload_bytes = build_page(
        slice_data,
        prompt,
        title=title,
        description=description,
        pinned_token_ids=pinned,
        mode="fetch",
        out_dir=output,
    )
    (output / "index.html").write_text(page, encoding="utf-8")
    result = {
        "index": str((output / "index.html").resolve()),
        "example": example_slug,
        "layers": slice_data.layers,
        "seq_len": slice_data.seq_len,
        "raw_bytes": raw_bytes,
        "payload_bytes": payload_bytes,
        "pinned_token_ids": sorted(pinned),
        "model_id": loaded.model_id,
        "model_revision": loaded.revision,
        "mamba_backend": loaded.mamba_backend,
        "patched_mamba_layers": loaded.patched_mamba_layers,
        "runtime": loaded.runtime_identity,
        "lens_sha256": lens_metadata["artifact_sha256"],
        "lens_n_prompts": lens.n_prompts,
        "prompt_sha256": sha256_text(prompt),
        "corpus_manifest_sha256": lens_metadata.get("corpus_manifest_sha256"),
        "dataset": lens_metadata.get("dataset"),
        "upstream_jlens_commit": UPSTREAM_JLENS_COMMIT,
        "adaptation_source_sha256": adaptation_source_sha256(),
    }
    (output / "build.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result
