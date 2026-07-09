"""Export genuine lens measurements to the local Neuronpedia-style demo."""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

import jlens
from jlens.examples import EXAMPLES, Example, resolve_prompt
from jlens.hooks import ActivationRecorder
from jlens.protocol import LensModel
from jlens.vis import _meaningful_token_mask
from nemotron_jlens.artifacts import (
    metadata_path,
    read_metadata,
    validate_artifact,
    validate_lens,
)
from nemotron_jlens.config import NEMOTRON, UPSTREAM_JLENS_COMMIT
from nemotron_jlens.corpus import sha256_file, sha256_text
from nemotron_jlens.loading import load_nemotron
from nemotron_jlens.provenance import adaptation_source_sha256, require_sha256

JACOBIAN_LENS = "JACOBIAN_LENS"
LOGIT_LENS = "LOGIT_LENS"
DEFAULT_TRACKED_TOKENS = (" fish", " whale", " coral", " lobster")

_CARRIER_SENTENCE = "She carefully placed the letter back inside the wooden drawer."
_CONTROL_EXAMPLES = (
    Example(
        slug="modulation-topic-suppress",
        section="Voluntary modulation: suppress topic",
        description="Released negated-think control for directed modulation.",
        user=(
            f'Write "{_CARRIER_SENTENCE}" '
            "Don't think about ocean creatures. Don't write anything else."
        ),
        assistant_prefill=_CARRIER_SENTENCE,
    ),
    Example(
        slug="modulation-topic-mention-control",
        section="Voluntary modulation: mention control",
        description="Released mention-only control for directed modulation.",
        user=(
            f'Write "{_CARRIER_SENTENCE}" This prompt contains a reference to '
            "ocean creatures. Don't write anything else."
        ),
        assistant_prefill=_CARRIER_SENTENCE,
    ),
    Example(
        slug="modulation-topic-neutral",
        section="Voluntary modulation: neutral carrier",
        description="Additional no-side-task baseline for the carrier sentence.",
        user=f'Write "{_CARRIER_SENTENCE}" Don\'t write anything else.',
        assistant_prefill=_CARRIER_SENTENCE,
    ),
)


def _require_finite_logits(logits: torch.Tensor) -> None:
    """Reject non-finite model output before ranks or probabilities are derived."""
    if not bool(torch.isfinite(logits).all().item()):
        raise ValueError("recorded fixture logits contain non-finite values")


def _write_fixture_atomic(output: Path, fixture: Mapping[str, Any]) -> None:
    """Serialize strict JSON and atomically publish it in ``output``'s directory."""
    payload = json.dumps(
        fixture,
        indent=2,
        ensure_ascii=False,
        allow_nan=False,
    ) + "\n"
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=output.parent,
            prefix=f".{output.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary_name = temporary.name
            temporary.write(payload)
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_name, output)
    except BaseException:
        if temporary_name is not None:
            Path(temporary_name).unlink(missing_ok=True)
        raise

def _decode(tokenizer: Any, token_id: int) -> str:
    return tokenizer.decode([token_id], clean_up_tokenization_spaces=False)


def _selected_layers(
    lens: jlens.JacobianLens, model: LensModel, layer_stride: int
) -> list[int]:
    if layer_stride <= 0:
        raise ValueError("layer_stride must be positive")
    layers = lens.source_layers[::layer_stride]
    if lens.source_layers[-1] not in layers:
        layers.append(lens.source_layers[-1])
    final_layer = model.n_layers - 1
    if final_layer not in layers:
        layers.append(final_layer)
    return sorted(set(layers))


def _top_readout(
    logits: torch.Tensor,
    *,
    tokenizer: Any,
    top_n: int,
    mask_display: bool,
) -> tuple[list[list[str]], list[list[float]]]:
    """Return decoded top tokens and true full-vocabulary probabilities."""
    if top_n <= 0 or top_n > logits.shape[-1]:
        raise ValueError(f"top_n must be in [1, {logits.shape[-1]}]")
    _require_finite_logits(logits)
    log_probs = logits.log_softmax(dim=-1)
    display_logits = logits
    if mask_display:
        mask = _meaningful_token_mask(tokenizer, logits.shape[-1], logits.device)
        if int(mask.sum().item()) < top_n:
            raise ValueError("tokenizer has fewer displayable tokens than top_n")
        display_logits = logits.masked_fill(~mask, -torch.inf)
    top_ids = display_logits.topk(top_n, dim=-1).indices
    top_probs = log_probs.gather(-1, top_ids).exp()
    decoded = [
        [_decode(tokenizer, int(token_id)) for token_id in row]
        for row in top_ids.detach().cpu().tolist()
    ]
    return decoded, top_probs.detach().float().cpu().tolist()


def _tracked_readout(
    logits: torch.Tensor,
    tracked_tokens: Mapping[str, int],
) -> dict[str, tuple[list[int], list[float]]]:
    """Return exact full-vocabulary rank and probability for pinned tokens."""
    if not tracked_tokens:
        return {}
    _require_finite_logits(logits)
    log_normalizer = logits.logsumexp(dim=-1)
    out: dict[str, tuple[list[int], list[float]]] = {}
    for token, token_id in tracked_tokens.items():
        if not 0 <= token_id < logits.shape[-1]:
            raise ValueError(f"tracked token id {token_id} is outside the vocabulary")
        selected = logits[:, token_id]
        ranks = (logits > selected[:, None]).sum(dim=-1) + 1
        probabilities = (selected - log_normalizer).exp()
        out[token] = (
            ranks.detach().cpu().tolist(),
            probabilities.detach().float().cpu().tolist(),
        )
    return out


def _validated_recording_provenance(
    model: LensModel,
    lens: jlens.JacobianLens,
    *,
    model_name: str,
    model_revision: str,
    lens_metadata: Mapping[str, Any],
) -> dict[str, Any]:
    """Bind a recorded fixture to the pinned model and fitted lens artifact."""
    if model_name != NEMOTRON.model_id or model_revision != NEMOTRON.revision:
        raise ValueError("recorded fixtures require the pinned Nemotron identity")
    if model.n_layers != NEMOTRON.n_layers or model.d_model != NEMOTRON.d_model:
        raise ValueError("recorded fixture model shape disagrees with pinned Nemotron")
    validate_artifact(lens, lens_metadata)
    if (
        lens_metadata.get("model_id") != NEMOTRON.model_id
        or lens_metadata.get("model_revision") != NEMOTRON.revision
        or lens_metadata.get("architecture") != NEMOTRON.to_dict()
    ):
        raise ValueError("lens provenance does not identify the pinned Nemotron artifact")
    if lens_metadata.get("upstream_jlens_commit") != UPSTREAM_JLENS_COMMIT:
        raise ValueError("lens upstream jlens revision differs from this reproduction")
    current_source = adaptation_source_sha256()
    lens_source = require_sha256(
        lens_metadata.get("adaptation_source_sha256"),
        field="lens adaptation_source_sha256",
    )
    if lens_source != current_source:
        raise ValueError("lens adaptation source differs from the current exporter source")
    lens_sha = require_sha256(
        lens_metadata.get("artifact_sha256"), field="lens artifact_sha256"
    )
    return {
        "sha256": lens_sha,
        "n_prompts": lens.n_prompts,
        "source_layers": lens.source_layers,
        "target_layer": lens_metadata["target_layer"],
        "prompt_indices": lens_metadata["prompt_indices"],
        "corpus_manifest_sha256": lens_metadata["corpus_manifest_sha256"],
        "dataset": lens_metadata["dataset"],
        "storage_dtype": lens_metadata["storage_dtype"],
        "acceptance": lens_metadata["acceptance"],
        "upstream_jlens_commit": lens_metadata["upstream_jlens_commit"],
        "adaptation_source_sha256": lens_source,
        "runtime": lens_metadata["runtime"],
        "fit": {
            key: lens_metadata.get(key)
            for key in (
                "dim_batch", "max_seq_len", "skip_first", "dtype",
                "compile_blocks", "disable_mamba_kernels",
                "mamba_backend", "patched_mamba_layers",
            )
        },
    }


@torch.no_grad()
def build_fixture(
    model: LensModel,
    lens: jlens.JacobianLens,
    prompt: str,
    *,
    model_name: str,
    model_revision: str,
    lens_metadata: Mapping[str, Any],
    prompt_len: int | None = None,
    completion: str = "",
    top_n: int = 8,
    layer_stride: int = 1,
    last_n_tokens: int | None = None,
    max_seq_len: int = 512,
    tracked_tokens: Mapping[str, int] | None = None,
    mask_display: bool = True,
    fixture_metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Compute a JSON-serializable fixture from real model activations.

    Both readouts use the same captured residuals. ``LOGIT_LENS`` applies the
    native final norm and unembedding directly; ``JACOBIAN_LENS`` first maps
    each fitted source residual through its average Jacobian. The final layer
    is included in both modes with ``J = I``.
    """
    lens_provenance = _validated_recording_provenance(
        model,
        lens,
        model_name=model_name,
        model_revision=model_revision,
        lens_metadata=lens_metadata,
    )
    if last_n_tokens is not None and last_n_tokens <= 0:
        raise ValueError("last_n_tokens must be positive")
    layers = _selected_layers(lens, model, layer_stride)
    tracked_tokens = dict(tracked_tokens or {})
    input_ids = model.encode(prompt, max_length=max_seq_len)
    full_len = int(input_ids.shape[1])
    if prompt_len is None:
        prompt_len = full_len
    if not 0 <= prompt_len <= full_len:
        raise ValueError(f"prompt_len={prompt_len} is outside sequence length {full_len}")
    start = 0 if last_n_tokens is None else max(0, full_len - last_n_tokens)
    positions = list(range(start, full_len))

    with ActivationRecorder(model.layers, at=layers) as recorder:
        model.forward(input_ids)
        activations = {
            layer: recorder.activations[layer][0, start:].detach() for layer in layers
        }

    results: dict[str, list[dict[str, Any]]] = {
        readout_type: [
            {
                "type": readout_type,
                "top_tokens": [],
                "top_probs": [],
                "tracked": [
                    {"token": token, "id": token_id, "ranks": [], "probs": []}
                    for token, token_id in tracked_tokens.items()
                ],
            }
            for _ in positions
        ]
        for readout_type in (JACOBIAN_LENS, LOGIT_LENS)
    }

    final_layer = model.n_layers - 1
    for layer in layers:
        raw_residual = activations[layer].float()
        residuals = {LOGIT_LENS: raw_residual}
        residuals[JACOBIAN_LENS] = (
            lens.transport(raw_residual, layer)
            if layer in lens.jacobians
            else raw_residual
        )
        cached_final: torch.Tensor | None = None
        for readout_type in (JACOBIAN_LENS, LOGIT_LENS):
            if layer == final_layer and cached_final is not None:
                logits = cached_final
            else:
                logits = model.unembed(residuals[readout_type]).float()
                if layer == final_layer:
                    cached_final = logits
            top_tokens, top_probs = _top_readout(
                logits,
                tokenizer=model.tokenizer,
                top_n=top_n,
                mask_display=mask_display,
            )
            tracked = _tracked_readout(logits, tracked_tokens)
            for row_index in range(len(positions)):
                result = results[readout_type][row_index]
                result["top_tokens"].append(top_tokens[row_index])
                result["top_probs"].append(top_probs[row_index])
                for tracked_row in result["tracked"]:
                    ranks, probs = tracked[tracked_row["token"]]
                    tracked_row["ranks"].append(int(ranks[row_index]))
                    tracked_row["probs"].append(float(probs[row_index]))
            if logits is not cached_final:
                del logits

    token_ids = input_ids[0].detach().cpu().tolist()
    tokens = []
    for row_index, position in enumerate(positions):
        tokens.append(
            {
                "kind": "token",
                "position": position,
                "token": _decode(model.tokenizer, int(token_ids[position])),
                "id": int(token_ids[position]),
                "is_generated": position >= prompt_len,
                "results": [
                    results[JACOBIAN_LENS][row_index],
                    results[LOGIT_LENS][row_index],
                ],
            }
        )

    transcript = []
    if start < min(prompt_len, full_len):
        transcript.append(
            {
                "label": "Prompt",
                "start": start,
                "end": min(prompt_len, full_len) - 1,
                "generated": False,
            }
        )
    if max(start, prompt_len) < full_len:
        transcript.append(
            {
                "label": "Teacher-forced response",
                "start": max(start, prompt_len),
                "end": full_len - 1,
                "generated": True,
            }
        )

    selected_positions = list(range(max(start, prompt_len), full_len))
    extra_metadata = dict(fixture_metadata or {})
    exporter_source = adaptation_source_sha256()
    provenance = {
        "model": {"id": model_name, "revision": model_revision},
        "tokenizer": {"id": model_name, "revision": model_revision},
        "runtime": {
            **lens_provenance["runtime"],
            "mamba_backend": lens_metadata.get("mamba_backend"),
            "patched_mamba_layers": lens_metadata.get("patched_mamba_layers"),
        },
        "lens": lens_provenance,
        "prompt": {
            "example": extra_metadata.get("example"),
            "sha256": sha256_text(prompt),
            "construction": "tokenizer chat template with teacher-forced response",
        },
        "exporter": {
            "upstream_jlens_commit": UPSTREAM_JLENS_COMMIT,
            "adaptation_source_sha256": exporter_source,
        },
    }
    fixture_info = {
        **extra_metadata,
        "schema": "nemotron-jlens-fixture/v1",
        "mode": "recorded",
        "title": extra_metadata.get("title", "Directed modulation · ocean creatures"),
        "description": extra_metadata.get(
            "description",
            "Measured teacher-forced readouts from the pinned Nemotron checkpoint "
            "and fitted Jacobian lens. Provenance declarations are embedded below.",
        ),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "transcript": transcript,
        "ui": {
            "selected_positions": selected_positions[-11:],
            "locked_tokens": list(tracked_tokens)[:4],
        },
        "model_revision": model_revision,
        "lens_sha256": lens_provenance["sha256"],
        "lens_n_prompts": lens.n_prompts,
        "lens": lens_provenance,
        "acceptance": lens_provenance["acceptance"],
        "upstream_jlens_commit": lens_provenance["upstream_jlens_commit"],
        "adaptation_source_sha256": lens_provenance["adaptation_source_sha256"],
        "exporter_upstream_jlens_commit": UPSTREAM_JLENS_COMMIT,
        "exporter_adaptation_source_sha256": exporter_source,
        "tokenizer": provenance["tokenizer"],
        "prompt_sha256": sha256_text(prompt),
        "provenance": provenance,
    }
    return {
        "_fixture": fixture_info,
        "meta": {
            "kind": "meta",
            "model": model_name,
            "types": [JACOBIAN_LENS, LOGIT_LENS],
            "layers_by_type": {
                JACOBIAN_LENS: layers,
                LOGIT_LENS: layers,
            },
            "top_n": top_n,
            "prompt_len": prompt_len,
            "num_completion_tokens": full_len - prompt_len,
            "temperature": 0,
            "prepend_bos": bool(getattr(model.tokenizer, "bos_token_id", None) is not None),
            "reuse_len": 0,
            "window_start": start,
        },
        "tokens": tokens,
        "done": {
            "kind": "done",
            "seq_len": full_len,
            "prompt_len": prompt_len,
            "vocab_size": int(model.unembed(activations[final_layer][:1].float()).shape[-1]),
            "completion": completion,
        },
    }


def _example_by_slug(slug: str) -> Example:
    examples = [*EXAMPLES, *_CONTROL_EXAMPLES]
    for example in examples:
        if example.slug == slug:
            return example
    raise ValueError(f"unknown example {slug!r}; have {[e.slug for e in examples]}")


def _prompt_boundary(model: LensModel, example: Example, full_prompt: str) -> int:
    """Find the assistant prefill boundary in the actual encoded sequence."""
    full_ids = model.encode(full_prompt, max_length=4096)[0].tolist()
    if not example.assistant_prefill:
        return len(full_ids)
    start = full_prompt.rfind(example.assistant_prefill)
    if start < 0:
        raise ValueError("assistant prefill is not present in the resolved prompt")
    prefix_ids = model.encode(full_prompt[:start], max_length=4096)[0].tolist()
    common = 0
    for left, right in zip(prefix_ids, full_ids, strict=False):
        if left != right:
            break
        common += 1
    if common < max(1, len(prefix_ids) - 1):
        raise ValueError("could not align the chat-template assistant boundary")
    return common


def _resolve_tracked_tokens(
    tokenizer: Any, values: Sequence[str]
) -> tuple[dict[str, int], dict[str, Any]]:
    resolved: dict[str, int] = {}
    decomposed: dict[str, Any] = {}
    for value in values:
        token_ids = tokenizer.encode(value, add_special_tokens=False)
        if len(token_ids) == 1:
            resolved[value] = int(token_ids[0])
            continue
        pieces = [
            tokenizer.decode([token_id], clean_up_tokenization_spaces=False)
            for token_id in token_ids
        ]
        decomposed[value] = {
            "reason": f"encodes to {len(token_ids)} tokens",
            "token_ids": [int(token_id) for token_id in token_ids],
            "pieces": pieces,
        }
        for token_id, piece in zip(token_ids, pieces, strict=True):
            label = piece
            if label in resolved and resolved[label] != int(token_id):
                label = f"{piece} [id {token_id}]"
            resolved.setdefault(label, int(token_id))
    return resolved, decomposed


def export_fixture(
    *,
    lens_path: str,
    output_path: str,
    example_slug: str = "modulation-topic",
    top_n: int = 8,
    layer_stride: int = 1,
    last_n_tokens: int | None = None,
    max_seq_len: int = 512,
    tracked_token_values: Sequence[str] = DEFAULT_TRACKED_TOKENS,
    dtype: str = "bfloat16",
    device_map: str = "auto",
    compile_blocks: bool = False,
    disable_mamba_kernels: bool = False,
    cache_dir: str | None = None,
) -> dict[str, Any]:
    """Load the pinned model and fitted lens, then write a recorded fixture."""
    lens = jlens.JacobianLens.load(lens_path)
    validate_lens(lens, expected_d_model=NEMOTRON.d_model)
    if not metadata_path(lens_path).exists():
        raise ValueError("recorded fixture export requires a lens metadata sidecar")
    lens_meta = read_metadata(lens_path)
    validate_artifact(lens, lens_meta)
    if (
        lens_meta["model_id"] != NEMOTRON.model_id
        or lens_meta["model_revision"] != NEMOTRON.revision
    ):
        raise ValueError(
            "lens sidecar does not identify the pinned Nemotron checkpoint: "
            f"{lens_meta['model_id']}@{lens_meta['model_revision']}"
        )
    loaded = load_nemotron(
        dtype=dtype,
        device_map=device_map,
        compile_blocks=compile_blocks,
        disable_mamba_kernels=disable_mamba_kernels,
        cache_dir=cache_dir,
    )
    if lens_meta.get("dtype") != dtype:
        raise ValueError(
            f"export dtype {dtype!r} differs from lens dtype {lens_meta.get('dtype')!r}"
        )
    if lens_meta.get("mamba_backend") != loaded.mamba_backend:
        raise ValueError(
            "export Mamba backend differs from the fitted lens backend"
        )
    if lens_meta.get("runtime") != loaded.runtime_identity:
        raise ValueError("export runtime identity differs from the fitted lens runtime")
    example = _example_by_slug(example_slug)
    prompt = resolve_prompt(example, loaded.tokenizer)
    prompt_len = _prompt_boundary(loaded.lens_model, example, prompt)
    tracked_tokens, skipped_tokens = _resolve_tracked_tokens(
        loaded.tokenizer, tracked_token_values
    )
    fit_provenance = {
        key: lens_meta.get(key)
        for key in (
            "dim_batch",
            "max_seq_len",
            "skip_first",
            "dtype",
            "compile_blocks",
            "disable_mamba_kernels",
            "mamba_backend",
            "patched_mamba_layers",
        )
    }
    lens_provenance = {
        "sha256": lens_meta["artifact_sha256"],
        "n_prompts": lens.n_prompts,
        "source_layers": lens.source_layers,
        "target_layer": lens_meta.get("target_layer"),
        "corpus_manifest_sha256": lens_meta.get("corpus_manifest_sha256"),
        "dataset": lens_meta.get("dataset"),
        "storage_dtype": lens_meta.get("storage_dtype"),
        "acceptance": lens_meta.get("acceptance"),
        "runtime": lens_meta.get("runtime"),
        "fit": fit_provenance,
    }
    provenance = {
        "model": {"id": loaded.model_id, "revision": loaded.revision},
        "tokenizer": {"id": loaded.model_id, "revision": loaded.revision},
        "runtime": {
            **loaded.runtime_identity,
            "mamba_backend": loaded.mamba_backend,
            "patched_mamba_layers": loaded.patched_mamba_layers,
        },
        "lens": lens_provenance,
        "prompt": {
            "example": example_slug,
            "sha256": sha256_text(prompt),
            "construction": "tokenizer chat template with teacher-forced response",
        },
        "exporter": {
            "upstream_jlens_commit": UPSTREAM_JLENS_COMMIT,
            "adaptation_source_sha256": adaptation_source_sha256(),
        },
    }
    fixture = build_fixture(
        loaded.lens_model,
        lens,
        prompt,
        model_name=loaded.model_id,
        model_revision=loaded.revision,
        lens_metadata=lens_meta,
        prompt_len=prompt_len,
        completion=example.assistant_prefill,
        top_n=top_n,
        layer_stride=layer_stride,
        last_n_tokens=last_n_tokens,
        max_seq_len=max_seq_len,
        tracked_tokens=tracked_tokens,
        fixture_metadata={
            "example": example_slug,
            "title": example.section,
            "description": example.description,
            "prompt": prompt,
            "prompt_sha256": sha256_text(prompt),
            "provenance": provenance,
            "model_revision": loaded.revision,
            "tokenizer": provenance["tokenizer"],
            "lens_sha256": lens_meta["artifact_sha256"],
            "lens_n_prompts": lens.n_prompts,
            "lens": lens_provenance,
            "upstream_jlens_commit": UPSTREAM_JLENS_COMMIT,
            "skipped_tracked_tokens": skipped_tokens,
        },
    )
    output = Path(output_path)
    _write_fixture_atomic(output, fixture)
    return {
        "output": str(output.resolve()),
        "sha256": sha256_file(output),
        "positions": len(fixture["tokens"]),
        "layers": fixture["meta"]["layers_by_type"][JACOBIAN_LENS],
        "tracked_tokens": tracked_tokens,
        "skipped_tracked_tokens": skipped_tokens,
    }
