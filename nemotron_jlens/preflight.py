"""Cheap vector-Jacobian preflight before starting a multi-day lens fit."""

from __future__ import annotations

import math
import time
from typing import Any

import torch

from jlens.hooks import ActivationRecorder
from jlens.protocol import LensModel


def check_native_logit_parity(
    hf_model: Any,
    model: LensModel,
    prompt: str,
    *,
    max_seq_len: int = 32,
    atol: float = 5e-2,
) -> dict[str, Any]:
    """Verify the explicit layout's norm+unembed matches native HF logits."""
    final_layer = model.n_layers - 1
    input_ids = model.encode(prompt, max_length=max_seq_len)
    with torch.no_grad(), ActivationRecorder(model.layers, at=[final_layer]) as recorder:
        model.forward(input_ids)
        residual = recorder.activations[final_layer][:, -1].float()
        adapted = model.unembed(residual).float()
        native = hf_model(input_ids=input_ids, use_cache=False).logits[:, -1].float()
    if adapted.shape != native.shape:
        raise RuntimeError(
            f"adapter/native logit shape mismatch: {adapted.shape} != {native.shape}"
        )
    delta = (adapted - native).abs()
    max_abs = float(delta.max().item())
    mean_abs = float(delta.mean().item())
    cosine = float(torch.nn.functional.cosine_similarity(adapted, native).mean().item())
    if max_abs > atol:
        raise RuntimeError(
            f"Nemotron layout does not reproduce native logits: max_abs={max_abs:.4g} "
            f"> atol={atol:.4g}"
        )
    return {
        "ok": True,
        "max_abs_error": max_abs,
        "mean_abs_error": mean_abs,
        "cosine_similarity": cosine,
        "atol": atol,
    }


def run_vjp_preflight(
    model: LensModel,
    prompt: str,
    *,
    source_layer: int = 0,
    target_layer: int = -1,
    max_seq_len: int = 32,
) -> dict[str, Any]:
    """Run one backward pass through the real residual stack.

    A full fit needs ``ceil(d_model / dim_batch)`` backward passes per prompt.
    This preflight performs only one vector-Jacobian product, but traverses all
    blocks between ``source_layer`` and ``target_layer``. Starting at layer 0
    therefore catches unsupported fused-Mamba backward kernels before any
    expensive artifact allocation or corpus processing begins.
    """
    n_layers = model.n_layers
    if target_layer < 0:
        target_layer += n_layers
    if source_layer < 0:
        source_layer += n_layers
    if not 0 <= source_layer < target_layer < n_layers:
        raise ValueError(
            f"need 0 <= source_layer < target_layer < {n_layers}; got "
            f"{source_layer}, {target_layer}"
        )

    input_ids = model.encode(prompt, max_length=max_seq_len)
    if input_ids.shape[1] < 2:
        raise ValueError("preflight prompt tokenized to fewer than two tokens")

    started = time.perf_counter()
    with (
        ActivationRecorder(
            model.layers,
            at=[source_layer, target_layer],
            start_graph_at=source_layer,
        ) as recorder,
        torch.enable_grad(),
    ):
        model.forward(input_ids)
        source = recorder.activations[source_layer]
        target = recorder.activations[target_layer]
        cotangent = torch.zeros_like(target)
        cotangent[:, -1, :] = 1 / math.sqrt(target.shape[-1])
        (gradient,) = torch.autograd.grad(
            outputs=target,
            inputs=[source],
            grad_outputs=cotangent,
        )

    grad_float = gradient.detach().float()
    finite = bool(torch.isfinite(grad_float).all().item())
    norm = float(grad_float.norm().item())
    if not finite:
        raise RuntimeError("Nemotron VJP contains NaN or Inf values")
    if norm == 0.0:
        raise RuntimeError("Nemotron VJP is identically zero")

    result: dict[str, Any] = {
        "ok": True,
        "source_layer": source_layer,
        "target_layer": target_layer,
        "seq_len": int(input_ids.shape[1]),
        "d_model": int(target.shape[-1]),
        "gradient_norm": norm,
        "gradient_abs_max": float(grad_float.abs().max().item()),
        "elapsed_seconds": time.perf_counter() - started,
        "source_device": str(source.device),
        "target_device": str(target.device),
    }
    if torch.cuda.is_available():
        result["cuda_peak_allocated_gib"] = [
            torch.cuda.max_memory_allocated(i) / 1024**3
            for i in range(torch.cuda.device_count())
        ]
    return result
