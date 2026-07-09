"""Load and validate the pinned Nemotron checkpoint for Jacobian analysis."""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from types import MethodType
from typing import Any

import torch

import jlens
from nemotron_jlens.config import NEMOTRON, NemotronSpec
from nemotron_jlens.runtime import require_pinned_transformers, runtime_identity


@dataclass
class LoadedNemotron:
    """The raw Hugging Face model and its jlens-compatible view."""

    hf_model: Any
    tokenizer: Any
    lens_model: jlens.HFLensModel
    spec: NemotronSpec
    model_id: str
    revision: str
    mamba_backend: str
    patched_mamba_layers: int
    runtime_identity: dict[str, Any]


def resolve_dtype(name: str) -> torch.dtype:
    try:
        dtype = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }[name]
    except KeyError as exc:
        raise ValueError(f"unsupported dtype {name!r}") from exc
    return dtype


def _validate_config(config: Any, spec: NemotronSpec) -> None:
    observed = {
        "model_type": getattr(config, "model_type", None),
        "num_hidden_layers": getattr(config, "num_hidden_layers", None),
        "hidden_size": getattr(config, "hidden_size", None),
        "vocab_size": getattr(config, "vocab_size", None),
    }
    expected = {
        "model_type": spec.model_type,
        "num_hidden_layers": spec.n_layers,
        "hidden_size": spec.d_model,
        "vocab_size": spec.vocab_size,
    }
    drift = {
        key: {"expected": expected[key], "observed": value}
        for key, value in observed.items()
        if value != expected[key]
    }
    if drift:
        raise ValueError(
            "Nemotron checkpoint architecture drifted from the pinned reproduction "
            f"specification: {drift}"
        )


def _torch_grouped_rmsnorm_forward(
    module: Any, hidden_states: torch.Tensor, gate: torch.Tensor | None = None
) -> torch.Tensor:
    """Differentiable torch equivalent of Nemotron's fused gated group RMSNorm."""
    input_dtype = hidden_states.dtype
    if gate is not None:
        hidden_states = hidden_states.float() * torch.nn.functional.silu(gate.float())
    shape = hidden_states.shape
    dim = shape[-1]
    group_size = int(module.group_size)
    if group_size <= 0 or dim % group_size:
        raise RuntimeError(f"invalid Nemotron Mamba RMSNorm group size {group_size}")
    grouped = hidden_states.float().reshape(
        *shape[:-1], dim // group_size, group_size
    )
    variance = grouped.square().mean(dim=-1, keepdim=True)
    normalized = grouped * torch.rsqrt(variance + module.variance_epsilon)
    normalized = normalized.reshape(shape) * module.weight.float()
    return normalized.to(input_dtype)


def _torch_mamba_forward(
    module: Any,
    hidden_states: torch.Tensor,
    cache_params: Any = None,
    cache_position: torch.Tensor | None = None,
    attention_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    return module.torch_forward(
        hidden_states,
        cache_params=cache_params,
        cache_position=cache_position,
        attention_mask=attention_mask,
    )


def _install_forward(module: Any, function: Any) -> None:
    """Replace the implementation beneath an optional Accelerate device hook."""
    attribute = (
        "_old_forward" if callable(getattr(module, "_old_forward", None)) else "forward"
    )
    setattr(module, attribute, MethodType(function, module))


def force_torch_mamba(hf_model: Any) -> int:
    """Patch every pinned remote-code Mamba mixer onto its naive torch path.

    The checkpoint's remote implementation selects fused kernels through a
    module-global flag and ignores config.use_mamba_kernels. Patching each
    mixer instance is explicit and fail-closed if the remote class changes.
    """
    block_types = getattr(hf_model.config, "layers_block_type", None)
    if not isinstance(block_types, (list, tuple)):
        raise RuntimeError("Nemotron config does not expose layers_block_type")
    expected = sum(block_type == "mamba" for block_type in block_types)
    patched = 0
    for module in hf_model.modules():
        if type(module).__name__ != "NemotronHMamba2Mixer":
            continue
        if not callable(getattr(module, "torch_forward", None)):
            raise RuntimeError("Nemotron Mamba mixer has no torch_forward fallback")
        norm = getattr(module, "norm", None)
        if norm is None or not hasattr(norm, "group_size"):
            raise RuntimeError("Nemotron Mamba mixer has an unknown gated RMSNorm")
        _install_forward(module, _torch_mamba_forward)
        _install_forward(norm, _torch_grouped_rmsnorm_forward)
        module._nemotron_jlens_backend = "torch"
        patched += 1
    if expected <= 0 or patched != expected:
        raise RuntimeError(
            f"patched {patched} Nemotron Mamba mixers, expected {expected}"
        )
    return patched


def _require_remote_dependencies() -> None:
    missing = [
        package
        for package in ("mamba_ssm", "causal_conv1d")
        if importlib.util.find_spec(package) is None
    ]
    if missing:
        raise RuntimeError(
            "the pinned Nemotron remote model requires CUDA-compatible builds of "
            f"{', '.join(missing)}; use the documented NVIDIA environment before "
            "running preflight"
        )


def load_nemotron(
    *,
    model_id: str = NEMOTRON.model_id,
    revision: str = NEMOTRON.revision,
    dtype: str = "bfloat16",
    device_map: str = "auto",
    compile_blocks: bool = False,
    disable_mamba_kernels: bool = False,
    cache_dir: str | None = None,
    spec: NemotronSpec = NEMOTRON,
) -> LoadedNemotron:
    """Load Nemotron and adapt its hybrid residual stack to ``jlens``.

    ``device_map='auto'`` is the recommended two-or-more-GPU configuration.
    ``device_map='cuda'`` loads without Accelerate sharding and then moves the
    full model onto one GPU. Per-block ``torch.compile`` is deliberately opt-in
    and rejected with Accelerate sharding, matching upstream jlens guidance.
    """
    # Check distribution metadata before importing any Transformers model code.
    # The pinned remote implementation is not supported on a merely compatible
    # version range: its exact vendor-tested release is part of the experiment.
    require_pinned_transformers()
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required to load Nemotron 3 Nano for fitting or inference. "
            "Run `nemotron-jlens info` for sizing without a GPU."
        )
    if compile_blocks and device_map != "cuda":
        raise ValueError(
            "--compile-blocks requires --device-map cuda; Accelerate's per-layer "
            "hooks are incompatible with the jlens per-block compile strategy"
        )

    _require_remote_dependencies()
    scientific_runtime = runtime_identity()
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    torch_dtype = resolve_dtype(dtype)
    config = AutoConfig.from_pretrained(
        model_id,
        revision=revision,
        trust_remote_code=True,
        cache_dir=cache_dir,
    )
    if model_id == spec.model_id:
        _validate_config(config, spec)

    load_kwargs: dict[str, Any] = {
        "config": config,
        "revision": revision,
        "trust_remote_code": True,
        "dtype": torch_dtype,
        "low_cpu_mem_usage": True,
        "cache_dir": cache_dir,
    }
    if device_map != "cuda":
        load_kwargs["device_map"] = device_map

    hf_model = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs)
    patched_mamba_layers = (
        force_torch_mamba(hf_model) if disable_mamba_kernels else 0
    )
    mamba_backend = "torch" if disable_mamba_kernels else "fused-or-auto"
    if device_map == "cuda":
        hf_model = hf_model.cuda()
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        revision=revision,
        trust_remote_code=True,
        cache_dir=cache_dir,
    )
    lens_model = jlens.from_hf(
        hf_model,
        tokenizer,
        layout=spec.layout,
        compile=compile_blocks,
        force_bos=True,
    )

    if lens_model.n_layers != spec.n_layers or lens_model.d_model != spec.d_model:
        raise ValueError(
            "loaded model disagrees with the Nemotron jlens adapter: "
            f"got {lens_model.n_layers} layers, d_model={lens_model.d_model}"
        )
    return LoadedNemotron(
        hf_model=hf_model,
        tokenizer=tokenizer,
        lens_model=lens_model,
        spec=spec,
        model_id=model_id,
        revision=revision,
        mamba_backend=mamba_backend,
        patched_mamba_layers=patched_mamba_layers,
        runtime_identity=scientific_runtime,
    )
