"""Persistent full-prefix inference and exact Jacobian/logit readouts."""

from __future__ import annotations

import hashlib
import json
import math
import threading
import time
from collections import OrderedDict
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

import torch

from nemotron_steering.constants import (
    LENS_SHA256,
    MAX_PROMPT_TOKENS,
    MODEL_ID,
    MODEL_REVISION,
    NEURONPEDIA_COMMIT,
    PILOT_DISCLOSURE,
    SOURCE_LAYERS,
)
from nemotron_steering.errors import InferenceCancelled, ValidationError
from nemotron_steering.interventions import (
    DirectionCache,
    ForwardContext,
    HookSession,
    unembedding_parameters,
)
from nemotron_steering.provenance import ValidatedLens, immutable_info
from nemotron_steering.requests import GenerationSpec, InferenceRequest
from nemotron_steering.validation import InterventionSpec

ProgressCallback = Callable[[dict[str, Any]], None]


def _progress(callback: ProgressCallback | None, phase: str, **values: Any) -> None:
    if callback is not None:
        callback({"phase": phase, **values})


def _cancelled(cancel_event: threading.Event | None) -> None:
    if cancel_event is not None and cancel_event.is_set():
        raise InferenceCancelled("request cancelled")


def _decode_token(tokenizer: Any, token_id: int) -> str:
    try:
        return tokenizer.decode(
            [token_id],
            clean_up_tokenization_spaces=False,
            skip_special_tokens=False,
        )
    except TypeError:
        return tokenizer.decode([token_id])


def _decode_sequence(tokenizer: Any, token_ids: Sequence[int]) -> str:
    try:
        return tokenizer.decode(
            list(token_ids),
            clean_up_tokenization_spaces=False,
            skip_special_tokens=False,
        )
    except TypeError:
        return tokenizer.decode(list(token_ids))


def prompt_token_hash(token_ids: Sequence[int]) -> str:
    payload = json.dumps(list(token_ids), separators=(",", ":")).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


class SteeringBackend:
    """One loaded model and CPU lens reused across serialized requests."""

    def __init__(
        self, loaded: Any, bundle: ValidatedLens, *, strict_identity: bool = True
    ) -> None:
        self.loaded = loaded
        self.bundle = bundle
        self.hf_model = loaded.hf_model
        self.tokenizer = loaded.tokenizer
        self.lens_model = loaded.lens_model
        self.backbone = self.hf_model.backbone
        self.layers = self.backbone.layers
        self.lm_head = self.hf_model.lm_head
        self.final_norm = getattr(
            self.lens_model, "_final_norm", getattr(self.backbone, "norm_f", None)
        )
        if strict_identity:
            self._validate_loaded_model()
        self.directions = DirectionCache(
            lm_head=self.lm_head,
            jacobians=bundle.lens.jacobians,
            lens_sha256=LENS_SHA256,
        )
        self._selected_parameter_cache: OrderedDict[
            tuple[int, ...], tuple[torch.Tensor, torch.Tensor | None]
        ] = OrderedDict()
        self._selected_parameter_lock = threading.Lock()
        self.info = immutable_info(bundle, loaded=loaded)

    def _validate_loaded_model(self) -> None:
        if self.loaded.model_id != MODEL_ID or self.loaded.revision != MODEL_REVISION:
            raise ValidationError(
                "loaded model identity differs from the pinned checkpoint"
            )
        if self.loaded.mamba_backend != "fused-or-auto":
            raise ValidationError("live inference requires fused-or-auto Mamba backend")
        if self.loaded.patched_mamba_layers != 0:
            raise ValidationError("live inference unexpectedly patched Mamba layers")
        if len(self.layers) != 52:
            raise ValidationError(
                f"loaded model has {len(self.layers)} residual blocks"
            )
        if tuple(self.lm_head.weight.shape) != (131072, 2688):
            raise ValidationError(
                f"loaded lm_head has unexpected shape {tuple(self.lm_head.weight.shape)}"
            )
        if self.lm_head.weight.dtype != torch.bfloat16:
            raise ValidationError(
                f"loaded lm_head dtype is {self.lm_head.weight.dtype}, expected bfloat16"
            )
        if self.loaded.runtime_identity != self.bundle.metadata["runtime"]:
            raise ValidationError(
                "loaded model runtime drifted after artifact validation"
            )

    @property
    def input_device(self) -> torch.device:
        return self.lens_model.input_device

    def _tokenize(self, request: InferenceRequest) -> tuple[torch.Tensor, str]:
        if request.prompt is not None:
            encoded = self.tokenizer(
                request.prompt,
                return_tensors="pt",
                add_special_tokens=True,
                truncation=False,
            )
            input_ids = (
                encoded["input_ids"]
                if isinstance(encoded, Mapping)
                else encoded.input_ids
            )
        else:
            input_ids = self.tokenizer.apply_chat_template(
                list(request.messages or ()),
                tokenize=True,
                add_generation_prompt=True,
                enable_thinking=request.enable_thinking,
                return_tensors="pt",
            )
            if isinstance(input_ids, Mapping):
                input_ids = input_ids["input_ids"]
        if not isinstance(input_ids, torch.Tensor):
            input_ids = torch.tensor(input_ids, dtype=torch.long)
        if input_ids.ndim == 1:
            input_ids = input_ids.unsqueeze(0)
        if input_ids.ndim != 2 or input_ids.shape[0] != 1:
            raise ValidationError("tokenizer must produce one [1, sequence] input")
        if input_ids.shape[1] == 0:
            raise ValidationError("formatted prompt produced no tokens")
        if input_ids.shape[1] > MAX_PROMPT_TOKENS:
            raise ValidationError(
                f"formatted prompt has {input_ids.shape[1]} tokens; maximum is "
                f"{MAX_PROMPT_TOKENS}"
            )
        ids = [int(token_id) for token_id in input_ids[0].tolist()]
        return input_ids.to(self.input_device), _decode_sequence(self.tokenizer, ids)

    def _backbone_last(self, input_ids: torch.Tensor) -> torch.Tensor:
        attention_mask = torch.ones_like(input_ids, dtype=torch.long)
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=True,
        )
        hidden = (
            outputs.last_hidden_state
            if hasattr(outputs, "last_hidden_state")
            else outputs[0]
        )
        if hidden.ndim != 3 or hidden.shape[:2] != input_ids.shape:
            raise RuntimeError(
                "backbone returned an unexpected last_hidden_state shape"
            )
        return hidden[:, -1]

    def _next_logits(self, input_ids: torch.Tensor) -> torch.Tensor:
        last = self._backbone_last(input_ids)
        logits = self._call_module(
            self.lm_head, last, dtype=self.lm_head.weight.dtype
        ).float()
        if logits.shape != (1, self.lm_head.weight.shape[0]):
            raise RuntimeError("lm_head returned an unexpected next-token shape")
        if not bool(torch.isfinite(logits).all()):
            raise RuntimeError("next-token logits contain NaN or Inf")
        return logits[0]

    @staticmethod
    def _call_module(
        module: Any, value: torch.Tensor, *, dtype: torch.dtype
    ) -> torch.Tensor:
        prepared = value.to(dtype=dtype)
        if getattr(module, "_hf_hook", None) is None:
            parameter = next(module.parameters(), None)
            if parameter is not None:
                if parameter.device.type == "meta":
                    raise RuntimeError(
                        f"{type(module).__name__} has meta parameters without an "
                        "Accelerate execution hook"
                    )
                prepared = prepared.to(parameter.device)
        return module(prepared)

    def _unembed_readout(self, residual: torch.Tensor) -> torch.Tensor:
        if self.final_norm is None:
            return self.lens_model.unembed(residual)
        dtype = self.lm_head.weight.dtype
        normalized = self._call_module(self.final_norm, residual, dtype=dtype)
        logits = self._call_module(self.lm_head, normalized, dtype=dtype)
        softcap = getattr(self.lens_model, "_logit_softcap", None)
        if softcap is not None:
            logits = softcap * torch.tanh(logits / softcap)
        return logits

    def selected_readout(
        self, residual: torch.Tensor, token_ids: Sequence[int]
    ) -> torch.Tensor:
        """Project residuals onto selected vocabulary logits without a full head."""
        if not isinstance(residual, torch.Tensor) or residual.ndim != 2:
            raise ValidationError("residual must have shape [positions, d_model]")
        if residual.shape[1] != self.lm_head.weight.shape[1]:
            raise ValidationError(
                f"residual width {residual.shape[1]} does not match lm_head width "
                f"{self.lm_head.weight.shape[1]}"
            )
        if residual.shape[0] == 0:
            raise ValidationError("residual must contain at least one position")
        if not bool(torch.isfinite(residual).all()):
            raise RuntimeError("selected readout residual contains NaN or Inf")

        key = tuple(token_ids)
        if not key:
            raise ValidationError("token_ids must contain at least one token ID")
        if any(
            isinstance(token_id, bool) or not isinstance(token_id, int)
            for token_id in key
        ):
            raise ValidationError("token_ids must contain only integers")
        vocab_size = self.lm_head.weight.shape[0]
        invalid = sorted(
            {token_id for token_id in key if not 0 <= token_id < vocab_size}
        )
        if invalid:
            raise ValidationError(
                f"token IDs {invalid} are outside the vocabulary range "
                f"[0, {vocab_size - 1}]"
            )
        with self._selected_parameter_lock:
            parameters = self._selected_parameter_cache.get(key)
            if parameters is not None:
                self._selected_parameter_cache.move_to_end(key)
        if parameters is None:
            parameters = unembedding_parameters(self.lm_head, key)
            with self._selected_parameter_lock:
                self._selected_parameter_cache[key] = parameters
                self._selected_parameter_cache.move_to_end(key)
                while len(self._selected_parameter_cache) > 16:
                    self._selected_parameter_cache.popitem(last=False)
        rows, bias = parameters
        dtype = self.lm_head.weight.dtype
        softcap = getattr(self.lens_model, "_logit_softcap", None)
        softcap_value: float | None = None
        if softcap is not None:
            try:
                softcap_value = float(softcap)
            except (TypeError, ValueError) as exc:
                raise RuntimeError(
                    "logit softcap must be a finite positive number"
                ) from exc
            if not math.isfinite(softcap_value) or softcap_value <= 0:
                raise RuntimeError("logit softcap must be a finite positive number")

        outputs: list[torch.Tensor] = []
        chunk_size = 16
        for start in range(0, residual.shape[0], chunk_size):
            chunk = residual[start : start + chunk_size]
            normalized = (
                self._call_module(self.final_norm, chunk, dtype=dtype)
                if self.final_norm is not None
                else chunk.to(dtype=dtype)
            )
            normalized = normalized.detach().to(device="cpu", dtype=torch.float32)
            if not bool(torch.isfinite(normalized).all()):
                raise RuntimeError("selected readout normalization contains NaN or Inf")
            logits = normalized @ rows.T
            if bias is not None:
                logits = logits + bias
            if softcap_value is not None:
                logits = softcap_value * torch.tanh(logits / softcap_value)
            if not bool(torch.isfinite(logits).all()):
                raise RuntimeError("selected readout logits contain NaN or Inf")
            outputs.append(logits)
        return torch.cat(outputs, dim=0)

    @staticmethod
    def _sample_token(
        logits: torch.Tensor, spec: GenerationSpec, rng: torch.Generator
    ) -> int:
        if not spec.sampling:
            return int(torch.argmax(logits).item())
        values = logits.detach().float().cpu() / spec.temperature
        probabilities = torch.softmax(values, dim=-1)
        if spec.top_p < 1.0:
            sorted_probs, sorted_ids = torch.sort(probabilities, descending=True)
            cumulative = sorted_probs.cumsum(dim=-1)
            remove = cumulative - sorted_probs >= spec.top_p
            sorted_probs = sorted_probs.masked_fill(remove, 0)
            sorted_probs /= sorted_probs.sum()
            selected = torch.multinomial(sorted_probs, 1, generator=rng)
            return int(sorted_ids[selected].item())
        return int(torch.multinomial(probabilities, 1, generator=rng).item())

    def _eos_ids(self) -> set[int]:
        raw = getattr(self.hf_model.generation_config, "eos_token_id", None)
        if raw is None:
            raw = getattr(self.tokenizer, "eos_token_id", None)
        if raw is None:
            return set()
        if isinstance(raw, int):
            return {raw}
        return {int(token_id) for token_id in raw}

    def _direction_map(
        self, intervention: InterventionSpec | None
    ) -> dict[int, tuple[torch.Tensor, torch.Tensor | None]]:
        if intervention is None:
            return {}
        return {
            layer: self.directions.combined(intervention, layer)
            for layer in intervention.layers
        }

    def _generate(
        self,
        prompt_ids: torch.Tensor,
        generation: GenerationSpec,
        intervention: InterventionSpec | None,
        *,
        cancel_event: threading.Event | None,
        progress: ProgressCallback | None,
        condition: str,
    ) -> torch.Tensor:
        _cancelled(cancel_event)
        original_prompt_length = prompt_ids.shape[1]
        context = ForwardContext(
            original_prompt_length=original_prompt_length,
            bos_token_id=getattr(self.tokenizer, "bos_token_id", None),
            apply_to_generated=(
                intervention.apply_to_generated if intervention is not None else False
            ),
        )
        _progress(progress, "directions", condition=condition)
        directions = self._direction_map(intervention)
        sequence = prompt_ids.clone()
        rng = torch.Generator(device="cpu")
        rng.manual_seed(generation.seed)
        eos_ids = self._eos_ids()
        with HookSession(
            self.layers,
            context,
            intervention=intervention,
            directions=directions,
            cancel_event=cancel_event,
        ):
            for index in range(generation.max_new_tokens):
                _cancelled(cancel_event)
                context.set_input_ids(sequence)
                logits = self._next_logits(sequence)
                token_id = self._sample_token(logits, generation, rng)
                next_token = torch.tensor(
                    [[token_id]], device=sequence.device, dtype=sequence.dtype
                )
                sequence = torch.cat((sequence, next_token), dim=1)
                _progress(
                    progress,
                    "generate",
                    condition=condition,
                    current=index + 1,
                    total=generation.max_new_tokens,
                    token_id=token_id,
                )
                if token_id in eos_ids:
                    break
        return sequence

    def _top_readout(
        self, residual: torch.Tensor, top_k: int
    ) -> list[list[dict[str, Any]]]:
        rows: list[list[dict[str, Any]]] = []
        chunk_size = 16
        for start in range(0, residual.shape[0], chunk_size):
            logits = self._unembed_readout(residual[start : start + chunk_size]).float()
            if not bool(torch.isfinite(logits).all()):
                raise RuntimeError("readout logits contain NaN or Inf")
            log_probs = torch.log_softmax(logits, dim=-1)
            top_logits, top_ids = torch.topk(logits, top_k, dim=-1)
            top_probs = log_probs.gather(-1, top_ids).exp()
            ids_cpu = top_ids.detach().cpu().tolist()
            logits_cpu = top_logits.detach().cpu().tolist()
            probs_cpu = top_probs.detach().cpu().tolist()
            for token_ids, token_logits, token_probs in zip(
                ids_cpu, logits_cpu, probs_cpu, strict=True
            ):
                rows.append(
                    [
                        {
                            "id": int(token_id),
                            "text": _decode_token(self.tokenizer, int(token_id)),
                            "probability": float(probability),
                            "logit": float(logit),
                        }
                        for token_id, probability, logit in zip(
                            token_ids, token_probs, token_logits, strict=True
                        )
                    ]
                )
        return rows

    def _teacher_forced_readouts(
        self,
        sequence: torch.Tensor,
        request: InferenceRequest,
        intervention: InterventionSpec | None,
        *,
        cancel_event: threading.Event | None,
        progress: ProgressCallback | None,
        condition: str,
        prompt_length: int,
    ) -> dict[str, Any]:
        _cancelled(cancel_event)
        context = ForwardContext(
            original_prompt_length=prompt_length,
            bos_token_id=getattr(self.tokenizer, "bos_token_id", None),
            apply_to_generated=(
                intervention.apply_to_generated if intervention is not None else False
            ),
            input_ids=sequence,
        )
        directions = self._direction_map(intervention)
        _progress(progress, "capture", condition=condition)
        with HookSession(
            self.layers,
            context,
            intervention=intervention,
            directions=directions,
            capture_layers=request.layers,
            cancel_event=cancel_event,
        ) as hooks:
            self._backbone_last(sequence)
            captures = dict(hooks.captures)

        result: dict[str, Any] = {
            "layers": list(request.layers),
            "jacobian": {},
            "logit": {},
        }
        for current, layer in enumerate(request.layers, start=1):
            _cancelled(cancel_event)
            _progress(
                progress,
                "readout",
                condition=condition,
                current=current,
                total=len(request.layers),
                layer=layer,
            )
            residual = captures[layer][0]
            result["logit"][str(layer)] = self._top_readout(residual, request.top_k)
            if layer in SOURCE_LAYERS:
                jacobian = self.bundle.lens.jacobians[layer]
                transported = residual.float().cpu() @ jacobian.float().cpu().T
                result["jacobian"][str(layer)] = self._top_readout(
                    transported, request.top_k
                )
        return result

    def _tokens(
        self, sequence: torch.Tensor, *, prompt_length: int
    ) -> list[dict[str, Any]]:
        bos_id = getattr(self.tokenizer, "bos_token_id", None)
        return [
            {
                "position": position,
                "id": int(token_id),
                "text": _decode_token(self.tokenizer, int(token_id)),
                "is_generated": position >= prompt_length,
                "is_bos": bos_id is not None and int(token_id) == bos_id,
            }
            for position, token_id in enumerate(sequence[0].detach().cpu().tolist())
        ]

    def _condition(
        self,
        prompt_ids: torch.Tensor,
        request: InferenceRequest,
        intervention: InterventionSpec | None,
        *,
        cancel_event: threading.Event | None,
        progress: ProgressCallback | None,
        name: str,
    ) -> dict[str, Any]:
        started = time.monotonic()
        sequence = self._generate(
            prompt_ids,
            request.generation,
            intervention,
            cancel_event=cancel_event,
            progress=progress,
            condition=name,
        )
        readouts = self._teacher_forced_readouts(
            sequence,
            request,
            intervention,
            cancel_event=cancel_event,
            progress=progress,
            condition=name,
            prompt_length=prompt_ids.shape[1],
        )
        full_ids = [int(token_id) for token_id in sequence[0].detach().cpu().tolist()]
        generated = full_ids[prompt_ids.shape[1] :]
        stopped = bool(generated and generated[-1] in self._eos_ids())
        completion_ids = generated[:-1] if stopped else generated
        return {
            "name": name,
            "completion": _decode_sequence(self.tokenizer, completion_ids),
            "completion_token_ids": completion_ids,
            "stop_token_id": generated[-1] if stopped else None,
            "finish_reason": "stop" if stopped else "length",
            "tokens": self._tokens(sequence, prompt_length=prompt_ids.shape[1]),
            "readouts": readouts,
            "elapsed_seconds": time.monotonic() - started,
            "finite": True,
        }

    def _hook_count(self) -> int:
        return sum(len(layer._forward_hooks) for layer in self.layers)

    def _memory(self) -> dict[str, Any]:
        if not torch.cuda.is_available():
            return {"available": False}
        return {
            "available": True,
            "allocated_bytes": int(torch.cuda.memory_allocated()),
            "reserved_bytes": int(torch.cuda.memory_reserved()),
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
            "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
        }

    def _run_provenance(
        self,
        request: InferenceRequest,
        prompt_ids: Sequence[int],
        formatted_prompt: str,
        *,
        timestamp: str,
        elapsed: float,
    ) -> dict[str, Any]:
        intervention = request.intervention
        intervention_value: dict[str, Any] | None = None
        if intervention is not None:
            intervention_value = {
                "mode": intervention.mode,
                "lens_type": intervention.lens_type,
                "layers": list(intervention.layers),
                "strength": intervention.strength,
                "position_scope": "all-non-bos",
                "apply_to_generated": intervention.apply_to_generated,
                "source_tokens": [
                    {
                        "id": token_id,
                        "text": _decode_token(self.tokenizer, token_id),
                    }
                    for token_id in intervention.source_token_ids
                ],
                "target_token": (
                    {
                        "id": intervention.target_token_id,
                        "text": _decode_token(
                            self.tokenizer, intervention.target_token_id
                        ),
                    }
                    if intervention.target_token_id is not None
                    else None
                ),
            }
        return {
            "disclosure": PILOT_DISCLOSURE,
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "scientific_runtime": self.bundle.metadata["runtime"],
            "mamba_backend": self.loaded.mamba_backend,
            "lens_sha256": LENS_SHA256,
            "lens_acceptance_tier": self.bundle.metadata["acceptance"]["tier"],
            "lens_prompt_count": self.bundle.metadata["n_prompts"],
            "fit_source_sha256": self.bundle.metadata["adaptation_source_sha256"],
            "live_application_source_sha256": self.bundle.application_source_sha256,
            "neuronpedia_reference_commit": NEURONPEDIA_COMMIT,
            "formatted_prompt_token_ids": list(prompt_ids),
            "formatted_prompt_token_sha256": prompt_token_hash(prompt_ids),
            "formatted_prompt": formatted_prompt,
            "prompt_format": "raw" if request.prompt is not None else "chat",
            "chat_template": (
                {
                    "source": "pinned-tokenizer",
                    "add_generation_prompt": True,
                    "enable_thinking": request.enable_thinking,
                }
                if request.messages is not None
                else None
            ),
            "readout_layers": list(request.layers),
            "top_k": request.top_k,
            "intervention": intervention_value,
            "generation": {
                "algorithm": "full-prefix-use-cache-false",
                "max_new_tokens": request.generation.max_new_tokens,
                "sampling": request.generation.sampling,
                "temperature": request.generation.temperature,
                "top_p": request.generation.top_p,
                "seed": request.generation.seed,
            },
            "timestamp": timestamp,
            "elapsed_seconds": elapsed,
            "memory": self._memory(),
        }

    @torch.inference_mode()
    def run(
        self,
        request: InferenceRequest,
        *,
        paired: bool,
        cancel_event: threading.Event | None = None,
        progress: ProgressCallback | None = None,
    ) -> dict[str, Any]:
        if paired and request.intervention is None:
            raise ValidationError("paired run requires an intervention")
        if not paired and request.intervention is not None:
            raise ValidationError("baseline run cannot contain an intervention")
        started = time.monotonic()
        timestamp = datetime.now(timezone.utc).isoformat()
        run_id = uuid4().hex
        hooks_before = self._hook_count()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        _progress(progress, "tokenize")
        prompt_ids, formatted_prompt = self._tokenize(request)
        clean = self._condition(
            prompt_ids,
            request,
            None,
            cancel_event=cancel_event,
            progress=progress,
            name="clean",
        )
        intervened = (
            self._condition(
                prompt_ids,
                request,
                request.intervention,
                cancel_event=cancel_event,
                progress=progress,
                name="intervened",
            )
            if paired
            else None
        )
        _cancelled(cancel_event)
        elapsed = time.monotonic() - started
        prompt_list = [
            int(token_id) for token_id in prompt_ids[0].detach().cpu().tolist()
        ]
        hooks_after = self._hook_count()
        if hooks_after != hooks_before:
            raise RuntimeError(
                f"hook leak detected: started with {hooks_before}, ended with {hooks_after}"
            )
        _progress(progress, "complete")
        return {
            "run_id": run_id,
            "disclosure": PILOT_DISCLOSURE,
            "status": "complete",
            "clean": clean,
            "intervened": intervened,
            "provenance": self._run_provenance(
                request,
                prompt_list,
                formatted_prompt,
                timestamp=timestamp,
                elapsed=elapsed,
            ),
            "diagnostics": {
                "hooks_before": hooks_before,
                "hooks_after": hooks_after,
                "direction_cache_entries": len(self.directions),
                "finite": True,
            },
        }
