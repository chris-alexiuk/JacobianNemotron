"""Efficient Nano Jacobian-lens mood analysis on the persistent backend."""

from __future__ import annotations

import hashlib
import json
import threading
import time
from dataclasses import dataclass
from typing import Any

import torch

from nemotron_mood.analysis import (
    EMOTION_COLORS,
    NEUTRAL_REFERENCE_TEXTS,
    calibrate,
    sentence_spans,
)
from nemotron_mood.anchors import EMOTION_ANCHORS, EMOTIONS, anchor_token_ids
from nemotron_mood.requests import MoodRequest
from nemotron_steering.backend import SteeringBackend
from nemotron_steering.constants import PILOT_DISCLOSURE
from nemotron_steering.errors import InferenceCancelled, ValidationError
from nemotron_steering.interventions import ForwardContext, HookSession

JLENS_MOOD_COMMIT = "7b444c77c1c451068bf80c06a31aba5f4da23af7"
MOOD_SCHEMA_VERSION = "nemotron-jlens-mood/v1"
MAX_MOOD_TOKENS = 2048
CALIBRATION_MARGIN = 0.25


def _cancelled(cancel_event: threading.Event | None) -> None:
    if cancel_event is not None and cancel_event.is_set():
        raise InferenceCancelled("request cancelled")


def _progress(callback: Any, phase: str, **values: Any) -> None:
    if callback is not None:
        callback({"phase": phase, **values})


def _decode_token(tokenizer: Any, token_id: int) -> str:
    try:
        return tokenizer.decode(
            [token_id],
            clean_up_tokenization_spaces=False,
            skip_special_tokens=False,
        )
    except TypeError:
        return tokenizer.decode([token_id])


def _finite_float(value: torch.Tensor) -> float:
    result = float(value.detach().float().cpu().item())
    if not torch.isfinite(torch.tensor(result)):
        raise RuntimeError("mood analysis produced NaN or Inf")
    return result


@dataclass(frozen=True)
class _Calibration:
    baseline: torch.Tensor
    threshold: float
    reference_intensities: tuple[float, ...]
    identity: str


@dataclass(frozen=True)
class _Trace:
    token_ids: tuple[int, ...]
    tokens: tuple[str, ...]
    scores: torch.Tensor
    chunk_count: int


class MoodAnalyzer:
    """Compute eight-axis mood scores without materializing vocabulary logits."""

    def __init__(self, backend: SteeringBackend) -> None:
        self.backend = backend
        self.tokenizer = backend.tokenizer
        self.clusters: dict[str, tuple[int, ...]] = {}
        self.dropped: dict[str, tuple[str, ...]] = {}
        for emotion, words in EMOTION_ANCHORS.items():
            token_ids, dropped = anchor_token_ids(self.tokenizer, words)
            if not token_ids:
                raise ValidationError(
                    f"Nano tokenizer has no single-token anchors for {emotion!r}"
                )
            self.clusters[emotion] = tuple(token_ids)
            self.dropped[emotion] = tuple(dropped)

        owner: dict[int, str] = {}
        for emotion, token_ids in self.clusters.items():
            for token_id in token_ids:
                previous = owner.setdefault(token_id, emotion)
                if previous != emotion:
                    raise ValidationError(
                        "mood anchor token collision between "
                        f"{previous!r} and {emotion!r}: {token_id}"
                    )
        self.anchor_token_ids = tuple(sorted(owner))
        token_column = {
            token_id: index for index, token_id in enumerate(self.anchor_token_ids)
        }
        self.cluster_columns = {
            emotion: tuple(token_column[token_id] for token_id in token_ids)
            for emotion, token_ids in self.clusters.items()
        }
        anchor_payload = {
            emotion: {
                "words": EMOTION_ANCHORS[emotion],
                "token_ids": list(self.clusters[emotion]),
                "dropped_words": list(self.dropped[emotion]),
            }
            for emotion in EMOTIONS
        }
        self.anchor_definition_sha256 = hashlib.sha256(
            json.dumps(anchor_payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        self._calibrations: dict[tuple[tuple[int, ...], int], _Calibration] = {}
        self._selected_parity_max_abs: float | None = None

    def _tokenize_raw(self, text: str) -> tuple[int, ...]:
        token_ids = tuple(
            int(token_id)
            for token_id in self.tokenizer.encode(text, add_special_tokens=False)
        )
        if not token_ids:
            raise ValidationError("text produced no tokens")
        if len(token_ids) > MAX_MOOD_TOKENS:
            raise ValidationError(
                f"text has {len(token_ids)} tokens; maximum is {MAX_MOOD_TOKENS}"
            )
        return token_ids

    def _check_selected_parity(
        self, transported: torch.Tensor, selected: torch.Tensor
    ) -> None:
        if self._selected_parity_max_abs is not None:
            return
        full = (
            self.backend._unembed_readout(transported[:1])
            .float()
            .detach()
            .cpu()[:, self.anchor_token_ids]
        )
        difference = (full - selected[:1]).abs()
        max_abs = float(difference.max().item())
        reference = float(full.abs().max().item())
        tolerance = 0.25 + 0.01 * reference
        if not bool(torch.isfinite(difference).all()) or max_abs > tolerance:
            raise RuntimeError(
                "selected mood logits do not match the full unembedding "
                f"(max abs {max_abs:.6f}, tolerance {tolerance:.6f})"
            )
        self._selected_parity_max_abs = max_abs

    def _trace(
        self,
        text: str,
        *,
        layers: tuple[int, ...],
        chunk_tokens: int,
        cancel_event: threading.Event | None,
        progress: Any,
        purpose: str,
    ) -> _Trace:
        token_ids = self._tokenize_raw(text)
        spans = [
            (start, min(start + chunk_tokens, len(token_ids)))
            for start in range(0, len(token_ids), chunk_tokens)
        ]
        layer_parts: dict[int, list[torch.Tensor]] = {layer: [] for layer in layers}
        total_steps = len(spans) * len(layers)
        completed = 0

        for chunk_index, (start, end) in enumerate(spans, start=1):
            _cancelled(cancel_event)
            chunk_ids = torch.tensor(
                [token_ids[start:end]],
                dtype=torch.long,
                device=self.backend.input_device,
            )
            context = ForwardContext(
                original_prompt_length=chunk_ids.shape[1],
                bos_token_id=getattr(self.tokenizer, "bos_token_id", None),
                apply_to_generated=False,
                input_ids=chunk_ids,
            )
            _progress(
                progress,
                "mood-prefill",
                purpose=purpose,
                chunk=chunk_index,
                chunks=len(spans),
            )
            with HookSession(
                self.backend.layers,
                context,
                capture_layers=layers,
                cancel_event=cancel_event,
            ) as hooks:
                self.backend._backbone_last(chunk_ids)
                captures = dict(hooks.captures)
            missing = set(layers).difference(captures)
            if missing:
                raise RuntimeError(f"mood capture missed layers: {sorted(missing)}")

            for layer in layers:
                _cancelled(cancel_event)
                residual = captures[layer][0]
                jacobian = self.backend.bundle.lens.jacobians[layer]
                transported = residual.float().cpu() @ jacobian.float().cpu().T
                selected = self.backend.selected_readout(
                    transported, self.anchor_token_ids
                )
                self._check_selected_parity(transported, selected)
                axis_scores = torch.stack(
                    [
                        selected[:, self.cluster_columns[emotion]].mean(dim=-1)
                        for emotion in EMOTIONS
                    ],
                    dim=-1,
                )
                if not bool(torch.isfinite(axis_scores).all()):
                    raise RuntimeError("mood axis scores contain NaN or Inf")
                layer_parts[layer].append(axis_scores)
                completed += 1
                _progress(
                    progress,
                    "mood-readout",
                    purpose=purpose,
                    current=completed,
                    total=total_steps,
                    layer=layer,
                )

        per_layer = [torch.cat(layer_parts[layer], dim=0) for layer in layers]
        scores = torch.stack(per_layer, dim=0).mean(dim=0)
        pieces = tuple(_decode_token(self.tokenizer, token_id) for token_id in token_ids)
        return _Trace(token_ids, pieces, scores, len(spans))

    def _calibration(
        self,
        *,
        layers: tuple[int, ...],
        chunk_tokens: int,
        cancel_event: threading.Event | None,
        progress: Any,
    ) -> _Calibration:
        key = (layers, chunk_tokens)
        cached = self._calibrations.get(key)
        if cached is not None:
            _progress(progress, "mood-calibration", cached=True)
            return cached

        means: list[torch.Tensor] = []
        for index, text in enumerate(NEUTRAL_REFERENCE_TEXTS, start=1):
            _progress(
                progress,
                "mood-calibration",
                cached=False,
                current=index,
                total=len(NEUTRAL_REFERENCE_TEXTS),
            )
            trace = self._trace(
                text,
                layers=layers,
                chunk_tokens=chunk_tokens,
                cancel_event=cancel_event,
                progress=progress,
                purpose=f"neutral-reference-{index}",
            )
            means.append(trace.scores.mean(dim=0))
        raw = torch.stack(means)
        baseline = raw.mean(dim=0)
        calibrated, _shares = calibrate(raw, baseline)
        intensities = tuple(float(value) for value in calibrated.max(dim=-1).values)
        threshold = max(intensities) + CALIBRATION_MARGIN
        identity_payload = {
            "model_revision": self.backend.info["model"]["revision"],
            "lens_sha256": self.backend.info["lens"]["sha256"],
            "layers": list(layers),
            "chunk_tokens": chunk_tokens,
            "anchor_definition_sha256": self.anchor_definition_sha256,
            "neutral_reference_texts": NEUTRAL_REFERENCE_TEXTS,
            "aggregation": "uniform-position-and-layer-mean",
            "margin": CALIBRATION_MARGIN,
        }
        identity = hashlib.sha256(
            json.dumps(
                identity_payload, sort_keys=True, separators=(",", ":")
            ).encode()
        ).hexdigest()
        calibration = _Calibration(
            baseline=baseline,
            threshold=threshold,
            reference_intensities=intensities,
            identity=identity,
        )
        self._calibrations[key] = calibration
        return calibration

    @staticmethod
    def _mood_of(
        scores: torch.Tensor, baseline: torch.Tensor, threshold: float
    ) -> tuple[str, str, float, dict[str, float]]:
        calibrated, share_values = calibrate(scores.mean(dim=0), baseline)
        shares = {
            emotion: _finite_float(share)
            for emotion, share in zip(EMOTIONS, share_values, strict=True)
        }
        strongest = max(shares, key=shares.get)
        intensity = _finite_float(calibrated.max())
        mood = "neutral" if intensity < threshold else strongest
        return mood, strongest, intensity, shares

    def analyze(
        self,
        request: MoodRequest,
        *,
        cancel_event: threading.Event | None = None,
        progress: Any = None,
    ) -> dict[str, Any]:
        started = time.monotonic()
        hooks_before = self.backend._hook_count()
        if hooks_before != 0:
            raise RuntimeError("mood analysis started with residual hooks installed")

        calibration = self._calibration(
            layers=request.layers,
            chunk_tokens=request.chunk_tokens,
            cancel_event=cancel_event,
            progress=progress,
        )
        trace = self._trace(
            request.text,
            layers=request.layers,
            chunk_tokens=request.chunk_tokens,
            cancel_event=cancel_event,
            progress=progress,
            purpose="input",
        )
        mood, strongest, intensity, shares = self._mood_of(
            trace.scores, calibration.baseline, calibration.threshold
        )

        sentences: list[dict[str, Any]] = []
        for start, end in sentence_spans(list(trace.tokens)):
            sentence_mood, sentence_strongest, sentence_intensity, sentence_shares = (
                self._mood_of(
                    trace.scores[start:end],
                    calibration.baseline,
                    calibration.threshold,
                )
            )
            sentences.append(
                {
                    "start": start,
                    "end": end,
                    "text": "".join(trace.tokens[start:end]).strip(),
                    "mood": sentence_mood,
                    "strongest": sentence_strongest,
                    "intensity": sentence_intensity,
                    "shares": sentence_shares,
                }
            )

        tokens: list[dict[str, Any]] = []
        for position, (token_id, piece) in enumerate(
            zip(trace.token_ids, trace.tokens, strict=True)
        ):
            token_mood, token_strongest, token_intensity, token_shares = self._mood_of(
                trace.scores[position : position + 1],
                calibration.baseline,
                calibration.threshold,
            )
            tokens.append(
                {
                    "position": position,
                    "id": token_id,
                    "text": piece,
                    "mood": token_mood,
                    "strongest": token_strongest,
                    "intensity": token_intensity,
                    "shares": token_shares,
                }
            )

        hooks_after = self.backend._hook_count()
        if hooks_after != hooks_before:
            raise RuntimeError("mood analysis leaked residual hooks")
        info = self.backend.info
        return {
            "schema_version": MOOD_SCHEMA_VERSION,
            "status": "complete",
            "disclosure": PILOT_DISCLOSURE,
            "mood": mood,
            "strongest": strongest,
            "intensity": intensity,
            "threshold": calibration.threshold,
            "shares": shares,
            "sentences": sentences,
            "tokens": tokens,
            "layers": list(request.layers),
            "anchors": {
                emotion: {
                    "color": EMOTION_COLORS[emotion],
                    "token_ids": list(self.clusters[emotion]),
                    "dropped_words": list(self.dropped[emotion]),
                }
                for emotion in EMOTIONS
            },
            "calibration": {
                "id": calibration.identity,
                "method": "neutral-reference-envelope",
                "reference_count": len(NEUTRAL_REFERENCE_TEXTS),
                "reference_intensities": list(calibration.reference_intensities),
                "margin": CALIBRATION_MARGIN,
                "threshold": calibration.threshold,
                "baseline": {
                    emotion: _finite_float(value)
                    for emotion, value in zip(
                        EMOTIONS, calibration.baseline, strict=True
                    )
                },
            },
            "provenance": {
                "model_id": info["model"]["id"],
                "model_revision": info["model"]["revision"],
                "lens_sha256": info["lens"]["sha256"],
                "lens_prompt_count": info["lens"]["prompt_count"],
                "fit_source_sha256": info["lens"]["fit_source_sha256"],
                "live_application_source_sha256": info[
                    "live_application_source_sha256"
                ],
                "jlens_mood_commit": JLENS_MOOD_COMMIT,
                "prompt_format": "raw",
                "chat_template_applied": False,
                "aggregation": "uniform-position-and-layer-mean",
                "anchor_definition_sha256": self.anchor_definition_sha256,
            },
            "diagnostics": {
                "hooks_before": hooks_before,
                "hooks_after": hooks_after,
                "input_token_count": len(trace.token_ids),
                "chunk_count": trace.chunk_count,
                "selected_logit_parity_max_abs": self._selected_parity_max_abs,
            },
            "elapsed_seconds": time.monotonic() - started,
        }
