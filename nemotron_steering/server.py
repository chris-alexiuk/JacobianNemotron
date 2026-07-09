"""Persistent LAN research server for the pinned Nano steering backend."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Header, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from nemotron_jlens.loading import load_nemotron
from nemotron_mood.requests import MoodRequestError, parse_mood_request
from nemotron_steering.constants import PILOT_DISCLOSURE
from nemotron_steering.errors import (
    InferenceBusy,
    InferenceCancelled,
    ValidationError,
)
from nemotron_steering.provenance import validate_lens_before_model_load
from nemotron_steering.requests import parse_request
from nemotron_steering.service import InferenceService
from nemotron_steering.validation import token_pieces

LOGGER = logging.getLogger("nemotron_steering")
MAX_REQUEST_BYTES = 64 * 1024


def create_app(
    service: InferenceService,
    *,
    static_dir: str | Path | None = None,
) -> FastAPI:
    app = FastAPI(
        title="Nano J-lens steering",
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )

    @app.middleware("http")
    async def bound_requests(request: Request, call_next: Any) -> Any:
        content_length = request.headers.get("content-length")
        if content_length is not None:
            try:
                too_large = int(content_length) > MAX_REQUEST_BYTES
            except ValueError:
                too_large = True
            if too_large:
                return JSONResponse(
                    status_code=413,
                    content={
                        "error": "request body is too large",
                        "disclosure": PILOT_DISCLOSURE,
                    },
                )
        return await call_next(request)

    @app.exception_handler(ValidationError)
    async def validation_error(_request: Request, exc: ValidationError) -> JSONResponse:
        return JSONResponse(
            status_code=422,
            content={
                "error": str(exc),
                "details": exc.details,
                "disclosure": PILOT_DISCLOSURE,
            },
        )

    @app.exception_handler(RequestValidationError)
    async def request_error(
        _request: Request, _exc: RequestValidationError
    ) -> JSONResponse:
        return JSONResponse(
            status_code=422,
            content={
                "error": "request must contain valid JSON",
                "disclosure": PILOT_DISCLOSURE,
            },
        )

    @app.exception_handler(InferenceBusy)
    async def busy_error(_request: Request, exc: InferenceBusy) -> JSONResponse:
        return JSONResponse(
            status_code=409,
            content={
                "error": str(exc),
                "status": "busy",
                "disclosure": PILOT_DISCLOSURE,
            },
        )

    @app.exception_handler(InferenceCancelled)
    async def cancelled_error(
        _request: Request, exc: InferenceCancelled
    ) -> JSONResponse:
        return JSONResponse(
            status_code=409,
            content={
                "error": str(exc),
                "status": "cancelled",
                "disclosure": PILOT_DISCLOSURE,
            },
        )

    @app.exception_handler(Exception)
    async def internal_error(request: Request, exc: Exception) -> JSONResponse:
        request_id = request.headers.get("x-request-id", "unknown")
        LOGGER.exception("inference request %s failed", request_id, exc_info=exc)
        return JSONResponse(
            status_code=500,
            content={
                "error": "inference failed; inspect server logs using the request ID",
                "request_id": request_id,
                "disclosure": PILOT_DISCLOSURE,
            },
        )

    @app.get("/health")
    def health() -> dict[str, Any]:
        return service.health()

    @app.get("/api/info")
    def info() -> dict[str, Any]:
        return service.backend.info

    @app.post("/api/tokenize")
    def tokenize(body: dict[str, Any]) -> dict[str, Any]:
        if set(body) != {"text"}:
            raise ValidationError("tokenize accepts exactly one text field")
        return {
            **token_pieces(service.backend.tokenizer, body.get("text")),
            "disclosure": PILOT_DISCLOSURE,
        }

    @app.post("/api/baseline")
    def baseline(
        body: dict[str, Any],
        x_request_id: str | None = Header(default=None),
    ) -> dict[str, Any]:
        request_id = service.request_id(x_request_id)
        parsed = parse_request(body, require_intervention=False)
        return service.run(parsed, paired=False, request_id=request_id)

    @app.post("/api/intervene")
    def intervene(
        body: dict[str, Any],
        x_request_id: str | None = Header(default=None),
    ) -> dict[str, Any]:
        request_id = service.request_id(x_request_id)
        parsed = parse_request(
            body, require_intervention=True, tokenizer=service.backend.tokenizer
        )
        return service.run(parsed, paired=True, request_id=request_id)

    @app.post("/api/mood")
    def mood(
        body: dict[str, Any],
        x_request_id: str | None = Header(default=None),
    ) -> dict[str, Any]:
        request_id = service.request_id(x_request_id)
        try:
            parsed = parse_mood_request(body)
        except MoodRequestError as exc:
            raise ValidationError(str(exc)) from exc
        return service.run_mood(parsed, request_id=request_id)

    @app.get("/api/status/{request_id}")
    def status(request_id: str) -> dict[str, Any]:
        request_id = service.request_id(request_id)
        value = service.status(request_id)
        if value is None:
            return {
                "request_id": request_id,
                "status": "unknown",
                "disclosure": PILOT_DISCLOSURE,
            }
        return value

    @app.post("/api/cancel/{request_id}")
    def cancel(request_id: str) -> dict[str, Any]:
        request_id = service.request_id(request_id)
        accepted = service.cancel(request_id)
        return {
            "request_id": request_id,
            "accepted": accepted,
            "status": "cancelling" if accepted else "not-active",
            "disclosure": PILOT_DISCLOSURE,
        }

    live_dir = Path(static_dir or Path(__file__).resolve().parents[1] / "steering_demo")
    if not live_dir.is_dir():
        raise RuntimeError(f"live browser assets do not exist: {live_dir}")
    app.mount("/", StaticFiles(directory=live_dir, html=True), name="steering-demo")
    return app


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--lens",
        default="artifacts/pilot100-h200-src7d4f5863.pt",
        help="immutable accepted-pilot lens path",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--device-map", choices=("auto", "cuda"), default="auto")
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument(
        "--log-level",
        choices=("critical", "error", "warning", "info", "debug"),
        default="info",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    if not 1 <= args.port <= 65535:
        raise SystemExit("--port must be in [1, 65535]")
    if args.host in {"0.0.0.0", "::"}:
        LOGGER.warning(
            "binding an unauthenticated research service to all interfaces; "
            "restrict port 8000 to a trusted LAN or Tailscale ACL"
        )

    # All scientific validation deliberately completes before model weights load.
    bundle = validate_lens_before_model_load(args.lens)
    loaded = load_nemotron(
        dtype="bfloat16",
        device_map=args.device_map,
        compile_blocks=False,
        disable_mamba_kernels=False,
        cache_dir=args.cache_dir,
    )
    from nemotron_steering.backend import SteeringBackend

    backend = SteeringBackend(loaded, bundle)
    service = InferenceService(backend)
    app = create_app(service)

    import uvicorn

    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        workers=1,
        log_level=args.log_level,
        access_log=True,
    )


if __name__ == "__main__":
    main()
