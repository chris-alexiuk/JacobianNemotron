"""Thread-safe lifecycle and progress state for the single inference worker."""

from __future__ import annotations

import re
import threading
from collections import OrderedDict
from collections.abc import Callable
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from nemotron_steering.backend import SteeringBackend
from nemotron_steering.constants import PILOT_DISCLOSURE
from nemotron_steering.errors import InferenceBusy, InferenceCancelled, ValidationError
from nemotron_steering.requests import InferenceRequest

if TYPE_CHECKING:
    from nemotron_mood.backend import MoodAnalyzer
    from nemotron_mood.requests import MoodRequest

_REQUEST_ID = re.compile(r"^[A-Za-z0-9_-]{8,80}$")


class InferenceService:
    """Serialize hooks and retain bounded status snapshots for polling."""

    def __init__(self, backend: SteeringBackend, *, status_limit: int = 64) -> None:
        self.backend = backend
        self._inference_lock = threading.Lock()
        self._state_lock = threading.Lock()
        self._active_id: str | None = None
        self._active_cancel: threading.Event | None = None
        self._statuses: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self._status_limit = status_limit
        self._mood_analyzer: MoodAnalyzer | None = None

    @staticmethod
    def request_id(value: str | None) -> str:
        if value is None or not value:
            return uuid4().hex
        if _REQUEST_ID.fullmatch(value) is None:
            raise ValidationError(
                "X-Request-ID must contain 8-80 letters, digits, underscores, or dashes"
            )
        return value

    def _set_status(self, request_id: str, **values: Any) -> None:
        with self._state_lock:
            previous = self._statuses.get(request_id, {})
            self._statuses[request_id] = {
                "request_id": request_id,
                "disclosure": PILOT_DISCLOSURE,
                **previous,
                **values,
            }
            self._statuses.move_to_end(request_id)
            while len(self._statuses) > self._status_limit:
                self._statuses.popitem(last=False)

    def status(self, request_id: str) -> dict[str, Any] | None:
        with self._state_lock:
            value = self._statuses.get(request_id)
            return dict(value) if value is not None else None

    def health(self) -> dict[str, Any]:
        with self._state_lock:
            active_id = self._active_id
        return {
            "status": "ready",
            "model_loaded": True,
            "busy": active_id is not None,
            "active_request_id": active_id,
            "disclosure": PILOT_DISCLOSURE,
        }

    def cancel(self, request_id: str) -> bool:
        with self._state_lock:
            if request_id != self._active_id or self._active_cancel is None:
                return False
            self._active_cancel.set()
        self._set_status(request_id, status="cancelling", phase="cancelling")
        return True

    def _run_operation(
        self,
        operation: Callable[[threading.Event, Callable[[dict[str, Any]], None]], dict[str, Any]],
        *,
        request_id: str,
    ) -> dict[str, Any]:
        if not self._inference_lock.acquire(blocking=False):
            with self._state_lock:
                active_id = self._active_id
            self._set_status(
                request_id,
                status="busy",
                phase="busy",
                active_request_id=active_id,
            )
            raise InferenceBusy("another inference request is active")

        cancel_event = threading.Event()
        with self._state_lock:
            self._active_id = request_id
            self._active_cancel = cancel_event
        self._set_status(request_id, status="running", phase="starting")

        def progress(update: dict[str, Any]) -> None:
            self._set_status(request_id, status="running", **update)

        try:
            result = operation(cancel_event, progress)
            result["request_id"] = request_id
            self._set_status(request_id, status="complete", phase="complete")
            return result
        except InferenceCancelled:
            self._set_status(request_id, status="cancelled", phase="cancelled")
            raise
        except BaseException:
            self._set_status(request_id, status="error", phase="error")
            raise
        finally:
            with self._state_lock:
                self._active_id = None
                self._active_cancel = None
            self._inference_lock.release()

    def run(
        self, request: InferenceRequest, *, paired: bool, request_id: str
    ) -> dict[str, Any]:
        return self._run_operation(
            lambda cancel_event, progress: self.backend.run(
                request,
                paired=paired,
                cancel_event=cancel_event,
                progress=progress,
            ),
            request_id=request_id,
        )

    def run_mood(
        self, request: MoodRequest, *, request_id: str
    ) -> dict[str, Any]:
        def analyze(
            cancel_event: threading.Event,
            progress: Callable[[dict[str, Any]], None],
        ) -> dict[str, Any]:
            if self._mood_analyzer is None:
                from nemotron_mood.backend import MoodAnalyzer

                self._mood_analyzer = MoodAnalyzer(self.backend)
            return self._mood_analyzer.analyze(
                request, cancel_event=cancel_event, progress=progress
            )

        return self._run_operation(analyze, request_id=request_id)
