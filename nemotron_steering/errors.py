"""Service-domain exceptions that are safe to expose as concise API errors."""

from __future__ import annotations

from typing import Any


class SteeringError(Exception):
    """Base class for expected steering failures."""


class ValidationError(SteeringError):
    """A fail-closed request or artifact validation error."""

    def __init__(self, message: str, *, details: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.details = details or {}


class InferenceCancelled(SteeringError):
    """Raised between forwards when a request has been cancelled."""


class InferenceBusy(SteeringError):
    """Raised when the single process-wide inference slot is occupied."""
