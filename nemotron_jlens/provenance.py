"""Stable fingerprints for the local scientific implementation."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

import jlens

_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


def require_sha256(value: object, *, field: str) -> str:
    """Return a canonical SHA-256 string or reject malformed provenance."""
    if not isinstance(value, str) or _SHA256_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{field} must be a 64-character lowercase SHA-256")
    return value


def adaptation_source_sha256() -> str:
    """Hash the Python sources that fit, apply, validate, and export a lens."""
    package_roots = [
        Path(__file__).resolve().parent,
        Path(jlens.__file__).resolve().parent,
    ]
    files = sorted(
        path
        for root in package_roots
        for path in root.rglob("*")
        if path.is_file()
        and (
            path.suffix == ".py"
            or path.name in {"slice_vis.html", "blackmail.json"}
        )
        and "__pycache__" not in path.parts
    )
    digest = hashlib.sha256()
    for path in files:
        root = next(root for root in package_roots if path.is_relative_to(root))
        package = root.name
        relative = path.relative_to(root).as_posix()
        digest.update(f"{package}/{relative}\0".encode())
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()
