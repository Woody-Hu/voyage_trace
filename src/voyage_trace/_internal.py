"""Shared private helpers for voyage_trace.

Tiny, dependency-free utilities used by several modules so we don't re-declare
the same one-liner in five places. Everything here is implementation detail and
MUST stay underscore-prefixed / private — the public surface lives in
:mod:`voyage_trace.types`, :mod:`voyage_trace.protocol`, etc.

Importing this module imports only the Python standard library.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone


def utcnow() -> datetime:
    """Return a timezone-aware UTC ``now``.

    Centralised so we never accidentally store naive datetimes — naive
    datetimes are a recurring source of bugs when comparing spans recorded in
    different timezones.
    """
    return datetime.now(timezone.utc)


def new_id(prefix: str) -> str:
    """Mint a short, prefix-tagged random id (``<prefix>-<12 hex chars>``)."""
    return f"{prefix}-{uuid.uuid4().hex[:12]}"


def dt_to_str(dt: datetime | None) -> str | None:
    """Serialise a datetime to its ISO-8601 string, or ``None``."""
    if dt is None:
        return None
    return dt.isoformat()


def dt_from_str(s: str | None) -> datetime | None:
    """Parse an ISO-8601 string back into a datetime, or ``None``.

    Tolerates the trailing ``Z`` suffix used by some exporters by normalising
    it to ``+00:00`` before :func:`datetime.fromisoformat`.
    """
    if not s:
        return None
    return datetime.fromisoformat(s.replace("Z", "+00:00"))
