"""Trace adapter base class and shared helpers.

Every concrete adapter (LangSmith, Langfuse, OTel, A2A, MCP, raw, DeepEval,
ACS) subclasses :class:`TraceAdapter` and implements :meth:`TraceAdapter.adapt`
to convert a backend-specific trace payload into a :class:`CanonicalTrace`.
The base class provides:

* :class:`AdapterError` — raised when a payload cannot be parsed.
* :meth:`TraceAdapter._normalise_span` — a tolerant dict -> :class:`TraceSpan`
  builder subclasses can reuse so span construction stays uniform.
* :meth:`TraceAdapter._finalise` — runs :func:`voyage_trace.protocol.normalise`
  (which fills ``dotted_order`` and enforces invariants) and returns the trace.
  Every adapter MUST call this at the end of ``adapt``.
* Module-level helpers :func:`_synthetic_id`, :func:`_now`, and
  :func:`_otel_status_code` that several adapters share.

Design rule: adapters depend only on :mod:`voyage_trace.types` and
:mod:`voyage_trace.protocol` — never on a backend SDK. They parse exported
JSON, not live API responses.
"""

from __future__ import annotations

import json
import uuid
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Any

from ..protocol import normalise
from ..types import (
    CanonicalTrace,
    OperationType,
    SourceProtocol,
    SpanStatus,
    TraceSpan,
)


class AdapterError(ValueError):
    """Raised when an adapter cannot parse its input into a canonical trace."""


def _synthetic_id(prefix: str) -> str:
    """Mint a short id for traces that arrive without one (``<prefix>-<8 hex>``)."""
    return f"{prefix}-{uuid.uuid4().hex[:8]}"


def _now() -> datetime:
    """Timezone-aware UTC ``now`` — the canonical fallback ``start_time``."""
    return datetime.now(timezone.utc)


def _otel_status_code(code: Any) -> SpanStatus:
    """Map an OTel span status code (int or string) onto :class:`SpanStatus`.

    OTel status codes: ``1`` / ``"OK"`` -> SUCCESS, ``2`` / ``"ERROR"`` ->
    FAILED. Unknown codes default to SUCCESS (matching the OTel spec's
    "unset" -> not-an-error convention).
    """
    code_s = str(code).upper()
    if code_s in ("ERROR", "2"):
        return SpanStatus.FAILED
    return SpanStatus.SUCCESS


class TraceAdapter(ABC):
    """Abstract base for all source-protocol trace adapters.

    Subclasses set the class attribute :attr:`source_protocol` and implement
    :meth:`adapt`. The shared helpers :meth:`_normalise_span` and
    :meth:`_finalise` keep span construction and invariant enforcement
    uniform across backends.
    """

    source_protocol: SourceProtocol = SourceProtocol.CUSTOM

    @abstractmethod
    def adapt(self, payload: "dict | list | str | bytes") -> CanonicalTrace:
        """Convert ``payload`` into a normalised :class:`CanonicalTrace`."""

    # -- shared helpers --------------------------------------------------- #
    @staticmethod
    def _decode(payload: Any) -> Any:
        """Decode a ``str``/``bytes`` JSON payload; pass everything else through."""
        if isinstance(payload, (str, bytes)):
            return json.loads(payload)
        return payload

    @staticmethod
    def _coerce_error(err: Any) -> str | None:
        """Coerce an error value into a string (or ``None``)."""
        if err is None:
            return None
        if isinstance(err, str):
            return err
        return str(err)

    @staticmethod
    def _parse_dt(value: Any) -> datetime | None:
        """Parse a datetime from a str / int / float / datetime.

        Strings use :func:`datetime.fromisoformat` with the ``Z`` suffix
        normalised to ``+00:00``. Numeric values are interpreted as unix
        timestamps with the unit (ns / us / ms / s) auto-detected by
        magnitude — matching how OTel exporters serialise ``_unix_nano``
        fields alongside RFC3339 strings.
        """
        if value is None or value == "":
            return None
        if isinstance(value, datetime):
            return value
        if isinstance(value, bool):  # bool is an int subclass — guard
            return None
        if isinstance(value, (int, float)):
            v = float(value)
            if v > 1e17:  # nanoseconds
                return datetime.fromtimestamp(v / 1e9, tz=timezone.utc)
            if v > 1e14:  # microseconds
                return datetime.fromtimestamp(v / 1e6, tz=timezone.utc)
            if v > 1e11:  # milliseconds
                return datetime.fromtimestamp(v / 1e3, tz=timezone.utc)
            return datetime.fromtimestamp(v, tz=timezone.utc)
        if isinstance(value, str):
            return datetime.fromisoformat(value.replace("Z", "+00:00"))
        return None

    def _normalise_span(
        self,
        raw: dict[str, Any],
        trace_id: str | None = None,
    ) -> TraceSpan:
        """Build a :class:`TraceSpan` from a canonical-ish dict.

        ``raw`` uses canonical field names (``trace_id``, ``span_id``,
        ``parent_span_id``, ``operation_type`` ...). Missing optional keys
        fall back to defaults; ``source_protocol`` defaults to this adapter's
        class attribute.

        Raises :class:`AdapterError` if ``span_id`` — or ``trace_id`` when no
        ``trace_id`` argument is supplied — is absent.
        """
        tid = trace_id or raw.get("trace_id")
        sid = raw.get("span_id")
        if not tid:
            raise AdapterError("span is missing trace_id")
        if not sid:
            raise AdapterError("span is missing span_id")

        op = raw.get("operation_type", OperationType.CHAT.value)
        status = raw.get("status", SpanStatus.SUCCESS.value)
        src = raw.get("source_protocol", self.source_protocol.value)

        now = _now()
        start = self._parse_dt(raw.get("start_time")) or now
        recorded = self._parse_dt(raw.get("recorded_at")) or now

        return TraceSpan(
            trace_id=tid,
            span_id=str(sid),
            parent_span_id=raw.get("parent_span_id"),
            dotted_order=raw.get("dotted_order", "") or "",
            session_id=raw.get("session_id", "") or "",
            agent_id=raw.get("agent_id", "") or "",
            agent_name=raw.get("agent_name", "") or "",
            agent_version=raw.get("agent_version", "") or "",
            operation_type=OperationType(op) if isinstance(op, str) else op,
            status=SpanStatus(status) if isinstance(status, str) else status,
            start_time=start,
            end_time=self._parse_dt(raw.get("end_time")),
            first_token_time=self._parse_dt(raw.get("first_token_time")),
            inputs=raw.get("inputs", {}) or {},
            outputs=raw.get("outputs"),
            error=self._coerce_error(raw.get("error")),
            metadata=raw.get("metadata", {}) or {},
            input_tokens=int(raw.get("input_tokens", 0) or 0),
            output_tokens=int(raw.get("output_tokens", 0) or 0),
            cost_usd=float(raw.get("cost_usd", 0.0) or 0.0),
            source_protocol=SourceProtocol(src) if isinstance(src, str) else src,
            recorded_at=recorded,
        )

    def _finalise(self, trace: CanonicalTrace) -> CanonicalTrace:
        """Run :func:`protocol.normalise` and return the trace.

        Every adapter MUST call this at the end of :meth:`adapt` so the
        emitted trace satisfies :func:`enforce_invariants`.
        """
        return normalise(trace)
