"""The voyage_trace trace protocol.

This module is the *contract* between the outside world (any observability
backend) and the inside of voyage_trace. It defines:

1. How a :class:`~voyage_trace.types.CanonicalTrace` / :class:`TraceSpan` is
   serialised to/from JSON (the on-the-wire and on-disk format).
2. How ``dotted_order`` strings are computed and validated — the single
   sortable field that encodes a span's position in the execution tree
   (borrowed from LangSmith, see ``docs/protocol.md``).
3. A small set of protocol-level invariants that every adapter MUST satisfy
   before handing a trace to downstream stages.

Design rule: this module imports only :mod:`voyage_trace.types` and the
standard library. It must stay dependency-free so the protocol can be
published/versioned independently of deepagents.
"""

from __future__ import annotations

import json
import re
import uuid
from datetime import datetime
from typing import Any

from .types import (
    CanonicalTrace,
    OperationType,
    SourceProtocol,
    SpanStatus,
    TraceSpan,
)

# A dotted-order segment: ``20250727T120000Z0000000000000001``
# (compact ISO-8601 UTC timestamp + a zero-padded suffix). We accept any
# hex/decimal suffix so foreign traces (LangSmith uses a UUID) survive
# round-trip without rewrites.
_DOTTED_SEGMENT = re.compile(r"^\d{8}T\d{6}Z[0-9A-Za-z]+$")
_DOTTED_SPLIT = re.compile(r"\.")


# --------------------------------------------------------------------------- #
# dotted_order helpers
# --------------------------------------------------------------------------- #
def format_dotted_timestamp(dt: datetime) -> str:
    """Render a datetime as the compact UTC prefix of a dotted_order segment.

    Naive datetimes are treated as UTC (matching LangSmith's export behaviour);
    aware datetimes are converted to UTC first.
    """
    from datetime import timezone

    if dt.tzinfo is None:
        utc = dt
    else:
        utc = dt.astimezone(timezone.utc).replace(tzinfo=None)
    return utc.strftime("%Y%m%dT%H%M%SZ")


def make_dotted_order(start_time: datetime, span_id: str, parent_order: str | None) -> str:
    """Build a dotted_order string for a span.

    Format: ``<parent_order>.<startZ><suffix>`` where ``<startZ>`` is the
    compact-UTC start timestamp and ``<suffix>`` is derived from ``span_id``
    (deterministic, so the same span always produces the same order).

    If ``parent_order`` is ``None`` this is a root span and the segment is
    returned alone.
    """
    # Use a stable, deterministic suffix derived from the span id so that
    # replay/regeneration produces identical dotted_orders.
    suffix = span_id.replace("-", "")[:24].ljust(24, "0") if span_id else uuid.uuid4().hex[:24]
    segment = f"{format_dotted_timestamp(start_time)}{suffix}"
    if parent_order:
        return f"{parent_order}.{segment}"
    return segment


def validate_dotted_order(order: str) -> bool:
    """Return ``True`` iff every segment of ``order`` is well-formed."""
    if not order:
        return False
    return all(_DOTTED_SEGMENT.match(seg) for seg in _DOTTED_SPLIT.split(order))


def depth_of(order: str) -> int:
    """Tree depth of a span given its dotted_order (1 = root)."""
    return len(_DOTTED_SPLIT.split(order)) if order else 0


# --------------------------------------------------------------------------- #
# JSON serialisation
# --------------------------------------------------------------------------- #
def _dt_to_str(dt: datetime | None) -> str | None:
    if dt is None:
        return None
    return dt.isoformat()


def _str_to_dt(s: str | None) -> datetime | None:
    if not s:
        return None
    # ``datetime.fromisoformat`` handles ``+00:00`` on 3.11+; for the ``Z``
    # suffix used in some exports, normalise first.
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def span_to_dict(span: TraceSpan) -> dict[str, Any]:
    """Serialise a :class:`TraceSpan` to a JSON-safe dict."""
    return {
        "trace_id": span.trace_id,
        "span_id": span.span_id,
        "parent_span_id": span.parent_span_id,
        "dotted_order": span.dotted_order,
        "session_id": span.session_id,
        "agent_id": span.agent_id,
        "agent_name": span.agent_name,
        "agent_version": span.agent_version,
        "operation_type": span.operation_type.value,
        "status": span.status.value,
        "start_time": _dt_to_str(span.start_time),
        "end_time": _dt_to_str(span.end_time),
        "first_token_time": _dt_to_str(span.first_token_time),
        "inputs": span.inputs,
        "outputs": span.outputs,
        "error": span.error,
        "metadata": span.metadata,
        "input_tokens": span.input_tokens,
        "output_tokens": span.output_tokens,
        "cost_usd": span.cost_usd,
        "source_protocol": span.source_protocol.value,
        "recorded_at": _dt_to_str(span.recorded_at),
    }


def span_from_dict(d: dict[str, Any]) -> TraceSpan:
    """Deserialise a dict back into a :class:`TraceSpan`.

    Unknown keys are ignored; missing optional keys fall back to defaults.
    """
    op = d.get("operation_type", "chat")
    status = d.get("status", "success")
    src = d.get("source_protocol", "custom")
    return TraceSpan(
        trace_id=d["trace_id"],
        span_id=d["span_id"],
        parent_span_id=d.get("parent_span_id"),
        dotted_order=d.get("dotted_order", ""),
        session_id=d.get("session_id", ""),
        agent_id=d.get("agent_id", ""),
        agent_name=d.get("agent_name", ""),
        agent_version=d.get("agent_version", ""),
        operation_type=OperationType(op) if isinstance(op, str) else op,
        status=SpanStatus(status) if isinstance(status, str) else status,
        start_time=_str_to_dt(d.get("start_time")) or datetime.fromisoformat("1970-01-01T00:00:00+00:00"),
        end_time=_str_to_dt(d.get("end_time")),
        first_token_time=_str_to_dt(d.get("first_token_time")),
        inputs=d.get("inputs", {}) or {},
        outputs=d.get("outputs"),
        error=d.get("error"),
        metadata=d.get("metadata", {}) or {},
        input_tokens=int(d.get("input_tokens", 0) or 0),
        output_tokens=int(d.get("output_tokens", 0) or 0),
        cost_usd=float(d.get("cost_usd", 0.0) or 0.0),
        source_protocol=SourceProtocol(src) if isinstance(src, str) else src,
        recorded_at=_str_to_dt(d.get("recorded_at")) or datetime.fromisoformat("1970-01-01T00:00:00+00:00"),
    )


def trace_to_dict(trace: CanonicalTrace) -> dict[str, Any]:
    """Serialise a :class:`CanonicalTrace` (with all spans) to a JSON-safe dict."""
    return {
        "trace_id": trace.trace_id,
        "agent_id": trace.agent_id,
        "agent_name": trace.agent_name,
        "agent_version": trace.agent_version,
        "session_id": trace.session_id,
        "source_protocol": trace.source_protocol.value,
        "spans": [span_to_dict(s) for s in trace.spans],
        "metadata": trace.metadata,
    }


def trace_from_dict(d: dict[str, Any]) -> CanonicalTrace:
    """Deserialise a dict back into a :class:`CanonicalTrace`."""
    src = d.get("source_protocol", "custom")
    return CanonicalTrace(
        trace_id=d["trace_id"],
        agent_id=d.get("agent_id", ""),
        agent_name=d.get("agent_name", ""),
        agent_version=d.get("agent_version", ""),
        session_id=d.get("session_id", ""),
        source_protocol=SourceProtocol(src) if isinstance(src, str) else src,
        spans=[span_from_dict(s) for s in d.get("spans", [])],
        metadata=d.get("metadata", {}) or {},
    )


def trace_to_json(trace: CanonicalTrace) -> str:
    """Serialise a trace to a compact JSON string."""
    return json.dumps(trace_to_dict(trace), sort_keys=True, separators=(",", ":"))


def trace_from_json(text: str | bytes) -> CanonicalTrace:
    """Parse a JSON string/bytes into a :class:`CanonicalTrace`."""
    return trace_from_dict(json.loads(text))


# --------------------------------------------------------------------------- #
# Protocol invariants — enforced on every adapted trace before it flows on.
# --------------------------------------------------------------------------- #
class ProtocolError(ValueError):
    """Raised when a trace violates the voyage_trace protocol contract."""


def enforce_invariants(trace: CanonicalTrace) -> None:
    """Validate that a trace satisfies the protocol contract.

    Raises :class:`ProtocolError` on the first violation. Adapters MUST call
    this (or :func:`normalise`) before returning a trace; downstream stages
    assume these invariants hold.

    Invariants:
      * At least one span.
      * Every span's ``trace_id`` matches the trace's ``trace_id``.
      * Every non-root span's ``parent_span_id`` resolves to a span in the
        trace (no dangling references).
      * ``dotted_order``, if present, is well-formed and consistent with the
        parent/child structure (parents sort before children).
      * ``start_time`` is timezone-aware or naive-UTC (never a stray local
        tz); ``end_time >= start_time`` when both are present.
    """
    if not trace.spans:
        raise ProtocolError("trace must contain at least one span")

    ids = {s.span_id for s in trace.spans}
    for span in trace.spans:
        if span.trace_id != trace.trace_id:
            raise ProtocolError(
                f"span {span.span_id} trace_id={span.trace_id!r} != trace.trace_id={trace.trace_id!r}"
            )
        if span.parent_span_id is not None and span.parent_span_id not in ids:
            raise ProtocolError(
                f"span {span.span_id} references unknown parent {span.parent_span_id!r}"
            )
        if span.dotted_order and not validate_dotted_order(span.dotted_order):
            raise ProtocolError(f"span {span.span_id} has malformed dotted_order={span.dotted_order!r}")
        if span.end_time is not None and span.start_time > span.end_time:
            raise ProtocolError(
                f"span {span.span_id} start_time {span.start_time} > end_time {span.end_time}"
            )

    # Parent-before-child ordering check on dotted_order, when all spans have it.
    if all(s.dotted_order for s in trace.spans):
        for span in trace.spans:
            if span.parent_span_id is None:
                continue
            parent = next(s for s in trace.spans if s.span_id == span.parent_span_id)
            if not span.dotted_order.startswith(parent.dotted_order + "."):
                raise ProtocolError(
                    f"span {span.span_id} dotted_order is not a child-prefix of its parent"
                )


def normalise(trace: CanonicalTrace) -> CanonicalTrace:
    """Fill in missing derived fields and enforce invariants.

    Specifically this:
      * Re-parents orphan spans (whose ``parent_span_id`` references a span
        not present in the trace) to ``None`` and clears stale
        ``dotted_order`` values throughout the orphan subtree.
      * Computes ``dotted_order`` for any span missing one (using
        :func:`make_dotted_order` from the parent).
      * Sorts spans by dotted_order.
      * Calls :func:`enforce_invariants`.

    Returns the same trace object (mutated in place) for convenience.
    """
    # Detect orphans BEFORE the dotted_order walk. Previously orphans were
    # fixed *after* the walk, which left their descendants with missing or
    # stale dotted_orders that still referenced the old (absent) parent
    # hierarchy — violating the parent-prefix invariant. By re-parenting
    # up front and clearing stale orders in the orphan subtree, the single
    # _assign traversal recomputes everything consistently.
    ids = {s.span_id for s in trace.spans}
    orphan_subtree: set[str] = set()
    for s in trace.spans:
        if s.parent_span_id is not None and s.parent_span_id not in ids:
            s.parent_span_id = None
            orphan_subtree.add(s.span_id)
    if orphan_subtree:
        # Propagate to descendants so their stale orders are cleared too.
        changed = True
        while changed:
            changed = False
            for s in trace.spans:
                if s.parent_span_id in orphan_subtree and s.span_id not in orphan_subtree:
                    orphan_subtree.add(s.span_id)
                    changed = True
        for s in trace.spans:
            if s.span_id in orphan_subtree:
                s.dotted_order = ""

    by_parent: dict[str | None, list[TraceSpan]] = {}
    for s in trace.spans:
        by_parent.setdefault(s.parent_span_id, []).append(s)

    # Assign dotted_order bottom-up from roots.
    roots = by_parent.get(None, [])
    # If multiple roots, sort by start_time for determinism.
    roots.sort(key=lambda s: s.start_time)

    def _assign(span: TraceSpan, parent_order: str | None) -> None:
        if not span.dotted_order:
            span.dotted_order = make_dotted_order(span.start_time, span.span_id, parent_order)
        for child in sorted(by_parent.get(span.span_id, []), key=lambda s: s.start_time):
            _assign(child, span.dotted_order)

    for root in roots:
        _assign(root, None)

    trace.spans = trace.sorted_spans()
    enforce_invariants(trace)
    return trace
