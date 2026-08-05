"""Langfuse export — PUSH a :class:`CanonicalTrace` into Langfuse via the SDK.

The pull side (Langfuse trace/observation export → :class:`CanonicalTrace`)
already lives in :mod:`voyage_trace.adapters.langfuse` and needs no SDK — it
parses exported JSON. This module is the **push side**: it walks a
:class:`CanonicalTrace`, opens one Langfuse trace + one observation per span,
and records per-span cost / token / status as Langfuse scores. The Langfuse SDK
(``langfuse`` v2 or the v3 OTel-based SDK) is imported lazily; if it is absent
the function returns a JSON-safe export payload (the same shape Langfuse's own
``fetch()`` returns) so the artefact can be ingested later, by the adapter,
without re-deriving the mapping.

The integration never edits the canonical schema; it is a lossless boundary
translation.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from ..types import CanonicalTrace, OperationType, SpanStatus, TraceSpan


def _import_langfuse() -> Any | None:
    """Lazily import the Langfuse SDK; return ``None`` if not installed."""
    try:
        from langfuse import Langfuse  # type: ignore[import-not-found]
        return Langfuse
    except ImportError:
        return None


def _span_to_observation_dict(span: TraceSpan, trace_id: str) -> dict[str, Any]:
    """Project a :class:`TraceSpan` onto the Langfuse observation dict shape.

    The dict mirrors what the Langfuse SDK would internally send for an
    observation: ``id``, ``trace_id``, ``parent_id``, ``type``, ``name``,
    ``start_time``, ``end_time``, ``input``, ``output``, ``metadata``,
    ``level``, ``status_message``, ``usage``, ``calculated_total_cost``.
    The pull-side adapter ingests exactly this shape, so push→pull is a
    round-trip with no information loss.
    """
    obs_type = (
        "generation" if span.operation_type == OperationType.CHAT
        else "span"
    )
    level = "ERROR" if span.status in (SpanStatus.FAILED, SpanStatus.ERROR) else "DEBUG"
    usage: dict[str, int] = {}
    if span.input_tokens:
        usage["input"] = span.input_tokens
    if span.output_tokens:
        usage["output"] = span.output_tokens
    return {
        "id": span.span_id,
        "trace_id": trace_id,
        "parent_id": span.parent_span_id,
        "type": obs_type,
        "name": span.agent_name or span.metadata.get("name", "") or span.span_id,
        "start_time": span.start_time.isoformat() if span.start_time else None,
        "end_time": span.end_time.isoformat() if span.end_time else None,
        "input": dict(span.inputs) if span.inputs else None,
        "output": dict(span.outputs) if span.outputs else None,
        "metadata": {**dict(span.metadata), "operation_type": span.operation_type.value},
        "level": level,
        "status_message": span.error,
        "usage": usage or None,
        "model": span.metadata.get("model", ""),
        "calculated_total_cost": span.cost_usd or 0.0,
    }


def _trace_to_export_dict(trace: CanonicalTrace) -> dict[str, Any]:
    """Build the JSON-safe Langfuse export shape for ``trace``.

    This is the artefact emitted when the SDK is absent — loadable by
    :class:`voyage_trace.adapters.langfuse.LangfuseAdapter` without the SDK.
    """
    observations = [_span_to_observation_dict(s, trace.trace_id) for s in trace.spans]
    return {
        "trace": {
            "id": trace.trace_id,
            "name": trace.agent_name or trace.trace_id,
            "session_id": trace.session_id,
            "user_id": trace.agent_id,
            "metadata": {**dict(trace.metadata), "agent_id": trace.agent_id},
        },
        "observations": observations,
    }


def export_to_langfuse(
    trace: CanonicalTrace,
    *,
    client: Any | None = None,
    public_key: str | None = None,
    secret_key: str | None = None,
    host: str | None = None,
    flush_at: int = 1,
) -> dict[str, Any]:
    """Push ``trace`` into Langfuse; return a JSON-safe export on any path.

    Parameters
    ----------
    trace:
        The :class:`CanonicalTrace` to push.
    client:
        An already-constructed Langfuse client. If absent and the SDK is
        installed, a client is built from ``public_key`` / ``secret_key`` /
        ``host`` (or from the ``LANGFUSE_*`` env vars the SDK reads itself).
    flush_at:
        Langfuse SDK flush threshold. Defaults to ``1`` so the trace is
        shipped synchronously in tests; raise for batched production use.

    Returns
    -------
    dict
        A JSON-safe Langfuse export shape — the same payload the pull-side
        :class:`~voyage_trace.adapters.langfuse.LangfuseAdapter` ingests. This
        is returned **regardless** of whether the SDK was used, so callers
        always have a serialisable artefact (log it, persist it, replay it
        through the adapter later).
    """
    export = _trace_to_export_dict(trace)

    Langfuse = _import_langfuse() if client is None else None
    if Langfuse is None and client is None:
        # SDK absent — return the JSON-safe artefact; caller can persist it.
        export["metadata"] = {"exported_with_sdk": False, **export.get("metadata", {})}
        return export

    lf = client or Langfuse(
        public_key=public_key,
        secret_key=secret_key,
        host=host,
        flush_at=flush_at,
    )
    try:
        lf_trace = lf.trace(
            id=trace.trace_id,
            name=trace.agent_name or trace.trace_id,
            session_id=trace.session_id or None,
            user_id=trace.agent_id or None,
            metadata={**dict(trace.metadata), "agent_id": trace.agent_id},
        )
        for span in trace.spans:
            obs_type = (
                "generation" if span.operation_type == OperationType.CHAT
                else "span"
            )
            obs = lf_trace.span(
                id=span.span_id,
                name=span.agent_name or span.metadata.get("name", "") or span.span_id,
                start_time=span.start_time,
                end_time=span.end_time,
                input=dict(span.inputs) if span.inputs else None,
                output=dict(span.outputs) if span.outputs else None,
                metadata={**dict(span.metadata), "operation_type": span.operation_type.value},
                level=("ERROR" if span.status in (SpanStatus.FAILED, SpanStatus.ERROR)
                       else "DEBUG"),
                status_message=span.error,
            )
            usage: dict[str, int] = {}
            if span.input_tokens:
                usage["input"] = span.input_tokens
            if span.output_tokens:
                usage["output"] = span.output_tokens
            if usage:
                try:
                    obs.update(usage=usage)
                except Exception:  # noqa: BLE001 — usage is optional
                    pass
            if span.cost_usd:
                try:
                    lf.score(
                        trace_id=trace.trace_id,
                        name="cost_usd",
                        value=float(span.cost_usd),
                        data_type="NUMERIC",
                        comment=f"cost for span {span.span_id}",
                    )
                except Exception:  # noqa: BLE001 — scoring is best-effort
                    pass
        try:
            lf.flush()
        except Exception:  # noqa: BLE001 — flush is best-effort
            pass
        export["metadata"] = {"exported_with_sdk": True, **export.get("metadata", {})}
    except Exception:  # noqa: BLE001 — degrade rather than crash the round
        export["metadata"] = {"exported_with_sdk": False, "error": "push_failed",
                              **export.get("metadata", {})}
    return export


def export_observation_now(
    span: TraceSpan,
    trace_id: str,
    *,
    client: Any | None = None,
) -> dict[str, Any]:
    """Convenience: emit one observation dict for a single span.

    Useful for streaming / live tracing where you want to project one span at
    a time. Returns the JSON-safe observation dict; the SDK path mirrors
    :func:`export_to_langfuse` but for one observation only.
    """
    obs = _span_to_observation_dict(span, trace_id)
    Langfuse = _import_langfuse() if client is None else None
    if Langfuse is None and client is None:
        return obs
    lf = client or Langfuse()
    try:
        lf.trace(id=trace_id).span(
            id=span.span_id,
            name=obs["name"],
            start_time=span.start_time,
            end_time=span.end_time,
            input=obs["input"],
            output=obs["output"],
            metadata=obs["metadata"],
        )
    except Exception:  # noqa: BLE001
        pass
    return obs


def parse_langfuse_datetime(value: str | datetime | None) -> datetime | None:
    """Parse a Langfuse RFC3339 timestamp; tolerant of ``Z`` suffix.

    Exposed as a helper because the pull adapter and tests both need it; kept
    here to avoid duplicating the ``Z → +00:00`` fix in two modules.
    """
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except (ValueError, TypeError):
        return None
