"""Adapter for A2A (Agent-to-Agent) protocol Task state sequences.

Input format
------------
Either an A2A ``Task`` dict — ``{id, contextId, status, history, artifacts,
metadata}`` where ``status`` / each history entry is
``{state, timestamp, message}`` — or a flat list of such status-update dicts
(each optionally carrying ``task_id`` / ``trace_id`` / ``agent_id``).

Mapping rules
-------------
Each status transition becomes one span with
``operation_type = invoke_agent``:

* A2A ``state`` -> ``SpanStatus``:
    ``submitted`` -> ``submitted``, ``working`` -> ``working``,
    ``input_required`` -> ``input_required``, ``completed`` -> ``success``,
    ``failed`` -> ``failed``, ``canceled`` / ``cancelled`` -> ``canceled``
* Spans are chained parent -> child in chronological order; span ``i``'s
  ``end_time`` is span ``i+1``'s ``start_time`` (the time spent in that state).
* ``agent_id`` is inferred from ``task.metadata.agent_id`` or
  ``task.contextId`` (fallback: ``trace_id``).
* ``task.id`` -> ``trace_id``; ``message`` with ``role=user`` -> ``inputs``,
  ``role=agent`` -> ``outputs``; ``artifact`` -> ``outputs``.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from ..types import CanonicalTrace, OperationType, SourceProtocol, SpanStatus
from .base import AdapterError, TraceAdapter

# A2A Task state -> canonical span status.
_STATE_MAP: dict[str, SpanStatus] = {
    "submitted": SpanStatus.SUBMITTED,
    "working": SpanStatus.WORKING,
    "input_required": SpanStatus.INPUT_REQUIRED,
    "completed": SpanStatus.SUCCESS,
    "failed": SpanStatus.FAILED,
    "canceled": SpanStatus.CANCELED,
    "cancelled": SpanStatus.CANCELED,
}


class A2AAdapter(TraceAdapter):
    """Convert an A2A Task / status sequence into a :class:`CanonicalTrace`."""

    source_protocol = SourceProtocol.A2A

    def adapt(self, payload: "dict | list | str | bytes") -> CanonicalTrace:
        data = self._decode(payload)
        trace_id, agent_id, agent_name, session_id, status_seq = self._extract(data)
        if not status_seq:
            raise AdapterError("a2a payload contained no status updates")

        spans = [
            self._update_to_span(upd, i, status_seq, trace_id, agent_id)
            for i, upd in enumerate(status_seq)
            if isinstance(upd, dict)
        ]

        trace = CanonicalTrace(
            trace_id=trace_id,
            agent_id=agent_id,
            agent_name=agent_name,
            session_id=session_id,
            source_protocol=self.source_protocol,
            spans=spans,
            metadata={"adapter": "a2a"},
        )
        return self._finalise(trace)

    # -- internals -------------------------------------------------------- #
    def _extract(
        self, data: Any
    ) -> tuple[str, str, str, str, list[dict[str, Any]]]:
        if isinstance(data, list):
            items = [d for d in data if isinstance(d, dict)]
            if not items:
                raise AdapterError("a2a payload empty")
            trace_id = items[0].get("task_id") or items[0].get("trace_id")
            if not trace_id:
                raise AdapterError("a2a status update missing task_id/trace_id")
            agent_id = items[0].get("agent_id", "") or str(trace_id)
            agent_name = items[0].get("agent_name", "") or ""
            session_id = items[0].get("session_id", "") or ""
            return str(trace_id), str(agent_id), agent_name, session_id, items

        if isinstance(data, dict):
            trace_id = data.get("id") or data.get("task_id") or data.get("trace_id")
            if not trace_id:
                raise AdapterError("a2a task missing id")
            meta = data.get("metadata", {}) or {}
            agent_id = meta.get("agent_id") or data.get("contextId") or trace_id
            agent_name = meta.get("agent_name") or data.get("agent_name", "") or ""
            session_id = data.get("contextId", "") or meta.get("session_id", "") or ""
            history = list(data.get("history", []) or [])
            status = data.get("status")
            if isinstance(status, dict):
                history.append(status)
            return str(trace_id), str(agent_id), agent_name, session_id, history

        raise AdapterError("a2a payload must be a task dict or list of status updates")

    def _update_to_span(
        self,
        upd: dict[str, Any],
        i: int,
        seq: list[dict[str, Any]],
        trace_id: str,
        agent_id: str,
    ):
        state = upd.get("state", "unknown")
        status = _STATE_MAP.get(str(state).lower(), SpanStatus.UNKNOWN)

        start = self._parse_dt(upd.get("timestamp")) or datetime.now(timezone.utc)
        end: datetime | None = None
        if i + 1 < len(seq):
            candidate = self._parse_dt(seq[i + 1].get("timestamp"))
            if candidate is not None and candidate >= start:
                end = candidate

        span_id = f"{trace_id}-s{i}"
        parent_span_id = f"{trace_id}-s{i - 1}" if i > 0 else None

        msg = upd.get("message")
        role = msg.get("role", "") if isinstance(msg, dict) else ""
        inputs: dict[str, Any] = {}
        outputs: dict[str, Any] | None = None
        if role == "user" and isinstance(msg, dict):
            inputs = {"message": msg}
        elif role == "agent" and isinstance(msg, dict):
            outputs = {"message": msg}

        meta: dict[str, Any] = {"state": state, "role": role}
        artifact = upd.get("artifact")
        if artifact:
            meta["artifact"] = artifact
            if outputs is None:
                outputs = {"artifact": artifact}

        raw_span = {
            "trace_id": trace_id,
            "span_id": span_id,
            "parent_span_id": parent_span_id,
            "agent_id": agent_id,
            "operation_type": OperationType.INVOKE_AGENT,
            "status": status,
            "start_time": start,
            "end_time": end,
            "inputs": inputs,
            "outputs": outputs,
            "metadata": meta,
            "source_protocol": self.source_protocol,
        }
        return self._normalise_span(raw_span, trace_id=trace_id)
