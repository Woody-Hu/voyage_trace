"""Adapter for LangSmith run exports.

Input format
------------
A LangSmith *run* JSON object — or a list of runs, or a dict carrying a
``runs`` key. Each run typically has: ``id``, ``trace_id``, ``parent_run_id``,
``run_type``, ``name``, ``inputs``, ``outputs``, ``error``, ``start_time``,
``end_time``, ``prompt_tokens``, ``completion_tokens``, ``total_cost``,
``status``, ``session_id`` and (optionally) ``dotted_order``.

Mapping rules
-------------
* ``run.id``                    -> ``span_id``
* ``run.trace_id`` (or ``id``)  -> ``trace_id``
* ``run.parent_run_id``         -> ``parent_span_id``
* ``run.dotted_order``          -> ``dotted_order`` (kept as-is when present)
* ``run.run_type``              -> ``operation_type`` (see ``_RUN_TYPE_MAP``)
* ``run.status`` / ``run.error``-> ``status`` (an ``error`` field wins => failed)
* ``run.prompt_tokens``         -> ``input_tokens``
* ``run.completion_tokens``     -> ``output_tokens``
* ``run.total_cost``            -> ``cost_usd``
* ``run.session_id``            -> ``session_id``
* ``run.name``                  -> ``agent_name``
* ``run.extra.metadata.agent_id`` -> ``agent_id`` (fallback: ``trace_id``)

The adapter only parses exported JSON — it never calls the LangSmith SDK.
"""

from __future__ import annotations

from typing import Any

from ..types import CanonicalTrace, OperationType, SourceProtocol, SpanStatus
from .base import AdapterError, TraceAdapter

# LangSmith run_type -> canonical operation type.
_RUN_TYPE_MAP: dict[str, OperationType] = {
    "chain": OperationType.INVOKE_AGENT,
    "llm": OperationType.CHAT,
    "tool": OperationType.EXECUTE_TOOL,
    "retriever": OperationType.RETRIEVAL,
    "embedding": OperationType.EMBEDDING,
    "prompt": OperationType.CHAT,
    "parser": OperationType.CHAT,
}

# LangSmith string status -> canonical span status.
_STATUS_MAP: dict[str, SpanStatus] = {
    "success": SpanStatus.SUCCESS,
    "error": SpanStatus.FAILED,
    "running": SpanStatus.WORKING,
    "awaiting": SpanStatus.INPUT_REQUIRED,
}


class LangSmithAdapter(TraceAdapter):
    """Convert LangSmith run JSON into a :class:`CanonicalTrace`."""

    source_protocol = SourceProtocol.LANGSMITH

    def adapt(self, payload: "dict | list | str | bytes") -> CanonicalTrace:
        data = self._decode(payload)
        runs = self._extract_runs(data)
        if not runs:
            raise AdapterError("langsmith payload contained no runs")

        trace_id = runs[0].get("trace_id") or runs[0].get("id")
        if not trace_id:
            raise AdapterError("langsmith run missing trace_id")

        spans = [self._run_to_span(run, trace_id) for run in runs]

        # Trace-level fields come from the root run (no parent), else the first.
        root = next((r for r in runs if not r.get("parent_run_id")), runs[0])
        extra = root.get("extra", {}) or {}
        meta = extra.get("metadata", {}) or {}
        agent_id = meta.get("agent_id") or trace_id

        trace = CanonicalTrace(
            trace_id=str(trace_id),
            agent_id=str(agent_id),
            agent_name=root.get("name", "") or "",
            session_id=root.get("session_id", "") or "",
            source_protocol=self.source_protocol,
            spans=spans,
            metadata={"adapter": "langsmith"},
        )
        return self._finalise(trace)

    # -- internals -------------------------------------------------------- #
    @staticmethod
    def _extract_runs(data: Any) -> list[dict[str, Any]]:
        if isinstance(data, list):
            return [d for d in data if isinstance(d, dict)]
        if isinstance(data, dict):
            runs = data.get("runs")
            if isinstance(runs, list):
                return [d for d in runs if isinstance(d, dict)]
            return [data]
        return []

    def _run_to_span(self, run: dict[str, Any], trace_id: str):
        span_id = run.get("id")
        if not span_id:
            raise AdapterError("langsmith run missing id")

        run_type = run.get("run_type", "chain")
        op = _RUN_TYPE_MAP.get(str(run_type).lower(), OperationType.INVOKE_AGENT)

        extra = run.get("extra", {}) or {}
        meta = extra.get("metadata", {}) or {}
        span_meta = {**meta, "run_type": run_type, "name": run.get("name", "")}

        raw_span = {
            "trace_id": trace_id,
            "span_id": str(span_id),
            "parent_span_id": run.get("parent_run_id"),
            "dotted_order": run.get("dotted_order", "") or "",
            "session_id": run.get("session_id", "") or "",
            "agent_id": meta.get("agent_id", "") or trace_id,
            "agent_name": run.get("name", "") or "",
            "operation_type": op,
            "status": self._run_status(run),
            "start_time": run.get("start_time"),
            "end_time": run.get("end_time"),
            "inputs": run.get("inputs", {}) or {},
            "outputs": run.get("outputs"),
            "error": run.get("error"),
            "metadata": span_meta,
            "input_tokens": run.get("prompt_tokens", 0) or 0,
            "output_tokens": run.get("completion_tokens", 0) or 0,
            "cost_usd": run.get("total_cost", 0.0) or 0.0,
            "source_protocol": self.source_protocol,
        }
        return self._normalise_span(raw_span, trace_id=trace_id)

    @staticmethod
    def _run_status(run: dict[str, Any]) -> SpanStatus:
        # An explicit error beats the declared status.
        if run.get("error"):
            return SpanStatus.FAILED
        status = run.get("status")
        if isinstance(status, str):
            return _STATUS_MAP.get(status.lower(), SpanStatus.UNKNOWN)
        if status == 1:
            return SpanStatus.SUCCESS
        if status == 0:
            return SpanStatus.FAILED
        return SpanStatus.UNKNOWN
