"""Adapter for Langfuse trace + observation exports.

Input format
------------
A Langfuse export is a dict carrying a ``trace`` object and an ``observations``
list, or a flat trace dict with ``observations`` embedded, or just a list of
observations. A trace has ``id``, ``name``, ``session_id``, ``user_id``,
``metadata`` and timestamps. An observation has ``id``, ``trace_id``,
``parent_id``, ``type`` (``event`` / ``span`` / ``generation``), ``name``,
``start_time``, ``end_time``, ``input``, ``output``, ``metadata``, ``level``,
``status_message``, ``usage`` (``input`` / ``output``) and
``calculated_total_cost``.

Mapping rules
-------------
* ``observation.id``           -> ``span_id``
* ``observation.trace_id`` / trace id -> ``trace_id``
* ``observation.parent_id``    -> ``parent_span_id`` (an observation id)
* ``observation.type``         -> ``operation_type``:
    ``generation`` -> ``chat``, ``event`` -> ``chat``,
    ``span`` -> ``invoke_agent`` (overridable via ``metadata.operation_type``)
* ``observation.level``        -> ``status`` (``ERROR`` => ``failed``)
* ``observation.usage.{input,output}`` -> ``input_tokens`` / ``output_tokens``
* ``observation.calculated_total_cost`` -> ``cost_usd``
* ``trace.session_id``         -> ``session_id``
* ``trace.user_id`` / ``trace.metadata.agent_id`` -> ``agent_id``

Langfuse has no ``dotted_order``; :func:`protocol.normalise` derives it from
the ``parent_id`` tree.
"""

from __future__ import annotations

from typing import Any

from ..types import CanonicalTrace, OperationType, SourceProtocol, SpanStatus
from .base import AdapterError, TraceAdapter

# Langfuse observation type -> canonical operation type.
_OBS_TYPE_MAP: dict[str, OperationType] = {
    "generation": OperationType.CHAT,
    "event": OperationType.CHAT,
    "span": OperationType.INVOKE_AGENT,
}


class LangfuseAdapter(TraceAdapter):
    """Convert Langfuse trace + observations into a :class:`CanonicalTrace`."""

    source_protocol = SourceProtocol.LANGFUSE

    def adapt(self, payload: "dict | list | str | bytes") -> CanonicalTrace:
        data = self._decode(payload)
        trace_info, observations = self._extract(data)
        if not observations:
            raise AdapterError("langfuse payload contained no observations")

        trace_id = trace_info.get("id") or observations[0].get("trace_id")
        if not trace_id:
            raise AdapterError("langfuse payload missing trace_id")

        trace_meta = trace_info.get("metadata", {}) or {}
        session_id = trace_info.get("session_id") or trace_info.get("session", "") or ""
        agent_id = trace_meta.get("agent_id") or trace_info.get("user_id") or trace_id
        agent_name = trace_info.get("name", "") or ""

        spans = [
            self._obs_to_span(obs, str(trace_id), session_id, str(agent_id))
            for obs in observations
            if isinstance(obs, dict)
        ]

        trace = CanonicalTrace(
            trace_id=str(trace_id),
            agent_id=str(agent_id),
            agent_name=agent_name,
            session_id=session_id,
            source_protocol=self.source_protocol,
            spans=spans,
            metadata={"adapter": "langfuse"},
        )
        return self._finalise(trace)

    # -- internals -------------------------------------------------------- #
    @staticmethod
    def _extract(data: Any) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        if isinstance(data, list):
            return {}, [d for d in data if isinstance(d, dict)]
        if isinstance(data, dict):
            inner = data.get("trace")
            if isinstance(inner, dict):
                obs = data.get("observations", []) or []
                return inner, [o for o in obs if isinstance(o, dict)]
            if "observations" in data:
                obs = data.get("observations", []) or []
                trace_info = {k: v for k, v in data.items() if k != "observations"}
                return trace_info, [o for o in obs if isinstance(o, dict)]
            # A single observation dict.
            if "id" in data and "type" in data:
                return {}, [data]
            return data, []
        return {}, []

    def _obs_to_span(
        self,
        obs: dict[str, Any],
        trace_id: str,
        session_id: str,
        agent_id: str,
    ):
        span_id = obs.get("id")
        if not span_id:
            raise AdapterError("langfuse observation missing id")

        obs_type = str(obs.get("type", "span")).lower()
        meta = obs.get("metadata", {}) or {}
        if obs_type == "span" and "operation_type" in meta:
            try:
                op = OperationType(meta["operation_type"])
            except ValueError:
                op = _OBS_TYPE_MAP.get(obs_type, OperationType.CHAT)
        else:
            op = _OBS_TYPE_MAP.get(obs_type, OperationType.CHAT)

        usage = obs.get("usage", {}) or {}
        cost = (
            obs.get("calculated_total_cost")
            or obs.get("total_cost")
            or meta.get("cost_usd", 0.0)
            or 0.0
        )

        raw_span = {
            "trace_id": trace_id,
            "span_id": str(span_id),
            "parent_span_id": obs.get("parent_id"),
            "session_id": session_id,
            "agent_id": agent_id,
            "agent_name": obs.get("name", "") or "",
            "operation_type": op,
            "status": self._obs_status(obs),
            "start_time": obs.get("start_time"),
            "end_time": obs.get("end_time"),
            "inputs": obs.get("input", {}) or {},
            "outputs": obs.get("output"),
            "error": obs.get("status_message"),
            "metadata": {**meta, "type": obs_type, "model": obs.get("model", "")},
            "input_tokens": usage.get("input", 0) or 0,
            "output_tokens": usage.get("output", 0) or 0,
            "cost_usd": cost,
            "source_protocol": self.source_protocol,
        }
        return self._normalise_span(raw_span, trace_id=trace_id)

    @staticmethod
    def _obs_status(obs: dict[str, Any]) -> SpanStatus:
        level = str(obs.get("level", "")).upper()
        if level == "ERROR":
            return SpanStatus.FAILED
        if obs.get("end_time") is None:
            return SpanStatus.WORKING
        return SpanStatus.SUCCESS
