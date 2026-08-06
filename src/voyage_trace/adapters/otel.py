"""Adapter for OpenTelemetry GenAI span exports.

Input format
------------
A list of OTel span dicts — or a single span, a dict with a ``spans`` key, or
an OTLP-style ``resourceSpans`` tree. Each span has ``trace_id``, ``span_id``,
``parent_span_id``, ``name``, ``start_time`` / ``end_time`` (RFC3339 strings
or ``*_unix_nano`` integers), ``attributes`` and ``status``.

Mapping rules (OpenTelemetry GenAI semantic conventions)
--------------------------------------------------------
* ``gen_ai.operation.name``       -> ``operation_type`` (see ``_OP_NAME_MAP``)
* ``gen_ai.agent.{id,name,version}`` -> ``agent_id`` / ``agent_name`` / ``agent_version``
* ``gen_ai.usage.input_tokens``   -> ``input_tokens``
* ``gen_ai.usage.output_tokens``  -> ``output_tokens``
* ``gen_ai.conversation.id``      -> ``session_id``
* span ``status.code`` == ``ERROR`` -> ``status = failed``
* standard OTel ``parent_span_id`` -> ``parent_span_id``

``source_protocol`` is :attr:`SourceProtocol.OTEL`.
"""

from __future__ import annotations

from typing import Any

from ..types import CanonicalTrace, OperationType, SourceProtocol, SpanStatus
from .base import AdapterError, TraceAdapter, _otel_status_code

# gen_ai.operation.name -> canonical operation type.
_OP_NAME_MAP: dict[str, OperationType] = {
    "chat": OperationType.CHAT,
    "generate_text": OperationType.CHAT,
    "text_completion": OperationType.CHAT,
    "generate": OperationType.CHAT,
    "embeddings": OperationType.EMBEDDING,
    "embedding": OperationType.EMBEDDING,
    "execute_tools": OperationType.EXECUTE_TOOL,
    "execute_tool": OperationType.EXECUTE_TOOL,
    "tool": OperationType.EXECUTE_TOOL,
    "invoke_agent": OperationType.INVOKE_AGENT,
    "retrieval": OperationType.RETRIEVAL,
    "retrieve": OperationType.RETRIEVAL,
    "retriever": OperationType.RETRIEVAL,
    "handoff": OperationType.HANDOFF,
}


class OTELAdapter(TraceAdapter):
    """Convert OpenTelemetry GenAI spans into a :class:`CanonicalTrace`."""

    source_protocol = SourceProtocol.OTEL

    def adapt(self, payload: "dict | list | str | bytes") -> CanonicalTrace:
        data = self._decode(payload)
        otel_spans = self._extract_spans(data)
        if not otel_spans:
            raise AdapterError("otel payload contained no spans")

        trace_id = otel_spans[0].get("trace_id")
        if not trace_id:
            raise AdapterError("otel span missing trace_id")

        spans = [self._otel_span_to_span(sp, str(trace_id)) for sp in otel_spans]

        agent_id = next((s.agent_id for s in spans if s.agent_id), str(trace_id))
        agent_name = next((s.agent_name for s in spans if s.agent_name), "")
        session_id = next((s.session_id for s in spans if s.session_id), "")

        trace = CanonicalTrace(
            trace_id=str(trace_id),
            agent_id=agent_id,
            agent_name=agent_name,
            session_id=session_id,
            source_protocol=self.source_protocol,
            spans=spans,
            metadata={"adapter": "otel"},
        )
        return self._finalise(trace)

    # -- internals -------------------------------------------------------- #
    @staticmethod
    def _extract_spans(data: Any) -> list[dict[str, Any]]:
        if isinstance(data, list):
            return [d for d in data if isinstance(d, dict)]
        if isinstance(data, dict):
            # OTLP collector export shape.
            if "resourceSpans" in data:
                out: list[dict[str, Any]] = []
                for rs in data.get("resourceSpans", []) or []:
                    if not isinstance(rs, dict):
                        continue
                    scopes = rs.get("scopeSpans") or rs.get("instrumentationLibrarySpans") or []
                    for scope in scopes:
                        if not isinstance(scope, dict):
                            continue
                        for sp in scope.get("spans", []) or []:
                            if isinstance(sp, dict):
                                out.append(sp)
                return out
            if "spans" in data and isinstance(data["spans"], list):
                return [d for d in data["spans"] if isinstance(d, dict)]
            return [data]
        return []

    def _otel_span_to_span(self, sp: dict[str, Any], trace_id: str):
        span_id = sp.get("span_id")
        if not span_id:
            raise AdapterError("otel span missing span_id")

        attrs = sp.get("attributes", {}) or {}
        op_name = attrs.get("gen_ai.operation.name")
        op = (
            _OP_NAME_MAP.get(str(op_name).lower(), OperationType.CHAT)
            if op_name
            else OperationType.CHAT
        )

        status = self._otel_status(sp.get("status"))

        start = sp.get("start_time") or sp.get("start_time_unix_nano")
        end = sp.get("end_time") or sp.get("end_time_unix_nano")

        status_obj = sp.get("status") or {}
        status_message = status_obj.get("message") if isinstance(status_obj, dict) else None

        raw_span = {
            "trace_id": trace_id,
            "span_id": str(span_id),
            "parent_span_id": sp.get("parent_span_id"),
            "session_id": attrs.get("gen_ai.conversation.id", "") or attrs.get("session.id", "") or "",
            "agent_id": attrs.get("gen_ai.agent.id", "") or trace_id,
            "agent_name": attrs.get("gen_ai.agent.name", "") or sp.get("name", "") or "",
            "agent_version": attrs.get("gen_ai.agent.version", "") or "",
            "operation_type": op,
            "status": status,
            "start_time": start,
            "end_time": end,
            "inputs": {},
            "outputs": None,
            "error": status_message,
            "metadata": dict(attrs),
            "input_tokens": int(attrs.get("gen_ai.usage.input_tokens", 0) or 0),
            "output_tokens": int(attrs.get("gen_ai.usage.output_tokens", 0) or 0),
            "cost_usd": float(
                attrs.get("gen_ai.usage.cost", 0.0)
                or attrs.get("cost.usd", 0.0)
                or 0.0
            ),
            "source_protocol": self.source_protocol,
        }
        return self._normalise_span(raw_span, trace_id=trace_id)

    @staticmethod
    def _otel_status(status: Any) -> SpanStatus:
        if isinstance(status, dict):
            code = status.get("code")
        else:
            code = status
        # OTel status codes: 1 / "OK" -> SUCCESS, 2 / "ERROR" -> FAILED;
        # unknown codes default to SUCCESS (matching the OTel "unset" -> not
        # an error convention). See :func:`._otel_status_code` for the table.
        return _otel_status_code(code)
