"""Adapter for MCP (Model Context Protocol) message logs and OTel MCP spans.

Input format
------------
Either a list of JSON-RPC messages (requests ``{jsonrpc, id, method, params}``,
notifications ``{jsonrpc, method, params}`` and responses
``{jsonrpc, id, result | error}``) — optionally wrapped as
``{trace_id, messages: [...]}`` — or a list of OTel-style span dicts whose
``attributes`` describe MCP calls. Each ``tools/call`` request becomes one
span; other method calls are also captured.

Mapping rules
-------------
* JSON-RPC ``method`` -> ``operation_type``:
    ``tools/*`` -> ``execute_tool``, ``resources/*`` -> ``retrieval``,
    ``prompts/*`` -> ``chat``, otherwise ``execute_tool``.
* ``params.name`` (tool name) and the matched response ``result`` /
  ``error`` populate ``inputs`` / ``outputs`` / ``error``.
* ``trace_id`` is read from a wrapper field, ``params._meta.trace_id`` /
  ``traceId``, a top-level ``trace_id`` / ``traceId`` on any message, or — as a
  last resort — a ``session_id`` / ``sessionId`` (an MCP session is the
  closest analogue to a trace).
* ``agent_id`` / ``agent_name`` come from ``params._meta.server_name`` /
  ``serverName`` (fallback ``mcp-server``).
* OTel-style inputs: each span's ``attributes`` carry the method; the standard
  OTel status code maps to ``failed`` / ``success``.

``source_protocol`` is :attr:`SourceProtocol.MCP`.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from ..types import CanonicalTrace, OperationType, SourceProtocol, SpanStatus
from .base import AdapterError, TraceAdapter


class MCPAdapter(TraceAdapter):
    """Convert MCP JSON-RPC logs / OTel MCP spans into a :class:`CanonicalTrace`."""

    source_protocol = SourceProtocol.MCP

    def adapt(self, payload: "dict | list | str | bytes") -> CanonicalTrace:
        data = self._decode(payload)
        items = self._extract_messages(data)
        if not items:
            raise AdapterError("mcp payload contained no messages")

        # OTel-style span list?
        if isinstance(items[0], dict) and "attributes" in items[0]:
            return self._adapt_otel_spans(items)

        return self._adapt_jsonrpc(items, data)

    # -- JSON-RPC path ---------------------------------------------------- #
    def _adapt_jsonrpc(
        self,
        messages: list[dict[str, Any]],
        data: Any = None,
    ) -> CanonicalTrace:
        trace_id = self._find_trace_id(messages) or self._wrapper_trace_id(data)
        if not trace_id:
            raise AdapterError("mcp payload missing trace_id")

        responses: dict[Any, dict[str, Any]] = {}
        for m in messages:
            mid = m.get("id")
            if mid is not None and ("result" in m or "error" in m):
                responses[mid] = m

        server_name = (
            self._find_server_name(messages)
            or self._wrapper_field(data, ("server_name", "serverName", "server"))
            or "mcp-server"
        )
        spans = []
        for m in messages:
            method = m.get("method")
            if not method:
                continue  # response or notification without method
            spans.append(
                self._jsonrpc_to_span(m, method, responses, str(trace_id), server_name)
            )

        if not spans:
            raise AdapterError("mcp payload contained no method calls")

        trace = CanonicalTrace(
            trace_id=str(trace_id),
            agent_id=server_name,
            agent_name=server_name,
            source_protocol=self.source_protocol,
            spans=spans,
            metadata={"adapter": "mcp"},
        )
        return self._finalise(trace)

    def _jsonrpc_to_span(
        self,
        msg: dict[str, Any],
        method: str,
        responses: dict[Any, dict[str, Any]],
        trace_id: str,
        server_name: str,
    ):
        mid = msg.get("id")
        params = msg.get("params", {}) or {}
        meta = params.get("_meta", {}) or {} if isinstance(params, dict) else {}

        op = self._method_to_op(method)
        request_ts = self._parse_dt(
            meta.get("start_time") or meta.get("timestamp") or msg.get("timestamp")
        )
        response = responses.get(mid) if mid is not None else None
        response_meta = (response.get("_meta") or {}) if response else {}
        response_ts = self._parse_dt(
            response_meta.get("end_time")
            or response_meta.get("timestamp")
            or (response.get("timestamp") if response else None)
        )

        start = request_ts or datetime.now(timezone.utc)
        end = response_ts or self._parse_dt(meta.get("end_time"))
        if end is not None and end < start:
            end = None

        error: str | None = None
        status = SpanStatus.SUCCESS
        outputs: Any = None
        if response is not None:
            if response.get("error") is not None:
                error = self._coerce_error(response.get("error"))
                status = SpanStatus.FAILED
            else:
                result = response.get("result")
                outputs = result
                if isinstance(result, dict) and result.get("isError"):
                    error = "tool call returned isError=True"
                    status = SpanStatus.FAILED

        span_id = f"mcp-{mid}" if mid is not None else f"mcp-{method}-{len(trace_id)}"

        raw_span = {
            "trace_id": trace_id,
            "span_id": span_id,
            "parent_span_id": meta.get("parent_span_id"),
            "agent_id": server_name,
            "agent_name": server_name,
            "operation_type": op,
            "status": status,
            "start_time": start,
            "end_time": end,
            "inputs": (
                {"method": method, "params": params}
                if isinstance(params, dict)
                else {"method": method}
            ),
            "outputs": outputs,
            "error": error,
            "metadata": {"method": method, "id": mid, "server": server_name},
            "source_protocol": self.source_protocol,
        }
        return self._normalise_span(raw_span, trace_id=trace_id)

    # -- OTel-style path -------------------------------------------------- #
    def _adapt_otel_spans(self, spans_raw: list[dict[str, Any]]) -> CanonicalTrace:
        trace_id = spans_raw[0].get("trace_id")
        if not trace_id:
            raise AdapterError("mcp otel span missing trace_id")

        spans = []
        for sp in spans_raw:
            attrs = sp.get("attributes", {}) or {}
            span_id = sp.get("span_id")
            if not span_id:
                raise AdapterError("mcp otel span missing span_id")
            method = attrs.get("mcp.method") or attrs.get("rpc.method") or sp.get("name", "")
            if isinstance(method, str) and "/" in method:
                op = self._method_to_op(method)
            else:
                op = OperationType.EXECUTE_TOOL

            status_obj = sp.get("status") or {}
            code = status_obj.get("code") if isinstance(status_obj, dict) else status_obj
            code_s = str(code).upper()
            if code_s in ("ERROR", "2") or code == 2:
                status = SpanStatus.FAILED
            else:
                status = SpanStatus.SUCCESS
            err_msg = status_obj.get("message") if isinstance(status_obj, dict) else None

            server = attrs.get("mcp.server.name", "") or "mcp-server"
            raw_span = {
                "trace_id": str(trace_id),
                "span_id": str(span_id),
                "parent_span_id": sp.get("parent_span_id"),
                "agent_id": server,
                "agent_name": attrs.get("mcp.server.name", "") or sp.get("name", "") or server,
                "operation_type": op,
                "status": status,
                "start_time": sp.get("start_time") or sp.get("start_time_unix_nano"),
                "end_time": sp.get("end_time") or sp.get("end_time_unix_nano"),
                "inputs": {"method": method},
                "outputs": None,
                "error": err_msg,
                "metadata": dict(attrs),
                "source_protocol": self.source_protocol,
            }
            spans.append(self._normalise_span(raw_span, trace_id=str(trace_id)))

        agent_id = next((s.agent_id for s in spans if s.agent_id), "mcp-server")
        trace = CanonicalTrace(
            trace_id=str(trace_id),
            agent_id=agent_id,
            agent_name=agent_id,
            source_protocol=self.source_protocol,
            spans=spans,
            metadata={"adapter": "mcp-otel"},
        )
        return self._finalise(trace)

    # -- helpers ---------------------------------------------------------- #
    @staticmethod
    def _extract_messages(data: Any) -> list[dict[str, Any]]:
        if isinstance(data, list):
            return [d for d in data if isinstance(d, dict)]
        if isinstance(data, dict):
            msgs = data.get("messages")
            if isinstance(msgs, list):
                return [d for d in msgs if isinstance(d, dict)]
            if "resourceSpans" in data or "spans" in data:
                if isinstance(data.get("spans"), list):
                    return [d for d in data["spans"] if isinstance(d, dict)]
                return [data]
            return [data]
        return []

    @staticmethod
    def _method_to_op(method: str) -> OperationType:
        if method.startswith("tools/"):
            return OperationType.EXECUTE_TOOL
        if method.startswith("resources/"):
            return OperationType.RETRIEVAL
        if method.startswith("prompts/"):
            return OperationType.CHAT
        return OperationType.EXECUTE_TOOL

    @staticmethod
    def _wrapper_trace_id(data: Any) -> str | None:
        return MCPAdapter._wrapper_field(data, ("trace_id", "traceId"))

    @staticmethod
    def _wrapper_field(data: Any, keys: tuple[str, ...]) -> str | None:
        """Read a scalar field off a wrapper dict (the original payload root)."""
        if isinstance(data, dict):
            for key in keys:
                if data.get(key):
                    return str(data[key])
        return None

    @staticmethod
    def _find_trace_id(messages: list[dict[str, Any]]) -> str | None:
        for m in messages:
            if not isinstance(m, dict):
                continue
            for key in ("trace_id", "traceId"):
                if m.get(key):
                    return str(m[key])
            params = m.get("params")
            if isinstance(params, dict):
                meta = params.get("_meta", {}) or {}
                for key in ("trace_id", "traceId"):
                    if meta.get(key):
                        return str(meta[key])
        # Last resort: a shared session id (MCP session ≈ trace).
        for m in messages:
            if not isinstance(m, dict):
                continue
            for key in ("session_id", "sessionId"):
                if m.get(key):
                    return str(m[key])
            params = m.get("params")
            if isinstance(params, dict):
                meta = params.get("_meta", {}) or {}
                for key in ("session_id", "sessionId"):
                    if meta.get(key):
                        return str(meta[key])
        return None

    @staticmethod
    def _find_server_name(messages: list[dict[str, Any]]) -> str | None:
        for m in messages:
            if not isinstance(m, dict):
                continue
            for key in ("server_name", "serverName", "server"):
                if m.get(key):
                    return str(m[key])
            params = m.get("params")
            if isinstance(params, dict):
                meta = params.get("_meta", {}) or {}
                for key in ("server_name", "serverName", "server"):
                    if meta.get(key):
                        return str(meta[key])
        return None
