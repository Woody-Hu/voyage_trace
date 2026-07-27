"""Trace adapters — convert backend-specific traces to the canonical schema.

This package exposes:

* :class:`TraceAdapter` — the abstract base every adapter implements.
* One concrete adapter per supported :class:`SourceProtocol`:
  :class:`LangSmithAdapter`, :class:`LangfuseAdapter`, :class:`OTELAdapter`,
  :class:`A2AAdapter`, :class:`MCPAdapter`, :class:`RawAdapter`.
* :data:`ADAPTER_REGISTRY` — ``SourceProtocol`` -> adapter class.
* :func:`adapt` — the single entry point: ``adapt(raw_payload, source_protocol)``.

Adapters depend only on :mod:`voyage_trace.types` and
:mod:`voyage_trace.protocol`; they parse exported JSON, never call a backend
SDK.
"""

from __future__ import annotations

import json
from typing import Any

from ..types import CanonicalTrace, SourceProtocol
from .a2a import A2AAdapter
from .base import AdapterError, TraceAdapter
from .langfuse import LangfuseAdapter
from .langsmith import LangSmithAdapter
from .mcp import MCPAdapter
from .otel import OTELAdapter
from .raw import RawAdapter

ADAPTER_REGISTRY: dict[SourceProtocol, type[TraceAdapter]] = {
    SourceProtocol.LANGSMITH: LangSmithAdapter,
    SourceProtocol.LANGFUSE: LangfuseAdapter,
    SourceProtocol.OTEL: OTELAdapter,
    SourceProtocol.A2A: A2AAdapter,
    SourceProtocol.MCP: MCPAdapter,
    SourceProtocol.CUSTOM: RawAdapter,
}


def _infer_protocol(payload: Any) -> SourceProtocol:
    """Best-effort guess of the source protocol from payload shape.

    Used only when ``adapt`` is called without an explicit
    ``source_protocol``; any ambiguity falls back to :attr:`SourceProtocol.CUSTOM`.
    """
    if isinstance(payload, CanonicalTrace):
        return SourceProtocol.CUSTOM
    if isinstance(payload, (str, bytes)):
        try:
            payload = json.loads(payload)
        except (ValueError, TypeError):
            return SourceProtocol.CUSTOM
    if isinstance(payload, dict):
        if "run_type" in payload or "runs" in payload:
            return SourceProtocol.LANGSMITH
        if "observations" in payload or isinstance(payload.get("trace"), dict):
            return SourceProtocol.LANGFUSE
        if "resourceSpans" in payload:
            return SourceProtocol.OTEL
        if "spans" in payload:
            # A canonical trace dict carries trace_id + agent_id at the top
            # level; a bare OTel {spans: [...]} wrapper does not.
            if "agent_id" in payload and "trace_id" in payload:
                return SourceProtocol.CUSTOM
            return SourceProtocol.OTEL
        if "history" in payload or "contextId" in payload:
            return SourceProtocol.A2A
        if "state" in payload and "timestamp" in payload:
            return SourceProtocol.A2A
        if "messages" in payload and isinstance(payload.get("messages"), list):
            return SourceProtocol.MCP
        if "jsonrpc" in payload or "method" in payload:
            return SourceProtocol.MCP
        return SourceProtocol.CUSTOM
    if isinstance(payload, list) and payload and isinstance(payload[0], dict):
        first = payload[0]
        if "run_type" in first:
            return SourceProtocol.LANGSMITH
        if "type" in first and "trace_id" in first and "parent_id" in first:
            return SourceProtocol.LANGFUSE
        if "attributes" in first or "resourceSpans" in first:
            return SourceProtocol.OTEL
        if "state" in first:
            return SourceProtocol.A2A
        if "jsonrpc" in first or "method" in first:
            return SourceProtocol.MCP
        return SourceProtocol.CUSTOM
    return SourceProtocol.CUSTOM


def adapt(
    raw_payload: "dict | list | str | bytes | CanonicalTrace",
    source_protocol: "SourceProtocol | str | None" = None,
) -> CanonicalTrace:
    """Convert ``raw_payload`` into a canonical :class:`CanonicalTrace`.

    Parameters
    ----------
    raw_payload:
        Backend-specific trace data (dict / list / JSON string / bytes), or an
        already-built :class:`CanonicalTrace` (passed through by the raw
        adapter).
    source_protocol:
        Selects the adapter. Accepts a :class:`SourceProtocol` member, its
        string value (e.g. ``"langsmith"``), or ``None`` to infer from the
        payload shape.
    """
    if source_protocol is None:
        proto = _infer_protocol(raw_payload)
    elif isinstance(source_protocol, SourceProtocol):
        proto = source_protocol
    else:
        proto = SourceProtocol(str(source_protocol).lower())

    adapter_cls = ADAPTER_REGISTRY.get(proto)
    if adapter_cls is None:
        raise AdapterError(f"no adapter registered for source_protocol={proto!r}")
    return adapter_cls().adapt(raw_payload)


__all__ = [
    "AdapterError",
    "TraceAdapter",
    "ADAPTER_REGISTRY",
    "adapt",
    "LangSmithAdapter",
    "LangfuseAdapter",
    "OTELAdapter",
    "A2AAdapter",
    "MCPAdapter",
    "RawAdapter",
]
