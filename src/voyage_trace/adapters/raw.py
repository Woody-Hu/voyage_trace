"""Adapter for already-canonical or semi-structured payloads.

Input format
------------
The :class:`RawAdapter` is the fallback (``source_protocol = custom``). It
accepts, in order of preference:

1. A :class:`CanonicalTrace` object — passed through (re-normalised).
2. A canonical dict (``{trace_id, agent_id, spans, ...}``) — parsed via
   :func:`protocol.trace_from_dict`.
3. A single span-like dict (``{trace_id, span_id | id, ...}``) — wrapped into
   a one-span trace.
4. A list of span-like dicts — wrapped into a multi-span trace.
5. A ``str`` / ``bytes`` JSON document of any of the above.

``id`` / ``parent_id`` are aliased to ``span_id`` / ``parent_span_id`` so
semi-structured exports still parse.
"""

from __future__ import annotations

from typing import Any

from ..protocol import trace_from_dict
from ..types import CanonicalTrace, SourceProtocol
from .base import AdapterError, TraceAdapter


class RawAdapter(TraceAdapter):
    """Best-effort adapter for canonical / semi-structured payloads."""

    source_protocol = SourceProtocol.CUSTOM

    def adapt(self, payload: "dict | list | str | bytes | CanonicalTrace") -> CanonicalTrace:
        # 1. CanonicalTrace passthrough.
        if isinstance(payload, CanonicalTrace):
            return self._finalise(payload)

        # 2. JSON string / bytes.
        if isinstance(payload, (str, bytes)):
            payload = self._decode(payload)

        # 3. Canonical dict (trace with spans).
        if isinstance(payload, dict) and "spans" in payload and "trace_id" in payload:
            trace = trace_from_dict(payload)
            if trace.source_protocol == SourceProtocol.CUSTOM:
                trace.source_protocol = self.source_protocol
            return self._finalise(trace)

        # 4. Single span-like dict.
        if isinstance(payload, dict) and "trace_id" in payload:
            span = self._normalise_span(self._canonicalise_span_dict(payload))
            trace = CanonicalTrace(
                trace_id=span.trace_id,
                agent_id=span.agent_id or span.trace_id,
                agent_name=span.agent_name,
                session_id=span.session_id,
                source_protocol=self.source_protocol,
                spans=[span],
                metadata={"adapter": "raw"},
            )
            return self._finalise(trace)

        # 5. List of span-like dicts.
        if isinstance(payload, list):
            if not payload:
                raise AdapterError("raw payload is empty")
            trace_id = None
            for item in payload:
                if isinstance(item, dict) and item.get("trace_id"):
                    trace_id = item["trace_id"]
                    break
            if not trace_id:
                raise AdapterError("raw list missing trace_id")
            spans = [
                self._normalise_span(self._canonicalise_span_dict(s), trace_id=str(trace_id))
                for s in payload
                if isinstance(s, dict)
            ]
            if not spans:
                raise AdapterError("raw list contained no span dicts")
            trace = CanonicalTrace(
                trace_id=str(trace_id),
                agent_id=spans[0].agent_id or str(trace_id),
                agent_name=spans[0].agent_name,
                session_id=spans[0].session_id,
                source_protocol=self.source_protocol,
                spans=spans,
                metadata={"adapter": "raw"},
            )
            return self._finalise(trace)

        raise AdapterError(f"unsupported raw payload type: {type(payload).__name__}")

    @staticmethod
    def _canonicalise_span_dict(d: dict[str, Any]) -> dict[str, Any]:
        """Alias ``id`` -> ``span_id`` and ``parent_id`` -> ``parent_span_id``."""
        out = dict(d)
        if "span_id" not in out and "id" in out:
            out["span_id"] = out["id"]
        if "parent_span_id" not in out and "parent_id" in out:
            out["parent_span_id"] = out["parent_id"]
        return out
