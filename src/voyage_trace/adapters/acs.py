"""Pull-side adapter for safety / consistency scorer results ("ACS").

Note on naming (honesty)
------------------------
"ACS" has no single canonical OSS referent in the LLM-agent-eval space —
unlike DeepEval / Langfuse / FLAML. The two most plausible readings are:

1. **Azure Content Safety** (``azure-ai-contentsafety``) — a real, widely
   used safety scorer (prompt shields, jailbreak detection, hate/sexual/
   violence/self-harm severity).
2. **Agent Consistency Scoring** — a generic self-consistency / multi-run
   agreement metric (no canonical package).

This adapter treats ACS as a **generic safety/consistency verdict format**
so it works for both: it ingests a plain-dict verdict
(``{scores: [{category, severity, pass}], verdict, trace_id}``) into a
:class:`CanonicalTrace` with one score span per category. The richer
SDK-using scorer (Azure Content Safety) lives in
:mod:`voyage_trace.integrations.acs` and imports the SDK lazily; if your
"ACS" is an internal service, point that scorer at your endpoint instead.

Input format
------------
``{"trace_id": ..., "verdict": "safe"|"unsafe"|"skipped",
   "scores": [{"category": str, "severity": 0..7, "pass": bool}]}``

Mapping
-------
* one ``TraceSpan`` per score, ``operation_type=CHAT`` (a judged step),
  ``status=SUCCESS`` when ``pass`` else ``FAILED``;
* ``severity`` -> ``metadata.severity``; ``category`` -> ``metadata.name``;
  trace-level ``verdict`` -> ``trace.metadata.verdict``.
"""

from __future__ import annotations

from typing import Any

from ..types import CanonicalTrace, OperationType, SourceProtocol, SpanStatus, TraceSpan
from .base import AdapterError, TraceAdapter, _synthetic_id


class ACSAdapter(TraceAdapter):
    """Convert safety/consistency verdicts into a :class:`CanonicalTrace`."""

    source_protocol = SourceProtocol.ACS

    def adapt(self, payload: "dict | list | str | bytes") -> CanonicalTrace:
        data = self._decode(payload)
        if not isinstance(data, dict):
            raise AdapterError("acs payload must be a dict")

        trace_id = data.get("trace_id") or _synthetic_id("acs")
        agent_id = data.get("agent_id") or "acs"
        verdict = data.get("verdict", "unknown")
        scores = list(data.get("scores") or [])

        spans: list[TraceSpan] = []
        for i, s in enumerate(scores):
            ok = bool(s.get("pass", True))
            spans.append(TraceSpan(
                trace_id=trace_id,
                span_id=s.get("category") or f"score-{i}",
                parent_span_id=None,
                operation_type=OperationType.CHAT,
                agent_id=agent_id,
                agent_name="acs",
                status=SpanStatus.SUCCESS if ok else SpanStatus.FAILED,
                metadata={
                    "name": s.get("category", ""),
                    "severity": s.get("severity"),
                    "kind": "acs.safety_score",
                },
                error=None if ok else f"acs {s.get('category', '')} failed",
                source_protocol=SourceProtocol.ACS,
            ))
        trace = CanonicalTrace(
            trace_id=trace_id, agent_id=agent_id, agent_name="acs",
            source_protocol=SourceProtocol.ACS, spans=spans,
            metadata={"source": "acs", "verdict": verdict,
                      "score_count": len(spans)},
        )
        return self._finalise(trace)
