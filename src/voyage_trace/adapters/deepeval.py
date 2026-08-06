"""Pull-side adapter for DeepEval metric results.

DeepEval (``confident-ai/deepeval``) is an LLM-as-judge / metric framework:
each ``TestResult`` carries ``name``/``success`` and a list of per-metric
scores. voyage_trace ingests those as a :class:`CanonicalTrace` with one
score span per metric, so a governance round can reason about evaluation
outcomes the same way it reasons about execution traces.

This adapter is **pull-only and SDK-free**: it parses the plain-dict shape
that ``deepeval``'s ``evaluate()`` returns (or its JSON serialisation). The
richer bidirectional helpers (push a governance outcome INTO a DeepEval
dataset; run live metrics) live in
:mod:`voyage_trace.integrations.deepeval` and import the SDK lazily.

Input format
------------
A dict with a ``trace_id`` (optional, defaults to a synthetic id) and one of:

* ``results``: a list of ``{"name": str, "success": bool,
  "metrics": [{"name", "score", "reason", "is_successful"}]}``, or
* ``metrics``: a flat list of ``{"name", "score", "reason",
  "is_successful", "threshold"}``.

Mapping
-------
* one ``TraceSpan`` per metric, ``operation_type=CHAT`` (it is an LLM-judged
  step), ``status=SUCCESS/FAILED`` from ``is_successful``;
* ``metric.score`` -> ``metadata.score``; ``metric.reason`` -> ``error``
  when unsuccessful; ``metric.name`` -> ``metadata.name``.
"""

from __future__ import annotations

from typing import Any

from ..types import CanonicalTrace, OperationType, SourceProtocol, SpanStatus, TraceSpan
from .base import AdapterError, TraceAdapter, _synthetic_id


class DeepEvalAdapter(TraceAdapter):
    """Convert DeepEval metric results into a :class:`CanonicalTrace`."""

    source_protocol = SourceProtocol.DEEPEVAL

    def adapt(self, payload: "dict | list | str | bytes") -> CanonicalTrace:
        data = self._decode(payload)
        if not isinstance(data, dict):
            raise AdapterError("deepeval payload must be a dict")

        trace_id = data.get("trace_id") or _synthetic_id("deepeval")
        agent_id = data.get("agent_id") or data.get("user_id") or "deepeval"
        raw_results = data.get("results") or []
        flat_metrics: list[dict[str, Any]] = list(data.get("metrics") or [])
        for res in raw_results:
            flat_metrics.extend(res.get("metrics") or [])

        spans: list[TraceSpan] = []
        for i, m in enumerate(flat_metrics):
            ok = bool(m.get("is_successful", m.get("success", True)))
            spans.append(TraceSpan(
                trace_id=trace_id,
                span_id=m.get("name") or f"metric-{i}",
                parent_span_id=None,
                operation_type=OperationType.CHAT,
                agent_id=agent_id,
                agent_name="deepeval",
                status=SpanStatus.SUCCESS if ok else SpanStatus.FAILED,
                metadata={
                    "name": m.get("name", ""),
                    "score": m.get("score"),
                    "threshold": m.get("threshold"),
                    "kind": "deepeval.metric",
                },
                error=None if ok else (m.get("reason") or "metric not successful"),
                source_protocol=SourceProtocol.DEEPEVAL,
            ))
        trace = CanonicalTrace(
            trace_id=trace_id, agent_id=agent_id, agent_name="deepeval",
            source_protocol=SourceProtocol.DEEPEVAL, spans=spans,
            metadata={"source": "deepeval", "metric_count": len(spans)},
        )
        return self._finalise(trace)
