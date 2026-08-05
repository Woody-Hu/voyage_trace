"""DeepEval integration — push governance outcomes, pull metric results.

Two directions, both lazily importing the ``deepeval`` SDK:

* :func:`to_deepeval_dataset` — PUSH a :class:`CanonicalTrace` (typically
  the trace a governance plan was produced from) into a DeepEval
  ``EvaluationDataset`` of ``LLMTestCase`` objects, so DeepEval's
  LLM-as-judge metrics can score the agent run. When the SDK is absent it
  returns the **same data as a JSON-safe list of dicts** (the DeepEval
  field names are stable strings), so the artefact can be saved and loaded
  into DeepEval later without re-deriving it.
* :func:`from_deepeval_results` — PULL DeepEval ``TestResult`` objects (or
  their dict form) back into a :class:`CanonicalTrace` of score spans,
  delegating to :class:`voyage_trace.adapters.deepeval.DeepEvalAdapter` for
  the dict form (no SDK needed).

The integration never edits the canonical schema; it is a lossless
translation at the boundary.
"""

from __future__ import annotations

from typing import Any

from ..adapters.deepeval import DeepEvalAdapter
from ..types import CanonicalTrace


def _import_deepeval() -> Any | None:
    """Lazily import deepeval; return ``None`` if it is not installed."""
    try:
        import deepeval  # type: ignore[import-not-found]
        from deepeval.test_case import LLMTestCase  # type: ignore[import-not-found]
        return deepeval, LLMTestCase
    except ImportError:
        return None


def to_deepeval_dataset(
    trace: CanonicalTrace,
    *,
    expected_output: str | None = None,
    retrieval_context: list[str] | None = None,
) -> Any:
    """Push ``trace`` into a DeepEval evaluation dataset.

    Each :class:`CanonicalTrace` becomes one ``LLMTestCase``: ``input``
    comes from the root span's inputs, ``actual_output`` from the final
    span's output, ``expected_output`` from the caller (a governance
    golden / contract), and ``retrieval_context`` from retrieval spans.

    Returns a ``deepeval.EvaluationDataset`` when the SDK is present, else a
    JSON-safe ``list[dict]`` with the same fields — loadable into DeepEval
    later without re-deriving the trace mapping.
    """
    root = trace.root_span
    spans = trace.sorted_spans()
    actual_output = ""
    for span in reversed(spans):
        if span.outputs:
            actual_output = str(span.outputs.get("output") or span.outputs.get("result") or "")
            if actual_output:
                break
    input_text = ""
    if root and root.inputs:
        input_text = str(root.inputs.get("input") or root.inputs.get("query") or "")
    rc = list(retrieval_context or [])
    if not rc:
        rc = [
            str(s.outputs.get("content") or s.outputs)
            for s in spans
            if s.operation_type.value == "retrieval" and s.outputs
        ]

    case_dict = {
        "input": input_text,
        "actual_output": actual_output,
        "expected_output": expected_output or "",
        "retrieval_context": rc,
        "context": [s.metadata.get("name", "") for s in spans if s.metadata],
    }

    imported = _import_deepeval()
    if imported is None:
        # Graceful: emit the JSON-safe artefact; skip live dataset build.
        return [case_dict]
    _deepeval, LLMTestCase = imported
    try:
        case = LLMTestCase(
            input=case_dict["input"],
            actual_output=case_dict["actual_output"],
            expected_output=case_dict["expected_output"] or None,
            retrieval_context=case_dict["retrieval_context"] or None,
            context=case_dict["context"] or None,
        )
        try:
            from deepeval.dataset import EvaluationDataset  # type: ignore[import-not-found]
            ds = EvaluationDataset()
            ds.add_test_cases(case) if hasattr(ds, "add_test_cases") else ds.add_test_cases([case])
            return ds
        except ImportError:
            return [case]
    except Exception:  # noqa: BLE001 — degrade rather than crash the governance round
        return [case_dict]


def from_deepeval_results(results: Any, *, trace_id: str | None = None) -> CanonicalTrace:
    """Pull DeepEval ``TestResult`` objects / dicts into a :class:`CanonicalTrace`.

    Accepts either real DeepEval ``TestResult`` objects (their public
    attributes are read) or the plain-dict shape
    (``{"results": [{"metrics": [...]}]}``). Dicts are delegated to
    :class:`DeepEvalAdapter` (no SDK needed); objects are normalised into
    that dict shape first.
    """
    # Plain-dict / JSON shape — delegate straight to the SDK-free adapter.
    if isinstance(results, (dict, str, bytes)):
        payload = dict(results) if isinstance(results, dict) else results
        if trace_id and isinstance(payload, dict):
            payload.setdefault("trace_id", trace_id)
        return DeepEvalAdapter().adapt(payload)

    if isinstance(results, list):
        # A list of plain dicts can go straight through the adapter; a list of
        # real TestResult objects needs attribute-reading first.
        if all(isinstance(r, dict) for r in results):
            payload = {"results": results}
            if trace_id:
                payload.setdefault("trace_id", trace_id)
            return DeepEvalAdapter().adapt(payload)
        results = list(results)
    else:
        results = [results]

    # Object form (TestResult / Metric instances): normalise to the dict shape.
    metrics: list[dict[str, Any]] = []
    for res in results:
        for m in getattr(res, "metrics", []) or []:
            metrics.append({
                "name": getattr(m, "name", ""),
                "score": getattr(m, "score", None),
                "reason": getattr(m, "reason", ""),
                "threshold": getattr(m, "threshold", None),
                "is_successful": getattr(m, "is_successful", lambda: True)(),
            })
    return DeepEvalAdapter().adapt({"trace_id": trace_id or "deepeval", "metrics": metrics})
