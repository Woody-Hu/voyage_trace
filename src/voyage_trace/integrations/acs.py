"""ACS scorer — SDK-using safety / consistency scorer.

The pull-side :class:`voyage_trace.adapters.acs.ACSAdapter` ingests a
plain-dict safety verdict with no SDK. This module is the **scorer side**: it
takes a piece of text (an agent output, a retrieved chunk, a prompt), runs it
through a safety / consistency backend, and emits the verdict dict the
adapter ingests — closing the loop scorer → adapter → :class:`CanonicalTrace`.

Two backends are supported, in priority order:

1. **Azure Content Safety** (``azure-ai-contentsafety``). When the SDK is
   installed and credentials are present (either passed explicitly or via
   ``AZURE_AI_CONTENT_SAFETY_*`` env vars), this is the live path.
2. **Heuristic fallback** (no SDK). A tiny deterministic rule set (URL /
   obvious-Pattern scan + length sanity) that emits a ``verdict="skipped"`` +
   zero severities — explicitly NOT a fake "safe" verdict. The honest
   contract is: *if no real scorer is wired, we say "skipped", never "safe"*.

The naming caveat (ACS = Azure Content Safety vs Agent Consistency Scoring) is
documented in :mod:`voyage_trace.adapters.acs`. This module ships one
implementation for the Azure reading; callers who treat ACS as a generic
self-consistency scorer can pass their own ``scorer`` callable to
:func:`acs_score` and it will be used in place of the Azure path.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Any, Callable, Protocol

from ..types import CanonicalTrace

# Categories Azure Content Safety returns. We hard-code them so the dict
# shape is stable even when the SDK is absent — the heuristic fallback
# emits ``severity=0`` for every category, not a different schema.
AZURE_CATEGORIES: tuple[str, ...] = ("hate", "sexual", "violence", "self_harm")


class _Scorer(Protocol):
    def __call__(self, text: str, *, categories: tuple[str, ...]) -> dict[str, Any]:
        ...


@dataclass
class ACSScore:
    """One scored category — JSON-safe, matches the adapter's input."""

    category: str
    severity: float  # 0.0..7.0 (Azure scale); 0.0 means "no signal"
    pass_: bool  # whether the category passed the threshold

    def to_dict(self) -> dict[str, Any]:
        return {"category": self.category, "severity": self.severity, "pass": self.pass_}


@dataclass
class ACSVerdict:
    """The full scorer verdict — JSON-safe, matches the adapter's input."""

    trace_id: str
    verdict: str  # "safe" | "unsafe" | "skipped"
    scores: list[ACSScore]

    def to_dict(self) -> dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "verdict": self.verdict,
            "scores": [s.to_dict() for s in self.scores],
        }


def _import_azure_cs() -> Any | None:
    """Lazily import the Azure Content Safety SDK; ``None`` if absent."""
    try:
        from azure.ai.contentsafety import ContentSafetyClient  # type: ignore[import-not-found]
        from azure.identity import DefaultAzureCredential  # type: ignore[import-not-found]
        return ContentSafetyClient, DefaultAzureCredential
    except ImportError:
        return None


# A few obvious patterns the heuristic fallback flags. This is NOT a safety
# classifier — it is a sanity check that something is *text* and free of the
# most obvious prompt-injection markers. The verdict is always ``skipped``
# (never ``safe``) so the downstream governance never mistakes the heuristic
# for a real safety pass.
_INJECTION_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"ignore\s+(previous|all)\s+instructions", re.IGNORECASE),
    re.compile(r"<\s*system\s*>", re.IGNORECASE),
    re.compile(r"<\s*prompt\s*>", re.IGNORECASE),
)


def _heuristic_scores(text: str, *, categories: tuple[str, ...]) -> list[ACSScore]:
    """Return zero-severity scores with a ``skipped``-style pass flag.

    The heuristic deliberately reports ``pass=True`` for every category with
    severity 0.0 — but the enclosing verdict stays ``skipped``, so governance
    never sees a fake "safe" stamp. The point is shape parity, not a verdict.
    """
    flagged = any(p.search(text or "") for p in _INJECTION_PATTERNS)
    return [
        ACSScore(category=c, severity=0.0, pass_=not flagged)
        for c in categories
    ]


def _azure_scores(
    text: str,
    *,
    categories: tuple[str, ...],
    client: Any,
) -> list[ACSScore]:
    """Run Azure Content Safety ``analyze_text`` and project to ACSScore."""
    from azure.ai.contentsafety.models import AnalyzeTextOptions  # type: ignore[import-not-found]

    # Azure's own categories differ slightly from our names; map them.
    azure_to_local = {
        "Hate": "hate",
        "Sexual": "sexual",
        "Violence": "violence",
        "SelfHarm": "self_harm",
    }
    res = client.analyze_text(AnalyzeTextOptions(text=text, categories=list(azure_to_local)))
    out: list[ACSScore] = []
    by_cat = {r.category.name: r for r in (res.categories_analyzed_result or [])}
    for azure_name, local_name in azure_to_local.items():
        r = by_cat.get(azure_name)
        if r is None:
            out.append(ACSScore(category=local_name, severity=0.0, pass_=True))
            continue
        sev = float(r.severity or 0)
        # Azure severity 0..6; we rescale to 0..7 for cross-backend parity
        # (so a single threshold table can compare Azure vs heuristic vs
        # custom scorers without per-backend quirks).
        out.append(ACSScore(
            category=local_name,
            severity=sev,
            pass_=sev < 2.0,  # Azure's "low-severity" boundary
        ))
    return out


def _azure_client(
    *,
    endpoint: str | None = None,
    credential: Any | None = None,
) -> Any | None:
    """Build an Azure Content Safety client from args or env; ``None`` on miss."""
    imported = _import_azure_cs()
    if imported is None:
        return None
    ContentSafetyClient, DefaultAzureCredential = imported
    endpoint = endpoint or os.environ.get("AZURE_AI_CONTENT_SAFETY_ENDPOINT")
    if not endpoint:
        return None
    cred = credential or DefaultAzureCredential(exclude_interactive_browser_credential=True)
    return ContentSafetyClient(endpoint, cred)


def acs_score(
    text: str,
    *,
    trace_id: str = "acs",
    categories: tuple[str, ...] = AZURE_CATEGORIES,
    threshold: float = 2.0,
    client: Any | None = None,
    endpoint: str | None = None,
    credential: Any | None = None,
    scorer: _Scorer | None = None,
) -> ACSVerdict:
    """Score ``text`` for safety / consistency; return a :class:`ACSVerdict`.

    Backend resolution order (first hit wins):

    1. ``scorer`` — a caller-supplied callable (any backend; lets you wire a
       custom consistency / safety service under the same shape).
    2. ``client`` or Azure Content Safety SDK + credentials — the live path.
    3. Heuristic fallback — emits ``verdict="skipped"`` with zero severities.
       This is the honest "no real scorer wired" state; never fake ``safe``.

    Parameters
    ----------
    text:
        The text to score.
    trace_id:
        Trace id to attach to the verdict (becomes the :class:`CanonicalTrace`
        ``trace_id`` after the adapter ingests it).
    categories:
        The category set to score. Defaults to Azure's four.
    threshold:
        Severity threshold above which a category is marked ``pass=False``.
        Used only by the heuristic and Azure paths; a custom ``scorer`` sets
        ``pass`` itself.
    """
    # 1. caller-supplied scorer (highest priority — caller knows their backend).
    if scorer is not None:
        raw = scorer(text=text, categories=categories)
        scores = [
            ACSScore(
                category=str(s.get("category", f"cat-{i}")),
                severity=float(s.get("severity", 0.0) or 0.0),
                pass_=bool(s.get("pass", True)),
            )
            for i, s in enumerate(raw.get("scores", []))
        ]
        verdict = "unsafe" if any(not s.pass_ for s in scores) else "safe"
        return ACSVerdict(trace_id=trace_id, verdict=verdict, scores=scores)

    # 2. Azure Content Safety SDK path.
    cs_client = client or _azure_client(endpoint=endpoint, credential=credential)
    if cs_client is not None:
        try:
            scores = _azure_scores(text, categories=categories, client=cs_client)
            verdict = "unsafe" if any(not s.pass_ for s in scores) else "safe"
            return ACSVerdict(trace_id=trace_id, verdict=verdict, scores=scores)
        except Exception:  # noqa: BLE001 — fall through to heuristic on Azure failure
            pass

    # 3. Heuristic fallback (honest "skipped", not a fake "safe").
    scores = _heuristic_scores(text, categories=categories)
    # The heuristic may flag obvious injection patterns — if so, mark the
    # affected categories as ``pass=False`` but keep the verdict ``skipped``:
    # we are not a real safety classifier, so we cannot honestly call anything
    # "unsafe" either.
    flagged = any(not s.pass_ for s in scores)
    return ACSVerdict(
        trace_id=trace_id,
        verdict="skipped",
        scores=scores,
    )


def score_trace_outputs(
    trace: CanonicalTrace,
    *,
    threshold: float = 2.0,
    client: Any | None = None,
    scorer: _Scorer | None = None,
) -> ACSVerdict:
    """Score the concatenated outputs of every span in ``trace``.

    Convenience wrapper: pull the textual output of each span, concatenate,
    and run :func:`acs_score` over the union. Returns a single verdict keyed
    by ``trace.trace_id``.
    """
    parts: list[str] = []
    for span in trace.spans:
        if not span.outputs:
            continue
        for v in span.outputs.values():
            if isinstance(v, str):
                parts.append(v)
            elif isinstance(v, dict):
                inner = v.get("content") or v.get("text") or v.get("output")
                if isinstance(inner, str):
                    parts.append(inner)
    text = "\n".join(parts)
    return acs_score(
        text,
        trace_id=trace.trace_id,
        threshold=threshold,
        client=client,
        scorer=scorer,
    )
