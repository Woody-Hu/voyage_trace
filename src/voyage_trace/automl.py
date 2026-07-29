"""AutoML as a tool for trace-driven modelling analysis.

This module exposes AutoML as a *tool* the meta-agent can call to turn a
collection of :class:`~voyage_trace.types.CanonicalTrace` objects into a
learned model of "what drives an outcome". It is deliberately
dependency-free (pure-Python statistics — no numpy / scikit-learn) so it
runs anywhere voyage_trace runs, and so its behaviour is fully auditable:
every number it produces can be reproduced with a calculator.

How AutoML matches the existing Markdown-based modelling approach
-----------------------------------------------------------------
The Markdown execution graph (see
:func:`~voyage_trace.execution_graph.render_markdown`) is a *descriptive*
model: its ``## Nodes`` table records, per node, the observed ``calls``,
``p50``/``p99`` latency, ``tokens``, ``cost`` and ``err%``. AutoML treats
that exact table as a **feature matrix** and learns which of those
columns drives a target outcome (cost or failure). The two views compose
into one loop:

    ExecutionGraph (MD)  ──►  AutoML feature matrix  ──►  learned model
            ▲                                                    │
            │                                                    ▼
    MD graph enriched          ◄──  suggested Modifications  ◄───┘
    (## Learned Signals,             (validated by simulator)
     ## Proposed Modifications)

* ``## Learned Signals`` — feature importances AutoML discovered, spliced
  back into the same MD document so a human reviewer sees the descriptive
  stats and the explanatory stats side by side.
* ``## Proposed Modifications`` — concrete
  :class:`~voyage_trace.simulator.Modification` objects derived from the
  learned importances, each to be validated by
  :func:`~voyage_trace.simulator.simulate` before it is accepted.

So AutoML does **not** replace the MD modelling — it *enriches* it with a
predictive/explanatory layer that the simulator then closes back into the
same Markdown artefact.

The agent-side guidance for *when* and *how* to call this tool lives in
:data:`AUTOML_COT_PROMPT` — a chain-of-thought prompt the modelling
sub-agent (see :mod:`voyage_trace.agents`) is seeded with.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from typing import Any

from .execution_graph import ExecutionGraph, aggregate_execution_graph
from .simulator import Modification
from .types import CanonicalTrace


# --------------------------------------------------------------------------- #
# Pure-Python statistics helpers (no numpy)
# --------------------------------------------------------------------------- #
def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def _variance(xs: list[float]) -> float:
    if len(xs) < 2:
        return 0.0
    m = _mean(xs)
    return sum((x - m) ** 2 for x in xs) / (len(xs) - 1)


def _std(xs: list[float]) -> float:
    return _variance(xs) ** 0.5


def _pearson(xs: list[float], ys: list[float]) -> float:
    """Pearson correlation, or 0.0 when undefined (degenerate input)."""
    n = min(len(xs), len(ys))
    if n < 2:
        return 0.0
    xs = xs[:n]
    ys = ys[:n]
    sx, sy = _std(xs), _std(ys)
    if sx == 0.0 or sy == 0.0:
        return 0.0
    mx, my = _mean(xs), _mean(ys)
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / (n - 1)
    return cov / (sx * sy)


def _linear_fit(xs: list[float], ys: list[float]) -> tuple[float, float]:
    """Ordinary least-squares fit ``y = slope * x + intercept``.

    Returns ``(0.0, mean(ys))`` when the fit is undefined (no variance in x).
    """
    n = min(len(xs), len(ys))
    if n < 2:
        return 0.0, _mean(ys)
    xs = xs[:n]
    ys = ys[:n]
    mx, my = _mean(xs), _mean(ys)
    denom = sum((x - mx) ** 2 for x in xs)
    if denom == 0.0:
        return 0.0, my
    slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / denom
    intercept = my - slope * mx
    return slope, intercept


def _r_squared(xs: list[float], ys: list[float], slope: float, intercept: float) -> float:
    """Coefficient of determination ``R^2`` of a linear fit.

    Can be negative (model worse than the mean) — that is the signal that
    the feature carries no explanatory power, which is exactly what
    feature-importance ranking needs.
    """
    n = min(len(xs), len(ys))
    if n < 2:
        return 0.0
    my = _mean(ys)
    ss_tot = sum((y - my) ** 2 for y in ys)
    if ss_tot == 0.0:
        # Target is constant: a mean predictor is "perfect" by convention.
        return 1.0 if all(y == my for y in ys) else 0.0
    ss_res = sum((y - (slope * x + intercept)) ** 2 for x, y in zip(xs, ys))
    return 1.0 - ss_res / ss_tot


def _rmse(xs: list[float], ys: list[float], slope: float, intercept: float) -> float:
    n = min(len(xs), len(ys))
    if n == 0:
        return 0.0
    return (sum((y - (slope * x + intercept)) ** 2 for x, y in zip(xs, ys)) / n) ** 0.5


# --------------------------------------------------------------------------- #
# Feature extraction
# --------------------------------------------------------------------------- #
# The canonical per-node feature set. These are exactly the columns of the
# execution-graph ``## Nodes`` table — by design, so the AutoML feature
# matrix and the Markdown node table are two views of the same numbers.
FEATURE_NAMES: tuple[str, ...] = (
    "calls",
    "p50_duration",
    "p99_duration",
    "total_tokens",
    "error_rate",
)

# Supported regression targets (per-node). ``cost_usd`` is the default
# governance target; ``error_rate`` is also useful (predict fragility).
TARGETS: tuple[str, ...] = ("cost_usd", "total_tokens", "total_duration_s")


@dataclass
class FeatureMatrix:
    """The feature matrix AutoML trains on — one row per graph node.

    ``rows`` is a list aligned with ``node_ids``; each row is a dict
    ``{feature_name: value}``. ``targets`` maps a target name to a list
    aligned with ``node_ids``.
    """

    node_ids: list[str]
    node_labels: list[str]
    rows: list[dict[str, float]]
    targets: dict[str, list[float]] = field(default_factory=dict)

    @property
    def n_samples(self) -> int:
        return len(self.rows)

    @property
    def n_features(self) -> int:
        return len(FEATURE_NAMES)

    def column(self, name: str) -> list[float]:
        if name in FEATURE_NAMES:
            return [r.get(name, 0.0) for r in self.rows]
        return self.targets.get(name, [])

    def target_column(self, target: str) -> list[float]:
        return self.targets.get(target, [])


def extract_feature_matrix(traces: list[CanonicalTrace]) -> FeatureMatrix:
    """Build a :class:`FeatureMatrix` from a collection of traces.

    Aggregates the traces into one template
    :class:`~voyage_trace.execution_graph.ExecutionGraph` (so one row per
    ``(operation_type, label)`` node), then turns each node's stats into a
    feature row. The resulting matrix is the same data shown in the
    execution-graph ``## Nodes`` table — AutoML and the MD document read
    from the same source of truth.
    """
    if not traces:
        raise ValueError("extract_feature_matrix requires at least one trace")
    graph = aggregate_execution_graph(traces)
    return feature_matrix_from_graph(graph)


def feature_matrix_from_graph(graph: ExecutionGraph) -> FeatureMatrix:
    """Build a :class:`FeatureMatrix` from an already-aggregated graph."""
    node_ids: list[str] = []
    node_labels: list[str] = []
    rows: list[dict[str, float]] = []
    cost_target: list[float] = []
    tokens_target: list[float] = []
    duration_target: list[float] = []
    for nid in sorted(graph.nodes):
        node = graph.nodes[nid]
        calls = max(node.calls, 1)  # guard against div-by-zero in derived rates
        p50 = node.p50_duration
        p99 = node.p99_duration
        tokens = node.input_tokens + node.output_tokens
        err_rate = node.error_rate
        node_ids.append(nid)
        node_labels.append(node.label)
        rows.append({
            "calls": float(node.calls),
            "p50_duration": float(p50),
            "p99_duration": float(p99),
            "total_tokens": float(tokens),
            "error_rate": float(err_rate),
        })
        cost_target.append(float(node.cost_usd))
        tokens_target.append(float(tokens))
        duration_target.append(float(node.total_duration_s))
    return FeatureMatrix(
        node_ids=node_ids,
        node_labels=node_labels,
        rows=rows,
        targets={
            "cost_usd": cost_target,
            "total_tokens": tokens_target,
            "total_duration_s": duration_target,
        },
    )


# --------------------------------------------------------------------------- #
# Model + report
# --------------------------------------------------------------------------- #
@dataclass
class TrainedModel:
    """One fitted model. Either a univariate linear regression on a single
    feature, or a constant mean baseline (when ``feature == "(mean)"``)."""

    feature: str
    slope: float
    intercept: float
    r_squared: float
    rmse: float

    @property
    def is_baseline(self) -> bool:
        return self.feature == "(mean)"

    def predict(self, x: float) -> float:
        return self.slope * x + self.intercept


@dataclass
class AutoMLReport:
    """The full output of :func:`run_automl`.

    Designed to be spliced back into an execution-graph Markdown document
    via :func:`render_automl_markdown` / :func:`inject_automl_into_graph_md`.
    """

    target: str
    n_samples: int
    n_features: int
    best_model: TrainedModel
    all_models: list[TrainedModel]
    # {feature: importance in [0,1]}, importance = |correlation| normalised.
    feature_importances: dict[str, float]
    top_cost_nodes: list[tuple[str, float]]  # (node_id, cost_usd)
    high_error_nodes: list[tuple[str, float]]  # (node_id, error_rate)
    suggested_modifications: list[tuple[Modification, str]]  # (mod, rationale)
    notes: list[str] = field(default_factory=list)

    @property
    def top_feature(self) -> str:
        """The most important feature (highest |correlation| with target)."""
        if not self.feature_importances:
            return "(none)"
        return max(self.feature_importances, key=self.feature_importances.get)

    def to_dict(self) -> dict[str, Any]:
        return {
            "target": self.target,
            "n_samples": self.n_samples,
            "n_features": self.n_features,
            "best_model": {
                "feature": self.best_model.feature,
                "slope": self.best_model.slope,
                "intercept": self.best_model.intercept,
                "r_squared": self.best_model.r_squared,
                "rmse": self.best_model.rmse,
            },
            "all_models": [
                {
                    "feature": m.feature, "slope": m.slope, "intercept": m.intercept,
                    "r_squared": m.r_squared, "rmse": m.rmse,
                }
                for m in self.all_models
            ],
            "feature_importances": self.feature_importances,
            "top_cost_nodes": self.top_cost_nodes,
            "high_error_nodes": self.high_error_nodes,
            "suggested_modifications": [
                {"modification": m.to_dict(), "rationale": r}
                for m, r in self.suggested_modifications
            ],
            "notes": self.notes,
        }


# --------------------------------------------------------------------------- #
# AutoML entry point
# --------------------------------------------------------------------------- #
def run_automl(
    traces: list[CanonicalTrace] | None = None,
    *,
    graph: ExecutionGraph | None = None,
    target: str = "cost_usd",
    cost_threshold: float = 0.01,
    error_threshold: float = 0.5,
    min_samples: int = 3,
) -> AutoMLReport:
    """Run AutoML over a set of traces (or a pre-aggregated graph).

    Pipeline:

    1. Build the :class:`FeatureMatrix` (one row per node — the same data
       as the MD ``## Nodes`` table).
    2. Fit a mean baseline + one univariate linear model per feature.
    3. Auto-select the best model by ``R^2`` (the mean baseline is the
       floor; if no feature beats it, the report says so honestly).
    4. Rank features by ``|correlation|`` with the target → importances.
    5. Surface top-cost / high-error nodes and turn them into candidate
       :class:`~voyage_trace.simulator.Modification` objects (to be
       validated later by :func:`~voyage_trace.simulator.simulate`).

    The candidate modifications are intentionally conservative: a
    ``swap_model`` is only suggested for a cost hotspot, and a
    ``cap_loops`` only for a high-call node. The simulator — not AutoML —
    decides whether a candidate is actually beneficial.
    """
    if graph is None:
        if not traces:
            raise ValueError("run_automl requires either `traces` or `graph`")
        matrix = extract_feature_matrix(traces)
    else:
        matrix = feature_matrix_from_graph(graph)

    if target not in matrix.targets:
        raise ValueError(
            f"unknown target {target!r}; expected one of {sorted(matrix.targets)}"
        )

    y = matrix.target_column(target)
    n = len(y)
    notes: list[str] = []

    # --- fit models ------------------------------------------------------- #
    # Mean baseline: slope 0, intercept = mean(y). R^2 of the mean predictor
    # is 0 by definition (it explains no variance beyond the mean).
    mean_y = _mean(y) if y else 0.0
    baseline = TrainedModel(
        feature="(mean)", slope=0.0, intercept=mean_y,
        r_squared=0.0,
        rmse=(statistics.pstdev(y) if y else 0.0),
    )
    models: list[TrainedModel] = [baseline]
    for feat in FEATURE_NAMES:
        x = matrix.column(feat)
        slope, intercept = _linear_fit(x, y)
        r2 = _r_squared(x, y, slope, intercept)
        rmse = _rmse(x, y, slope, intercept)
        models.append(TrainedModel(feature=feat, slope=slope, intercept=intercept,
                                   r_squared=r2, rmse=rmse))

    best = max(models, key=lambda m: m.r_squared)
    # If no feature beats the baseline, be explicit: AutoML did not find a
    # signal. This is the honest "no free lunch" outcome for tiny samples.
    if not best.is_baseline and best.r_squared <= 0.0:
        notes.append(
            f"No feature explained more variance than the mean baseline "
            f"(best R^2={best.r_squared:.3f} for {best.feature}). "
            f"Recommend collecting more traces before trusting any modification."
        )
        best = baseline

    # --- feature importances (|correlation|, normalised to sum to 1) ------ #
    importances: dict[str, float] = {}
    for feat in FEATURE_NAMES:
        x = matrix.column(feat)
        importances[feat] = abs(_pearson(x, y))
    total = sum(importances.values())
    if total > 0:
        importances = {k: v / total for k, v in importances.items()}

    # --- node-level signals → candidate modifications -------------------- #
    # Re-aggregate to get per-node cost/error in a stable order.
    if graph is None:
        assert traces is not None
        graph = aggregate_execution_graph(traces)
    top_cost = sorted(
        ((nid, graph.nodes[nid].cost_usd) for nid in graph.nodes),
        key=lambda kv: kv[1], reverse=True,
    )
    high_error = sorted(
        ((nid, graph.nodes[nid].error_rate) for nid in graph.nodes
         if graph.nodes[nid].error_rate > 0.0),
        key=lambda kv: kv[1], reverse=True,
    )

    suggestions: list[tuple[Modification, str]] = []
    seen_targets: set[str] = set()
    # High-error node → cap its loops so a failing path can't dominate a run.
    # Processed BEFORE cost hotspots so a node that is BOTH failing AND
    # expensive gets the guardrail (cap_loops), not a cheaper-model swap —
    # swapping a broken node to a cheaper model still leaves it broken.
    for nid, err in high_error:
        if err < error_threshold:
            continue
        if nid in seen_targets:
            continue
        seen_targets.add(nid)
        suggestions.append((
            Modification(
                target_node_id=nid, kind="cap_loops",
                params={"max_visits": 1},
                note="AutoML: high error rate",
            ),
            f"Node {nid} has error rate {err * 100:.0f}%; candidate loop cap to limit blast radius.",
        ))
    # Cost hotspot → try a cheaper model (0.3x cost / 0.8x tokens).
    for nid, cost in top_cost:
        if cost <= cost_threshold:
            continue
        if nid in seen_targets:
            continue
        seen_targets.add(nid)
        suggestions.append((
            Modification(
                target_node_id=nid, kind="swap_model",
                params={"cost_multiplier": 0.3, "token_multiplier": 0.8},
                note="AutoML: cost hotspot",
            ),
            f"Node {nid} is a cost hotspot (${cost:.4f}); candidate cheaper-model swap.",
        ))

    # The low-sample warning is about the number of TRACES observed (the
    # governance signal), not the number of nodes (rows in the matrix).
    # ``n`` is the row count; ``graph.observed_runs`` is the trace count.
    n_traces = graph.observed_runs
    if n_traces < min_samples:
        notes.append(
            f"Only {n_traces} trace(s); AutoML results are directional, not statistically robust. "
            f"Aggregate at least {min_samples} traces before acting on importances."
        )

    return AutoMLReport(
        target=target,
        n_samples=n,
        n_features=len(FEATURE_NAMES),
        best_model=best,
        all_models=models,
        feature_importances=importances,
        top_cost_nodes=top_cost[:5],
        high_error_nodes=high_error[:5],
        suggested_modifications=suggestions,
        notes=notes,
    )


# --------------------------------------------------------------------------- #
# Markdown rendering — splice back into the execution-graph document
# --------------------------------------------------------------------------- #
def render_automl_markdown(report: AutoMLReport) -> str:
    """Render an :class:`AutoMLReport` as Markdown sections.

    The output is a sequence of ``##`` sections (``## Learned Signals``,
    ``## Models``, ``## Suggested Modifications``) intended to be spliced
    into an execution-graph document by
    :func:`inject_automl_into_graph_md`, so the descriptive node table and
    the explanatory AutoML view live in one file.
    """
    lines: list[str] = []

    lines.append("## Learned Signals")
    lines.append(f"- target: `{report.target}`")
    lines.append(f"- samples: {report.n_samples}")
    lines.append(f"- best_model: `{report.best_model.feature}` "
                 f"(R^2={report.best_model.r_squared:.3f}, "
                 f"RMSE={report.best_model.rmse:.4f})")
    lines.append("- feature_importances:")
    for feat, imp in sorted(report.feature_importances.items(),
                            key=lambda kv: kv[1], reverse=True):
        lines.append(f"  - {feat}: {imp:.3f}")
    if report.top_cost_nodes:
        lines.append("- top_cost_nodes:")
        for nid, cost in report.top_cost_nodes:
            lines.append(f"  - {nid}: ${cost:.6f}")
    if report.high_error_nodes:
        lines.append("- high_error_nodes:")
        for nid, err in report.high_error_nodes:
            lines.append(f"  - {nid}: {err * 100:.1f}%")
    for note in report.notes:
        lines.append(f"- note: {note}")
    lines.append("")

    lines.append("## Models")
    lines.append("| feature | slope | intercept | R^2 | RMSE |")
    lines.append("|---|---|---|---|---|")
    for m in report.all_models:
        lines.append(
            f"| {m.feature} | {m.slope:.6f} | {m.intercept:.6f} | "
            f"{m.r_squared:.4f} | {m.rmse:.6f} |"
        )
    lines.append("")

    lines.append("## Suggested Modifications")
    if report.suggested_modifications:
        lines.append("| target | kind | params | rationale |")
        lines.append("|---|---|---|---|")
        for mod, rat in report.suggested_modifications:
            params = ",".join(f"{k}={v}" for k, v in mod.params.items())
            rat = rat.replace("|", "/")
            lines.append(f"| {mod.target_node_id} | {mod.kind} | {params} | {rat} |")
        lines.append("")
        lines.append("> Candidate modifications MUST be validated by "
                     "`simulator.simulate()` before being accepted into a "
                     "governance plan. AutoML proposes; the simulator disposes.")
    else:
        lines.append("- (no candidates — no node crossed the cost/error thresholds)")
    lines.append("")

    return "\n".join(lines)


def inject_automl_into_graph_md(graph_md: str, report: AutoMLReport) -> str:
    """Splice an :class:`AutoMLReport` into an execution-graph Markdown doc.

    Inserts the ``## Learned Signals`` / ``## Models`` /
    ``## Suggested Modifications`` sections right before the existing
    ``## Bottlenecks`` section (or appends them if there is none). The
    descriptive node table and the explanatory AutoML view thus share one
    document — closing the MD-graph ↔ AutoML loop.
    """
    section = render_automl_markdown(report)
    idx = graph_md.find("\n## Bottlenecks")
    if idx >= 0:
        return graph_md[:idx + 1] + section + graph_md[idx + 1:]
    return graph_md.rstrip() + "\n\n" + section


# --------------------------------------------------------------------------- #
# Agent chain-of-thought guidance
# --------------------------------------------------------------------------- #
AUTOML_COT_PROMPT = """\
You are the **modelling sub-agent** in the voyage_trace governance pipeline.
You have access to one tool — `voyage_trace.automl.run_automl` — that turns a
collection of traces into a learned model of what drives an outcome (cost or
failure). Use it deliberately. Think step by step.

## When to call AutoML (and when NOT to)
1. Do you have ≥3 traces of the SAME target agent? If NO → do NOT call AutoML
   yet. With <3 samples the importances are directional only; instead, build
   the execution graph (descriptive MD) and stop. Record this decision as an
   `AnalysisStep(kind=MODEL)` with rationale "insufficient samples for AutoML".
2. Is there a concrete outcome to explain (total cost too high? a node failing
   repeatedly?). If NO → call AutoML with the default target `cost_usd` purely
   to surface feature importances, but do NOT propose modifications.
3. If YES to both → proceed.

## How to call it
- Input: a list of `CanonicalTrace` (or a pre-aggregated `ExecutionGraph`).
- Pick `target`: `cost_usd` (default), `total_tokens`, or `total_duration_s`.
- Read `report.best_model.feature` and `report.feature_importances` FIRST.
- Read `report.notes` — they tell you when the signal is too weak to trust.

## How AutoML relates to the Markdown execution graph
The execution graph's `## Nodes` table (calls, p50, p99, tokens, cost, err%)
IS the AutoML feature matrix. They are two views of the SAME numbers:
- MD graph = *descriptive* (what happened).
- AutoML = *explanatory* (what drove it).
Your job is to MERGE them: call `inject_automl_into_graph_md(graph_md, report)`
so the human reviewer sees `## Learned Signals` and `## Suggested
Modifications` right next to the `## Nodes` table in one document.

## What to do with the output
1. For each item in `report.suggested_modifications`, create an
   `OptimizationProposal(modification=..., rationale=...)` and append it to the
   `AnalysisRecord` via `record.add_proposal(...)`. Record an
   `AnalysisStep(kind=PROPOSE)` per proposal with the rationale.
2. Do NOT accept proposals yourself. Hand them to the **simulation sub-agent**,
   which validates each with `simulator.simulate()` and fills in
   `expected_savings`. AutoML proposes; the simulator disposes.
3. If `best_model.feature == "(mean)"` or R^2 ≤ 0, record an
   `AnalysisStep(kind=MODEL)` with rationale "AutoML found no explanatory
   signal above the mean baseline" and SKIP proposal generation.

## Honesty contract
- Never present a correlation as causation. `feature_importances` are
  `|Pearson r|` normalised — they rank associations, they do not prove a
  lever will work. Only the simulator's `project_savings` can do that.
- If `report.notes` warns about low sample size, echo that warning into the
  governance plan summary. Do not suppress it.
- Record EVERY call to AutoML as an `AnalysisStep` (inputs: target +
  trace_count; outputs: best_model + top_feature; artifacts: the enriched
  graph MD key in storage). This is the whole point of the AnalysisRecord.
"""
