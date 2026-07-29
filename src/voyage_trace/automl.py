"""AutoML as a tool for trace-driven modelling analysis.

This module wraps `AutoGluon <https://auto.gluon.ai/stable/index.html>`_
:class:`~autogluon.tabular.TabularPredictor` to turn a collection of
:class:`~voyage_trace.types.CanonicalTrace` objects into a learned model of
"what drives an outcome". AutoGluon handles model selection, hyperparameter
tuning, and ensembling; this module adapts its output to voyage_trace's
Markdown + Modification workflow.

AutoGluon is imported lazily inside :func:`run_automl` so that the module's
constants, data classes, and rendering functions remain importable without
AutoGluon installed — only the actual model training requires it.

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

* ``## Learned Signals`` — feature importances AutoGluon discovered, spliced
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

import tempfile
from dataclasses import dataclass, field
from typing import Any

from .execution_graph import ExecutionGraph, aggregate_execution_graph
from .simulator import Modification
from .types import CanonicalTrace


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
    """One fitted AutoGluon model.

    ``model_name`` is the AutoGluon model identifier (e.g.
    ``WeightedEnsemble_L2``). ``feature`` is the top feature from
    AutoGluon's permutation importance, or ``"(mean)"`` for the fallback
    mean baseline.
    """

    model_name: str
    feature: str
    r_squared: float
    rmse: float

    @property
    def is_baseline(self) -> bool:
        return self.feature == "(mean)"


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
    # {feature: importance in [0,1]}, normalised permutation importance.
    feature_importances: dict[str, float]
    top_cost_nodes: list[tuple[str, float]]  # (node_id, cost_usd)
    high_error_nodes: list[tuple[str, float]]  # (node_id, error_rate)
    suggested_modifications: list[tuple[Modification, str]]  # (mod, rationale)
    notes: list[str] = field(default_factory=list)

    @property
    def top_feature(self) -> str:
        """The most important feature (highest importance with target)."""
        if not self.feature_importances:
            return "(none)"
        return max(self.feature_importances, key=self.feature_importances.get)

    def to_dict(self) -> dict[str, Any]:
        return {
            "target": self.target,
            "n_samples": self.n_samples,
            "n_features": self.n_features,
            "best_model": {
                "model_name": self.best_model.model_name,
                "feature": self.best_model.feature,
                "r_squared": self.best_model.r_squared,
                "rmse": self.best_model.rmse,
            },
            "all_models": [
                {
                    "model_name": m.model_name,
                    "feature": m.feature,
                    "r_squared": m.r_squared,
                    "rmse": m.rmse,
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
# AutoML entry point (wraps AutoGluon TabularPredictor)
# --------------------------------------------------------------------------- #
def run_automl(
    traces: list[CanonicalTrace] | None = None,
    *,
    graph: ExecutionGraph | None = None,
    target: str = "cost_usd",
    cost_threshold: float = 0.01,
    error_threshold: float = 0.5,
    min_samples: int = 3,
    time_limit: float | None = None,
    **fit_kwargs: Any,
) -> AutoMLReport:
    """Run AutoML over a set of traces (or a pre-aggregated graph).

    Wraps :class:`autogluon.tabular.TabularPredictor` to learn which
    per-node features drive the target outcome. Pipeline:

    1. Build the :class:`FeatureMatrix` (one row per node — the same data
       as the MD ``## Nodes`` table) and convert to a ``pandas.DataFrame``.
    2. Fit AutoGluon's ``TabularPredictor`` (regression, ``r2`` metric)
       with cross-validation (``num_bag_folds=2``) so it works even with
       the tiny datasets typical of trace analysis.
    3. Extract permutation feature importances and normalise to ``[0, 1]``.
    4. Evaluate the best model in-sample for R² and RMSE.
    5. Surface top-cost / high-error nodes and turn them into candidate
       :class:`~voyage_trace.simulator.Modification` objects (to be
       validated later by :func:`~voyage_trace.simulator.simulate`).

    AutoGluon is imported lazily — if it is not installed, a
    :class:`ImportError` is raised with a helpful message.

    The candidate modifications are intentionally conservative: a
    ``swap_model`` is only suggested for a cost hotspot, and a
    ``cap_loops`` only for a high-error node. The simulator — not AutoML —
    decides whether a candidate is actually beneficial.
    """
    import pandas as pd

    # --- build feature matrix -------------------------------------------- #
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

    # Re-aggregate to get per-node cost/error for suggestions + trace count.
    if graph is None:
        assert traces is not None
        graph = aggregate_execution_graph(traces)
    n_traces = graph.observed_runs
    if n_traces < min_samples:
        notes.append(
            f"Only {n_traces} trace(s); AutoML results are directional, not statistically robust. "
            f"Aggregate at least {min_samples} traces before acting on importances."
        )

    # --- build DataFrame for AutoGluon ----------------------------------- #
    df_data: dict[str, list[float]] = {feat: matrix.column(feat) for feat in FEATURE_NAMES}
    df_data[target] = y
    df = pd.DataFrame(df_data)

    # --- fit AutoGluon --------------------------------------------------- #
    from autogluon.tabular import TabularPredictor

    all_models: list[TrainedModel] = []
    best_model: TrainedModel
    importances: dict[str, float]

    with tempfile.TemporaryDirectory() as model_dir:
        predictor = TabularPredictor(
            label=target,
            path=model_dir,
            problem_type="regression",
            eval_metric="r2",
            verbosity=0,
        )
        predictor.fit(
            df,
            num_bag_folds=2,
            num_bag_sets=1,
            num_stack_levels=0,
            time_limit=time_limit,
            raise_on_no_models_fitted=False,
            **fit_kwargs,
        )

        trained_names = predictor.model_names()
        if not trained_names:
            # AutoGluon could not train any model (e.g. degenerate data).
            notes.append(
                "AutoGluon could not train any model on the provided data; "
                "falling back to the mean baseline. Recommend collecting more traces."
            )
            best_model = TrainedModel(
                model_name="(mean)", feature="(mean)",
                r_squared=0.0, rmse=0.0,
            )
            importances = {feat: 0.0 for feat in FEATURE_NAMES}
        else:
            # --- feature importance (permutation) ---------------------- #
            raw_fi: dict[str, float] = {}
            try:
                fi_df = predictor.feature_importance(df, silent=True)
                for feat in FEATURE_NAMES:
                    if feat in fi_df.index:
                        raw_fi[feat] = abs(float(fi_df.loc[feat, "importance"]))
                    else:
                        raw_fi[feat] = 0.0
            except Exception:
                raw_fi = {feat: 0.0 for feat in FEATURE_NAMES}

            total_imp = sum(raw_fi.values())
            if total_imp > 0:
                importances = {k: v / total_imp for k, v in raw_fi.items()}
            else:
                importances = raw_fi  # all zeros

            # --- evaluate best model ----------------------------------- #
            scores = predictor.evaluate(df, auxiliary_metrics=True, silent=True)
            best_r2 = float(scores.get("r2", 0.0))
            # AutoGluon returns RMSE negated (higher-is-better convention).
            best_rmse = abs(float(scores.get("root_mean_squared_error", 0.0)))

            best_model_name = str(predictor.model_best)
            top_feat = max(importances, key=importances.get) if total_imp > 0 else "(mean)"

            best_model = TrainedModel(
                model_name=best_model_name,
                feature=top_feat,
                r_squared=best_r2,
                rmse=best_rmse,
            )

            # --- leaderboard → all_models ------------------------------ #
            lb = predictor.leaderboard(df, silent=True)
            for _, row in lb.iterrows():
                all_models.append(TrainedModel(
                    model_name=str(row["model"]),
                    feature=top_feat,
                    r_squared=float(row.get("score_test", 0.0)),
                    rmse=0.0,  # not available from leaderboard
                ))

    # --- no-signal honesty check ----------------------------------------- #
    if sum(importances.values()) == 0.0:
        notes.append(
            "No feature explained more variance than the mean baseline "
            "(all permutation importances are zero). "
            "Recommend collecting more traces before trusting any modification."
        )
        best_model = TrainedModel(
            model_name="(mean)", feature="(mean)",
            r_squared=0.0, rmse=best_model.rmse,
        )

    # --- node-level signals → candidate modifications -------------------- #
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

    return AutoMLReport(
        target=target,
        n_samples=n,
        n_features=len(FEATURE_NAMES),
        best_model=best_model,
        all_models=all_models,
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
    lines.append(f"- best_model: `{report.best_model.model_name}` "
                 f"(feature: `{report.best_model.feature}`, "
                 f"R^2={report.best_model.r_squared:.3f}, "
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
    lines.append("| model | feature | R^2 | RMSE |")
    lines.append("|---|---|---|---|")
    for m in report.all_models:
        lines.append(
            f"| {m.model_name} | {m.feature} | "
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
failure). It wraps `AutoGluon <https://auto.gluon.ai/stable/index.html>`_
TabularPredictor for model selection and ensembling. Use it deliberately.
Think step by step.

## When to call AutoML (and when NOT to)
1. Do you have >=3 traces of the SAME target agent? If NO -> do NOT call AutoML
   yet. With <3 samples the importances are directional only; instead, build
   the execution graph (descriptive MD) and stop. Record this decision as an
   `AnalysisStep(kind=MODEL)` with rationale "insufficient samples for AutoML".
2. Is there a concrete outcome to explain (total cost too high? a node failing
   repeatedly?). If NO -> call AutoML with the default target `cost_usd` purely
   to surface feature importances, but do NOT propose modifications.
3. If YES to both -> proceed.

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
3. If `best_model.feature == "(mean)"` or all importances are zero, record an
   `AnalysisStep(kind=MODEL)` with rationale "AutoML found no explanatory
   signal above the mean baseline" and SKIP proposal generation.

## Honesty contract
- Never present a correlation as causation. `feature_importances` are
  AutoGluon permutation importances normalised to [0, 1] — they rank
  associations, they do not prove a lever will work. Only the simulator's
  `project_savings` can do that.
- If `report.notes` warns about low sample size, echo that warning into the
  governance plan summary. Do not suppress it.
- Record EVERY call to AutoML as an `AnalysisStep` (inputs: target +
  trace_count; outputs: best_model + top_feature; artifacts: the enriched
  graph MD key in storage). This is the whole point of the AnalysisRecord.
"""
