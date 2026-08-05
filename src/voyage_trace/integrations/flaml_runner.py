"""FLAML as an alternative AutoML backend.

This module is a **drop-in alternative** to :func:`voyage_trace.automl.run_automl`
backed by `FLAML <https://microsoft.github.io/FLAML/>`_ instead of AutoGluon.
It accepts the same inputs (a list of :class:`CanonicalTrace` or an
:class:`ExecutionGraph`) and returns the same :class:`AutoMLReport`, so the
governance pipeline can switch backends by changing one call site.

Why FLAML
---------
FLAML and AutoGluon optimise for slightly different regimes:

* AutoGluon favours ensembling (``WeightedEnsemble_L2`` …) — high accuracy,
  higher compute.
* FLAML favours *cost-aware* model selection (its inner loop picks the model
  with the best accuracy-per-second) — usually much faster on tiny datasets,
  which is the trace-analysis regime.

For governance rounds running on a small fleet of agent traces, FLAML is a
useful second opinion: when AutoGluon and FLAML agree on the top feature,
that agreement is itself a signal. When they disagree, the disagreement is
worth recording on the report (the ``notes`` field) so a reviewer can see
both views.

Honesty contract
----------------
FLAML's permutation importances are normalised to ``[0, 1]`` exactly as in
:mod:`voyage_trace.automl`, the same hard-leakage guards apply (see
:func:`voyage_trace.automl.leakage_safe_features`), and the same
mean-predictor baseline is computed so ``beats_baseline`` has the same
meaning across backends. FLAML is imported lazily; when absent, an
``ImportError`` with a helpful message is raised (matching AutoGluon's
behaviour so callers can switch backends with no other code change).
"""

from __future__ import annotations

from typing import Any

from ..automl import (
    AutoMLReport,
    FEATURE_NAMES,
    MeanBaseline,
    TrainedModel,
    extract_feature_matrix,
    feature_matrix_from_graph,
    leakage_safe_features,
)
from ..execution_graph import ExecutionGraph, aggregate_execution_graph
from ..simulator import Modification
from ..types import CanonicalTrace


def _import_flaml() -> Any:
    """Lazily import FLAML's AutoML; raise a helpful ImportError when absent."""
    try:
        from flaml import AutoML  # type: ignore[import-not-found]
        return AutoML
    except ImportError as e:
        raise ImportError(
            "FLAML is not installed. Install it with `pip install 'flaml[automl]>=2.1'` "
            "(voyage-trace[integrations] also pulls it) and retry."
        ) from e


def run_automl_flaml(
    traces: list[CanonicalTrace] | None = None,
    *,
    graph: ExecutionGraph | None = None,
    target: str = "cost_usd",
    cost_threshold: float = 0.01,
    error_threshold: float = 0.5,
    min_samples: int = 3,
    time_limit: float | None = 30,
    eval_metric: str = "r2",
    **fit_kwargs: Any,
) -> AutoMLReport:
    """Run FLAML AutoML over a set of traces (or a pre-aggregated graph).

    Returns a :class:`voyage_trace.automl.AutoMLReport` — the same type
    :func:`voyage_trace.automl.run_automl` returns — so the downstream
    governance pipeline, Markdown rendering, and simulator workflow need no
    changes when switching backends.

    Differences vs AutoGluon backend:

    * The best model is FLAML's ``best_estimator`` (a string like
      ``"lgbm"`` / ``"xgboost"`` / ``"rf"``), recorded in
      :attr:`TrainedModel.model_name`.
    * Per-feature importances come from FLAML's
      ``model.estimator.feature_importances_`` when the underlying estimator
      exposes it (LGBM/XGB/RF do; linear models don't).
    * The mean-baseline ``beats_baseline`` gate uses the same definition:
      the best model's in-sample RMSE must be strictly smaller than
      ``std(target)`` (the mean-predictor's RMSE) — otherwise the importances
      are reported as noise.
    """
    import numpy as np
    import pandas as pd

    AutoML = _import_flaml()

    # --- build feature matrix (same as run_automl) ---------------------- #
    if graph is None:
        if not traces:
            raise ValueError("run_automl_flaml requires either `traces` or `graph`")
        matrix = extract_feature_matrix(traces)
    else:
        matrix = feature_matrix_from_graph(graph)

    if target not in matrix.targets:
        raise ValueError(
            f"unknown target {target!r}; expected one of {sorted(matrix.targets)}"
        )

    features = leakage_safe_features(target)
    dropped = tuple(f for f in FEATURE_NAMES if f not in features)

    y = matrix.target_column(target)
    n = len(y)
    notes: list[str] = []
    if dropped:
        notes.append(
            f"Dropped hard-leakage feature(s) {list(dropped)} for target "
            f"{target!r}: a feature that equals the target is identity leakage."
        )

    if graph is None:
        assert traces is not None
        graph = aggregate_execution_graph(traces)
    n_traces = graph.observed_runs
    if n_traces < min_samples:
        notes.append(
            f"Only {n_traces} trace(s); FLAML results are directional, not statistically robust. "
            f"Aggregate at least {min_samples} traces before acting on importances."
        )

    # --- mean-predictor baseline (honesty reference) --------------------- #
    y_arr = np.asarray(y, dtype=float)
    mean_baseline = MeanBaseline(
        rmse=float(np.std(y_arr, ddof=0)) if n > 0 else 0.0,
        mae=float(np.mean(np.abs(y_arr - np.mean(y_arr)))) if n > 0 else 0.0,
    )

    # --- build DataFrame for FLAML (leakage-safe features only) ----------- #
    df_data: dict[str, list[float]] = {feat: matrix.column(feat) for feat in features}
    df_data[target] = y
    df = pd.DataFrame(df_data)

    automl = AutoML()
    try:
        automl.fit(
            dataframe=df,
            label=target,
            task="regression",
            metric=eval_metric if eval_metric != "r2" else "r2",
            time_budget=time_limit,
            verbose=0,
            **fit_kwargs,
        )
    except Exception as e:  # noqa: BLE001 — FLAML can fail on degenerate data
        notes.append(
            f"FLAML could not train any model ({type(e).__name__}: {e}); "
            "falling back to the mean baseline. Recommend collecting more traces."
        )
        best_model = TrainedModel(
            model_name="(mean)", feature="(mean)", r_squared=0.0, rmse=0.0,
        )
        importances = {feat: 0.0 for feat in features}
        all_models: list[TrainedModel] = []
        best_mae = mean_baseline.mae
        best_rmse = mean_baseline.rmse
        best_r2 = 0.0
    else:
        best_estimator = str(automl.best_estimator or "(unknown)")
        # FLAML stores the underlying sklearn-style estimator.
        estimator = getattr(automl, "model", None)
        if estimator is not None and hasattr(estimator, "estimator"):
            estimator = estimator.estimator

        raw_fi: dict[str, float] = {feat: 0.0 for feat in features}
        if estimator is not None and hasattr(estimator, "feature_importances_"):
            try:
                fi = estimator.feature_importances_
                # feature_importances_ is aligned with the columns seen by fit.
                cols = getattr(estimator, "feature_names_in_", list(features))
                for i, name in enumerate(cols):
                    if name in raw_fi and i < len(fi):
                        raw_fi[name] = abs(float(fi[i]))
            except Exception:  # noqa: BLE001 — feature importance is best-effort
                pass

        total_imp = sum(raw_fi.values())
        importances = (
            {k: v / total_imp for k, v in raw_fi.items()} if total_imp > 0 else raw_fi
        )

        preds = automl.predict(df[list(features)])
        residuals = np.asarray(y, dtype=float) - np.asarray(preds, dtype=float)
        best_rmse = float(np.sqrt(np.mean(residuals ** 2)))
        best_mae = float(np.mean(np.abs(residuals)))
        ss_res = float(np.sum(residuals ** 2))
        ss_tot = float(np.sum((y_arr - y_arr.mean()) ** 2))
        best_r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0

        top_feat = max(importances, key=importances.get) if total_imp > 0 else "(mean)"
        best_model = TrainedModel(
            model_name=best_estimator,
            feature=top_feat,
            r_squared=best_r2,
            rmse=best_rmse,
        )
        all_models = [TrainedModel(
            model_name=best_estimator, feature=top_feat,
            r_squared=best_r2, rmse=best_rmse,
        )]

    # --- no-signal honesty check (matches AutoGluon backend) ------------- #
    if sum(importances.values()) == 0.0:
        notes.append(
            "No feature explained more variance than the mean baseline "
            "(all permutation importances are zero). "
            "Recommend collecting more traces before trusting any modification."
        )
        best_model = TrainedModel(
            model_name="(mean)", feature="(mean)",
            r_squared=0.0, rmse=best_rmse,
        )

    # --- mean-baseline gate (matches AutoGluon backend) ----------------- #
    if best_model.is_baseline:
        beats_baseline = False
    elif mean_baseline.rmse == 0.0:
        beats_baseline = False
    else:
        beats_baseline = best_model.rmse < mean_baseline.rmse
    if beats_baseline is False and not best_model.is_baseline:
        notes.append(
            "The best FLAML model did NOT beat the mean-predictor baseline on RMSE; "
            "treat its importances as noise and do not act on its suggestions."
        )

    # --- node-level signals → candidate modifications (same logic) ------ #
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
                note="FLAML: high error rate",
            ),
            f"Node {nid} has error rate {err * 100:.0f}%; candidate loop cap to limit blast radius.",
        ))
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
                note="FLAML: cost hotspot",
            ),
            f"Node {nid} is a cost hotspot (${cost:.4f}); candidate cheaper-model swap.",
        ))

    return AutoMLReport(
        target=target,
        n_samples=n,
        n_features=len(features),
        best_model=best_model,
        all_models=all_models,
        feature_importances=importances,
        top_cost_nodes=top_cost[:5],
        high_error_nodes=high_error[:5],
        suggested_modifications=suggestions,
        notes=notes,
        dropped_features=dropped,
        mean_baseline=mean_baseline,
        beats_baseline=beats_baseline,
        mae=best_mae,
        eval_metric=eval_metric,
    )
