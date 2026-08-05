# AutoML (AutoGluon) Best Practices & Evaluation Plan

> How `voyage_trace.automl` uses AutoGluon `TabularPredictor` to model
> "what drives cost / failure in LLM-agent traces" — and the honest
> evaluation protocol that keeps it from cheating on the tiny datasets
> typical of trace analysis (often 3–20 rows, one per aggregated
> execution-graph node).

This document is the reference for the modelling choices in
[`src/voyage_trace/automl.py`](../src/voyage_trace/automl.py) and the
evaluation tests in
[`tests/test_automl.py`](../tests/test_automl.py). It is intentionally
normative: any change to the AutoML pipeline must update both this doc and
the tests.

## 1. The problem regime

| Property | Value |
|---|---|
| Unit of observation | one **aggregated execution-graph node** (`<operation_type>:<label>`) |
| Features (`FEATURE_NAMES`) | `calls`, `p50_duration`, `p99_duration`, `total_tokens`, `error_rate` |
| Targets (`TARGETS`) | `cost_usd`, `total_tokens`, `total_duration_s` |
| Typical `n_samples` | **3–20** (tiny) |
| Feature matrix source | the same numbers as the Markdown `## Nodes` table — two views of one truth |
| Goal | *explanatory* (rank what drives an outcome), not predictive deployment |

Tiny `n` dominates every design decision below: variance is high,
significance is essentially unattainable below `n≈10`, and the biggest risk
is **believing a model that has learned nothing**.

## 2. AutoGluon configuration (and why)

`run_automl()` fits `TabularPredictor` with these defaults:

| Knob | Value | Rationale |
|---|---|---|
| `problem_type` | `"regression"` | All targets are continuous. |
| `eval_metric` | `"r2"` (default; `mae`/etc. via param) | `r2` matches existing tests; switch to `mean_absolute_error` for tuning when the target is heavy-tailed (research recommends MAE for tiny skewed data — R² is outlier-sensitive and inflates with features). |
| `num_bag_folds` | `2` | Low on purpose: more folds over-fit on `n<20`. AutoGluon docs warn values >10 can harm results via overfitting. |
| `num_bag_sets` | `10` (default) | The **variance-reduction knob** the docs recommend *instead of* raising folds. `2 folds × 10 sets = 20` models per base learner; cheap on tiny data, shrinks the CV-score variance that would otherwise make `model_best` a coin-flip. |
| `num_stack_levels` | `0` | Stacking learns meta-weights from a handful of OOF points → over-fits on tiny `n`. Docs warn stacked overfitting can "make multi-layer stacking fail drastically." |
| `time_limit` | `None` | On tiny data training is seconds; the limit is moot. Note: with `time_limit=None`, AutoGluon does **not** repeatedly bag unless `num_bag_sets` is set — which is exactly why we set it. |
| `hyperparameter_tune_kwargs` | **not used** | Official tutorial: *"We don't recommend hyperparameter-tuning with AutoGluon in most cases."* HPO on `n<20` is essentially guaranteed to over-fit the validation split. |
| `raise_on_no_models_fitted` | `False` | Degenerate data (e.g. constant target) must not crash the pipeline — it falls back to the mean baseline honestly. |

**Presets.** We do **not** pass `presets=` because every preset either sets
`auto_stack=True` (which may re-introduce stacking) or carries a
`time_limit=3600` expectation (`best_quality`). Instead we pin the three
knobs above explicitly. The closest sane preset would be
`medium_quality_faster_train` (the default) — but pinning is clearer.

> **Future option (not enabled).** AutoGluon 1.5 adds foundational models
> (`extreme_quality`: TabPFNv2 / Mitra / TabICL) that are *"significantly
> more accurate than best_quality on datasets ≤ 100,000 samples"* and
> excel on small/medium data via in-context learning (no over-fit-prone
> HPO). These are the single biggest accuracy lever for this regime —
> enable when a GPU is available and `autogluon.tabular[tabpfn]` is
> installable.

## 3. The anti-cheating boundary: leakage protection

This is the most important honesty guarantee in the pipeline.

A feature that **equals** the target (or a deterministic transform of it)
would let AutoGluon score a trivial `R²=1` by identity rather than by
learning. Predicting `total_tokens` from a `total_tokens` feature is pure
leakage — leaving it in would be cheating.

[`leakage_safe_features(target)`](../src/voyage_trace/automl.py) drops the
hard-leakage features per target before the DataFrame is built:

| Target | Dropped features | Reason |
|---|---|---|
| `total_tokens` | `total_tokens` | identity — the feature *is* the target |
| `cost_usd` | (none) | `total_tokens` is **soft** leakage (see below), kept |
| `total_duration_s` | (none) | `p50`/`p99` are percentiles, not components of the sum |

**Soft leakage (kept, but flagged).** `total_tokens` is highly correlated
with `cost_usd` (cost ≈ tokens × price). It is **kept** because tokens
genuinely drive cost, but the report carries a `SOFT_LEAKAGE_NOTES`
caveat and the reviewer is told to treat the cost-driver importance as
**directional**, not causal. Removing `total_tokens` from the cost model
would make the model less "cheaty" but also less useful — the honest
middle ground is to keep it and document the caveat.

Every dropped feature is recorded on `AutoMLReport.dropped_features` and
rendered in the Markdown (`dropped_features (hard leakage): [...]`) so the
anti-cheating posture is visible in the artefact, not hidden in code.

## 4. The "is there any signal?" gate: mean-predictor baseline

A **mean-predictor** (predict the training mean) has `R²=0` by
construction and `RMSE = std(target)`. It is the trivial reference: any
model that does not beat it has learned nothing usable, no matter how
positive its in-sample `R²` looks (in-sample `R²` is monotonically
inflated by adding features, even pure noise — on tiny `n` this is acute).

`run_automl()` computes a `MeanBaseline(rmse, mae)` and sets
`AutoMLReport.beats_baseline`:

- `True` → `best_model.rmse < mean_baseline.rmse` (real signal).
- `False` → the model did not beat "predict mean"; a note is appended
  telling the reviewer to **treat its importances as noise** and not act
  on its suggestions.
- `False` (constant target) → `mean_baseline.rmse == 0` → every model
  ties → no signal.

This gate composes with the existing "no permutation importance > 0"
fallback (`best_model.is_baseline`) into a two-layer honesty check.

## 5. Evaluation protocol (how to read a run)

Because `n` is tiny, **no single number is trustworthy**. A run is
evaluated as a small, honest panel:

1. **`best_model.r_squared`** — in-sample, **optimistic**. Treat as an
   upper bound, never as the headline. Undefined/negative on held-out data
   when the model is worse than the mean.
2. **`best_model.rmse`** — same units as the target; sensitive to
   outliers (relevant for `cost_usd`, which is heavy-tailed).
3. **`best_model.mae`** — robust, interpretable ("±$X average error");
   the **preferred primary metric** for tiny, outlier-prone data.
4. **`mean_baseline`** — `RMSE=std(target)`, `MAE=mean(|y-ȳ|)`. The
   "no-signal" reference; `beats_baseline` is the headline verdict.
5. **`feature_importances`** — normalised permutation importance in
   `[0,1]`; **directional only**. Do not present as causal.

**Honest caveats baked into the report (`notes`):**
- low-sample warning when `observed_runs < min_samples` (default 3);
- soft-leakage caveat for `cost_usd`;
- "did not beat baseline" warning;
- "all importances zero → mean baseline" fallback.

**What we do NOT claim at this `n`:**
- Statistical significance. Below `n≈10` permutation-importance p-values
  are severely underpowered; we report them only as exploratory.
- Generalisation. AutoGluon's internal `score_val` (2-fold CV) is a
  **training diagnostic**, not a generalisation estimate.

> **Stronger evaluation (optional, for production audits).** When you need
> a defensible generalisation number, run **leave-one-out CV** (`K=N`) and
> report the mean ± std across repeats. On tiny `n` LOOCV is the most
> defensible single estimate (uses all data, no fold-size arbitrariness);
> pair it with repeated-2-fold (≥20 repeats) for a variance estimate.
> This is intentionally **not** in the hot path — it retrains `N` times.

## 6. AutoGluon sign conventions (don't misread them)

Two conventions that have bitten every AutoGluon user at least once:

1. **`leaderboard` / `evaluate` flip the sign of lower-is-better metrics.**
   *"Metrics scores always show in higher-is-better form. Metrics such as
   log_loss and root_mean_squared_error will have their signs FLIPPED, and
   values will be negative."* `run_automl` reads RMSE/MAE via `abs(...)`.
2. **Negative permutation importance = a harmful feature** (the model
   would do better *without* it), **not** "negatively correlated with the
   target." `run_automl` uses `abs(...)` for ranking because we only care
   about *magnitude* of influence, but the raw sign is meaningful when
   inspecting the leaderboard directly.

## 7. Permutation-importance pitfalls on this data

- **Multicollinearity.** `p50_duration`/`p99_duration`/`total_duration_s`
  and `total_tokens`/`cost_usd` are correlated blocks. Permutation
  importance **under-states** the importance of correlated features
  (shuffling one leaves its signal in the others) — expect
  counter-intuitive rankings. For correlated blocks prefer
  group/joint permutation (not yet wired; see §10 roadmap).
- **Computed on training data by default.** AutoGluon warns importance
  is *biased when computed on training data*; on tiny `n` we accept this
  (we have no spare data to hold out) and flag it as directional.
- **`subsample_size` default 5000.** On `n<5000` all rows are used, so
  the subsample knob is moot; to shrink CIs crank `num_shuffle_sets`
  (default 5; ≥10 recommended) — wiring this is on the roadmap (§10).

## 8. From AutoML to governance: who decides what

AutoML **proposes**; the **simulator disposes**; **verification** closes
the loop. Concretely:

1. `run_automl()` → `AutoMLReport.suggested_modifications` (candidate
   `Modification`s: `swap_model` for cost hotspots, `cap_loops` for
   high-error nodes). These are *conservative* — a swap is only suggested
   for a node above `cost_threshold`, a cap only above
   `error_threshold`.
2. `SimulationAgent` validates each candidate with
   `simulator.simulate_graph()` and fills `expected_savings`. A candidate
   is `validated` only if the simulator did not diverge **and** projected
   savings are non-negative.
3. `GovernanceAgent` accepts only validated proposals whose (calibrated)
   projected savings clear `min_savings_usd`.
4. `VerificationAgent` compares projected vs **actual** post-deployment
   savings (real graph arithmetic, never the simulator's own output) and
   folds the gap into the calibration multiplier `τ` for the next round.

AutoML never decides acceptance. This separation is the structural
guarantee that an over-optimistic AutoML run cannot push a bad change into
production on its own.

## 9. Evaluation tests

[`tests/test_automl.py`](../tests/test_automl.py) covers, beyond the
baseline functional tests:

- **`TestLeakageProtection`** — `total_tokens` is dropped when it is the
  target; `cost_usd` keeps `total_tokens` (soft) but carries the caveat;
  `dropped_features` is recorded and rendered in Markdown.
- **`TestMeanBaseline`** — the mean-predictor baseline is computed;
  `beats_baseline` is `True` when there is real signal and `False` on a
  constant-target / no-signal dataset; the "did not beat baseline" note
  fires.
- **`TestEvaluationMetrics`** — `mae` and `eval_metric` are present on
  the report and survive a JSON round-trip; the rendered Markdown shows
  MAE + metric.

These tests are **honest**: they use synthetic traces with known cost
structure, assert directional (not fabricated) outcomes, and explicitly
check the "no signal" path rather than assuming AutoML always wins.

## 10. Roadmap (not yet implemented; tracked here for honesty)

- [ ] `num_shuffle_sets` knob on `feature_importance` (≥20) for tighter CIs.
- [ ] Group/joint permutation importance for correlated feature blocks.
- [ ] Optional LOOCV / repeated-k-fold evaluation helper (off the hot path).
- [ ] Foundational models (`extreme_quality`) when a GPU is available.
- [ ] FLAML as an alternative AutoML backend behind the same
      `FeatureMatrix → AutoMLReport` interface (see
      [integrations.md](integrations.md)).

## Sources

- AutoGluon `TabularPredictor.fit` API: <https://auto.gluon.ai/dev/api/autogluon.tabular.TabularPredictor.fit.html>
- AutoGluon `feature_importance` API: <https://auto.gluon.ai/dev/api/autogluon.tabular.TabularPredictor.feature_importance.html>
- AutoGluon Tabular — In Depth (HPO not recommended; `auto_stack`): <https://auto.gluon.ai/1.1.0/tutorials/tabular/tabular-indepth.html>
- AutoGluon Tabular — Essentials (`presets`, `dynamic_stacking`): <https://auto.gluon.ai/stable/tutorials/tabular/tabular-essentials.html>
- AutoGluon presets definitions: <https://github.com/autogluon/autogluon/blob/master/tabular/src/autogluon/tabular/configs/presets_configs.py>
- AutoGluon Foundational Models (TabPFNv2/Mitra for small data): <https://auto.gluon.ai/dev/tutorials/tabular/tabular-foundational-models.html>
- GitHub issue #4476 (permutation importance + multicollinearity): <https://github.com/autogluon/autogluon/issues/4476>
- Chen & Qi (2023), "How much should we trust R²" (R² vs LOO-R² inflation ~40%): <https://amstat.tandfonline.com/doi/full/10.1080/15140326.2023.2207326>
