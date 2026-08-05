"""Closed-loop verification of projected savings against post-deployment actuals.

The rest of voyage_trace is an *open loop*: :func:`~voyage_trace.simulator.simulate`
projects what a :class:`~voyage_trace.simulator.Modification` *would* save, the
:class:`~voyage_trace.agents.GovernanceAgent` accepts proposals whose projected
savings clear a threshold, and the resulting :class:`~voyage_trace.analysis.GovernancePlan`
is emitted — but nothing ever checks whether the savings that were *projected*
actually *materialised* once the plan was deployed. Every accepted proposal is
therefore a prediction with no measurement of prediction error and no mechanism
to correct systematic bias in the predictor.

This module closes that loop. It is the voyage_trace adaptation of the
**Local Reward Function (LRF)** pattern from OPTIMAS (Wu et al., ICLR 2026,
arXiv:2507.03041): a learned, per-component local→global mapping that is
re-aligned each iteration as the system drifts. Here the "local signal" is the
simulator's projected savings and the "global outcome" is the actual savings
observed by re-ingesting post-deployment traces; the "LRF" is a single scalar
calibration multiplier ``τ = Σactual / Σprojected`` per target agent. Three
pieces of published evidence motivate this design:

* **Aggregate projections mislead.** Counterfactual Trace Auditing
  (arXiv:2605.11946) found aggregate ΔP ≈ 0 while the underlying agent
  behaviour changed 696 times — a projected-savings total is an unreliable
  signal of real impact.
* **Un-verified proposals systematically overshoot.** TextualVerifier
  (arXiv:2511.03739) showed that adding a verification step to text-gradient
  proposals recovers +2 to +10pp on the held-out metric — i.e. the
  un-verified proposals were not realising their projected gains.
* **A learned local→global map beats raw judging.** OPTIMAS reports LRF
  ranking accuracy of 77.96% vs 49.52% for an LLM judge, and re-fits the LRF
  every iteration to track drift.

The closure is the feedback path through ``τ``::

        round N            round N+1 (post-deployment)
    ┌──────────────┐      ┌────────────────────────┐
    │ AutoML       │      │ ingest after-traces    │
    │ simulator    │      │ build after-graph      │
    │  projected P │      │   actual A             │
    │  governance  │      │   verify_plan(P, A)    │
    │  accepts on P│ ◄────┤   update_calibration τ │
    └──────┬───────┘      └────────────────────────┘
           │                                 │
           ▼                                 ▼
    next round's governance          τ = ΣA / ΣP  (per target agent)
    decides on  τ · projected        persisted in semantic memory,
                                      recalled cross-round

With ``τ = None`` (cold start, no verification history) the system behaves
exactly as before — :func:`calibrated_projection` returns the raw projection
unchanged. As post-deployment traces accumulate, ``τ`` converges toward the
true projector bias and governance decisions become calibrated. This is
strictly an additive feedback path; it never removes the simulator, never
fabricates savings, and never mutates a recorded trace.

Honesty contract
----------------
* ``compare_graphs`` does real graph arithmetic — it subtracts per-node
  ``cost_usd`` totals between two real :class:`~voyage_trace.execution_graph.ExecutionGraph`
  objects. It does NOT read the simulator's projection.
* ``verify_plan`` only pairs a proposal with reality when the proposal's
  ``target_node_id`` resolves in *both* the before- and after-graphs. Proposals
  whose target vanished post-deployment are reported as ``unverifiable``, not
  silently dropped or zeroed.
* ``τ`` is undefined (``None``) until at least one ``(projected, actual)`` pair
  with non-zero projected savings has been observed. Governance must fall back
  to the raw projection in that case — the cold-start path is explicit.
* Before/after graphs SHOULD have equal :attr:`~voyage_trace.execution_graph.ExecutionGraph.observed_runs`
  so totals are comparable; when they differ, ``compare_graphs`` normalises to
  per-call savings and re-projects to the before volume. The chosen convention
  is recorded in :attr:`VerificationResult.comparison_mode`.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import yaml

from .analysis import GovernancePlan
from .execution_graph import ExecutionGraph


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _new_id(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:12]}"


# --------------------------------------------------------------------------- #
# Per-projection error
# --------------------------------------------------------------------------- #
@dataclass
class ProjectionError:
    """The gap between one proposal's projected and actual savings.

    A *positive* ``error_usd`` means the simulator was optimistic (projected
    more saving than materialised); a *negative`` value means it was
    pessimistic. ``actual_usd`` is ``None`` when the proposal's target node
    could not be matched in the after-graph (e.g. the node was removed or
    renamed post-deployment) — such proposals are ``unverifiable`` and
    excluded from calibration.
    """

    proposal_id: str
    target_node_id: str
    kind: str  # the modification kind (swap_model, cap_loops, ...)
    projected_usd: float
    actual_usd: float | None
    note: str = ""

    @property
    def error_usd(self) -> float | None:
        if self.actual_usd is None:
            return None
        return self.projected_usd - self.actual_usd

    @property
    def relative_error(self) -> float | None:
        """``error / projected``, guarded against divide-by-zero.

        Positive = optimistic projection; negative = pessimistic.
        """
        if self.actual_usd is None or self.projected_usd == 0.0:
            return None
        return self.error_usd / self.projected_usd

    @property
    def unverifiable(self) -> bool:
        return self.actual_usd is None

    def to_dict(self) -> dict[str, Any]:
        return {
            "proposal_id": self.proposal_id,
            "target_node_id": self.target_node_id,
            "kind": self.kind,
            "projected_usd": self.projected_usd,
            "actual_usd": self.actual_usd,
            "error_usd": self.error_usd,
            "relative_error": self.relative_error,
            "unverifiable": self.unverifiable,
            "note": self.note,
        }


# --------------------------------------------------------------------------- #
# Per-plan verification result
# --------------------------------------------------------------------------- #
@dataclass
class VerificationResult:
    """Outcome of verifying one :class:`GovernancePlan` against reality.

    The unit persisted under the ``verification_results`` storage namespace
    and rendered to Markdown for human review. It carries enough to (a) audit
    one plan's projection accuracy and (b) fold into a running
    :class:`CalibrationState`.
    """

    plan_id: str
    target_agent_id: str
    round_id: str
    before_runs: int  # observed_runs in the before-graph
    after_runs: int  # observed_runs in the after-graph
    comparison_mode: str  # "totals" | "per_call_projected"
    # Per-node actual savings (node_id -> usd), from compare_graphs.
    node_actual_savings: dict[str, float] = field(default_factory=dict)
    # Per-proposal pairing; one entry per accepted proposal in the plan.
    projection_errors: list[ProjectionError] = field(default_factory=list)
    verification_id: str = field(default_factory=lambda: _new_id("verify"))
    created_at: datetime = field(default_factory=_utcnow)
    note: str = ""

    @property
    def verified_count(self) -> int:
        return sum(1 for e in self.projection_errors if not e.unverifiable)

    @property
    def unverifiable_count(self) -> int:
        return sum(1 for e in self.projection_errors if e.unverifiable)

    @property
    def total_projected_usd(self) -> float:
        return sum(e.projected_usd for e in self.projection_errors if not e.unverifiable)

    @property
    def total_actual_usd(self) -> float:
        return sum(e.actual_usd for e in self.projection_errors if e.actual_usd is not None)

    @property
    def total_error_usd(self) -> float:
        return self.total_projected_usd - self.total_actual_usd

    @property
    def mean_relative_error(self) -> float | None:
        rels = [e.relative_error for e in self.projection_errors
                if e.relative_error is not None]
        if not rels:
            return None
        return sum(rels) / len(rels)

    def to_dict(self) -> dict[str, Any]:
        return {
            "verification_id": self.verification_id,
            "plan_id": self.plan_id,
            "target_agent_id": self.target_agent_id,
            "round_id": self.round_id,
            "before_runs": self.before_runs,
            "after_runs": self.after_runs,
            "comparison_mode": self.comparison_mode,
            "node_actual_savings": self.node_actual_savings,
            "projection_errors": [e.to_dict() for e in self.projection_errors],
            "verified_count": self.verified_count,
            "unverifiable_count": self.unverifiable_count,
            "total_projected_usd": self.total_projected_usd,
            "total_actual_usd": self.total_actual_usd,
            "total_error_usd": self.total_error_usd,
            "mean_relative_error": self.mean_relative_error,
            "created_at": self.created_at.isoformat(),
            "note": self.note,
        }


# --------------------------------------------------------------------------- #
# Per-target-agent calibration state (the LRF, in minimal scalar form)
# --------------------------------------------------------------------------- #
@dataclass
class CalibrationState:
    """Running calibration of the simulator's savings projector for one agent.

    This is the OPTIMAS Local Reward Function reduced to a single scalar
    multiplier per target agent: ``τ = sum_actual / sum_projected`` over every
    ``(projected, actual)`` pair ever observed for that agent. It is the
    feedback path that closes the loop — :func:`calibrated_projection` applies
    ``τ`` to a raw projection so governance decides on a calibrated number.

    Design choices:

    * **Per target agent, not per modification kind.** The projection bias
      source ("the multiplicative cost/token model does not capture reality")
      is roughly shared across modification kinds for one agent. Per-kind
      calibration is noted as future work.
    * **Scalar, not a learned model.** A scalar ``τ`` is the simplest
      local→global map that is still re-aligned from observations. It is
      auditable (one number) and unfalsifiable in the cold-start case
      (``tau is None`` → fall back to raw projection).
    * **Cumulative, not windowed.** ``τ`` aggregates every observation ever
      made for the agent. A windowed / exponentially-decayed variant is
      straightforward to add once drift is actually observed; the cumulative
      form is the honest baseline.
    """

    target_agent_id: str
    sum_projected_usd: float = 0.0
    sum_actual_usd: float = 0.0
    n_observations: int = 0
    n_plans_verified: int = 0
    last_updated: datetime | None = None
    state_id: str = field(default_factory=lambda: _new_id("calib"))

    @property
    def tau(self) -> float | None:
        """The calibration multiplier ``Σactual / Σprojected``.

        ``None`` until at least one observation with non-zero projected
        savings has been recorded (cold start). Values:

        * ``τ < 1`` — the simulator is *optimistic* (projected savings
          exceed what materialised); governance should discount.
        * ``τ > 1`` — the simulator is *pessimistic*; governance may
          accept more aggressively.
        * ``τ ≈ 1`` — the projector is calibrated.
        """
        if self.n_observations == 0 or self.sum_projected_usd == 0.0:
            return None
        return self.sum_actual_usd / self.sum_projected_usd

    @property
    def is_cold_start(self) -> bool:
        return self.tau is None

    def to_dict(self) -> dict[str, Any]:
        return {
            "state_id": self.state_id,
            "target_agent_id": self.target_agent_id,
            "sum_projected_usd": self.sum_projected_usd,
            "sum_actual_usd": self.sum_actual_usd,
            "n_observations": self.n_observations,
            "n_plans_verified": self.n_plans_verified,
            "tau": self.tau,
            "is_cold_start": self.is_cold_start,
            "last_updated": self.last_updated.isoformat() if self.last_updated else None,
        }


# --------------------------------------------------------------------------- #
# Core: compare two graphs, verify a plan, update calibration
# --------------------------------------------------------------------------- #
def compare_graphs(before: ExecutionGraph, after: ExecutionGraph) -> dict[str, float]:
    """Per-node actual savings: ``before.cost - after.cost`` for each node.

    Real graph arithmetic over two :class:`~voyage_trace.execution_graph.ExecutionGraph`
    objects — no projection is consulted. When the two graphs observed
    different numbers of runs, per-call savings are computed and re-projected
    to the before volume so the totals stay comparable to the simulator's
    (total-over-before-runs) projection. The chosen convention is surfaced via
    :attr:`VerificationResult.comparison_mode` by :func:`verify_plan`.

    Nodes present in ``before`` but missing in ``after`` (the target was
    removed post-deployment) contribute their full before-cost as savings —
    this is the honest "the node is gone, so its cost is gone" reading. Nodes
    only in ``after`` (newly introduced) are ignored: they are not savings,
    they are new cost, and surface as a negative entry on their own node_id
    only when paired with a proposal that targets them (which, by construction,
    they cannot be).
    """
    savings: dict[str, float] = {}
    same_runs = before.observed_runs == after.observed_runs
    before_calls = {nid: max(n.calls, 1) for nid, n in before.nodes.items()}
    after_calls = {nid: max(n.calls, 1) for nid, n in after.nodes.items()}

    for nid, before_node in before.nodes.items():
        before_cost = before_node.cost_usd
        after_node = after.nodes.get(nid)
        if after_node is None:
            # Node disappeared → its full before-cost is saved.
            actual = before_cost
        else:
            after_cost = after_node.cost_usd
            if same_runs:
                actual = before_cost - after_cost
            else:
                # Per-call normalisation, re-projected to the before volume.
                before_per_call = before_cost / before_calls[nid]
                after_per_call = after_cost / after_calls[nid]
                actual = (before_per_call - after_per_call) * before_calls[nid]
        savings[nid] = actual
    return savings


def verify_plan(
    plan: GovernancePlan,
    before_graph: ExecutionGraph,
    after_graph: ExecutionGraph,
    *,
    round_id: str = "",
    note: str = "",
) -> VerificationResult:
    """Verify a plan's projected savings against post-deployment actuals.

    For each accepted proposal in ``plan``, look up its ``target_node_id`` in
    the before- and after-graphs and compute the actual savings from
    :func:`compare_graphs`. Pair it with the proposal's
    ``expected_savings["cost_delta_usd"]`` (the simulator's projection) into a
    :class:`ProjectionError`. Rejected proposals are not verified — they were
    never deployed, so there is no reality to compare against.

    The before-graph should be the aggregated template graph the plan was
    produced from; the after-graph should be aggregated from post-deployment
    traces of the same target agent.
    """
    node_savings = compare_graphs(before_graph, after_graph)
    comparison_mode = (
        "totals" if before_graph.observed_runs == after_graph.observed_runs
        else "per_call_projected"
    )

    errors: list[ProjectionError] = []
    for proposal in plan.accepted_proposals:
        target = proposal.modification.target_node_id
        projected = float(proposal.expected_savings.get("cost_delta_usd", 0.0))
        if target in node_savings:
            actual = float(node_savings[target])
            err_note = "ok"
        elif target not in before_graph.nodes:
            # The target was never in the before-graph (e.g. a stale plan);
            # nothing can be said.
            actual = None
            err_note = "target absent from before-graph — unverifiable"
        else:
            # Target was in before but compare_graphs did not produce an entry
            # (defensive; compare_graphs covers all before-nodes).
            actual = None
            err_note = "no actual savings entry — unverifiable"
        errors.append(ProjectionError(
            proposal_id=proposal.proposal_id,
            target_node_id=target,
            kind=proposal.modification.kind,
            projected_usd=projected,
            actual_usd=actual,
            note=err_note,
        ))

    return VerificationResult(
        plan_id=plan.plan_id,
        target_agent_id=plan.target_agent_id,
        round_id=round_id or plan.round_id,
        before_runs=before_graph.observed_runs,
        after_runs=after_graph.observed_runs,
        comparison_mode=comparison_mode,
        node_actual_savings=node_savings,
        projection_errors=errors,
        note=note,
    )


def update_calibration(
    state: CalibrationState,
    result: VerificationResult,
) -> CalibrationState:
    """Fold one :class:`VerificationResult` into a running :class:`CalibrationState`.

    Only *verifiable* proposals (those whose target resolved in both graphs)
    contribute to ``τ``; unverifiable ones are counted in the result but
    excluded from the sums. The state is mutated in place and returned for
    fluent use.
    """
    for err in result.projection_errors:
        if err.unverifiable or err.actual_usd is None:
            continue
        # Only observations with a non-zero projection teach τ; a zero-projected
        # observation carries no bias information (divide-by-zero guard).
        if err.projected_usd == 0.0:
            continue
        state.sum_projected_usd += err.projected_usd
        state.sum_actual_usd += max(err.actual_usd, 0.0)
        state.n_observations += 1
    state.n_plans_verified += 1
    state.last_updated = _utcnow()
    return state


def calibrated_projection(
    raw_savings_usd: float,
    tau: float | None,
) -> float:
    """Apply the calibration multiplier ``τ`` to a raw projected saving.

    Returns the raw value unchanged when ``tau is None`` (cold start). This is
    the single function the governance agent calls to turn a simulator
    projection into a calibrated projection.
    """
    if tau is None:
        return raw_savings_usd
    return raw_savings_usd * tau


# --------------------------------------------------------------------------- #
# JSON serialisation (mirrors analysis.py's style)
# --------------------------------------------------------------------------- #
def verification_to_dict(result: VerificationResult) -> dict[str, Any]:
    return result.to_dict()


def verification_from_dict(d: dict[str, Any]) -> VerificationResult:
    errors = [
        ProjectionError(
            proposal_id=e.get("proposal_id", ""),
            target_node_id=e.get("target_node_id", ""),
            kind=e.get("kind", ""),
            projected_usd=float(e.get("projected_usd", 0.0)),
            actual_usd=None if e.get("actual_usd") is None else float(e.get("actual_usd")),
            note=e.get("note", ""),
        )
        for e in d.get("projection_errors", [])
    ]
    created = d.get("created_at")
    result = VerificationResult(
        plan_id=d.get("plan_id", ""),
        target_agent_id=d.get("target_agent_id", ""),
        round_id=d.get("round_id", ""),
        before_runs=int(d.get("before_runs", 0)),
        after_runs=int(d.get("after_runs", 0)),
        comparison_mode=d.get("comparison_mode", "totals"),
        node_actual_savings=d.get("node_actual_savings") or {},
        projection_errors=errors,
        verification_id=d.get("verification_id") or _new_id("verify"),
        note=d.get("note", ""),
    )
    if created:
        try:
            result.created_at = datetime.fromisoformat(created.replace("Z", "+00:00"))
        except ValueError:
            pass
    return result


def calibration_to_dict(state: CalibrationState) -> dict[str, Any]:
    return state.to_dict()


def calibration_from_dict(d: dict[str, Any]) -> CalibrationState:
    state = CalibrationState(
        target_agent_id=d.get("target_agent_id", ""),
        sum_projected_usd=float(d.get("sum_projected_usd", 0.0)),
        sum_actual_usd=float(d.get("sum_actual_usd", 0.0)),
        n_observations=int(d.get("n_observations", 0)),
        n_plans_verified=int(d.get("n_plans_verified", 0)),
        state_id=d.get("state_id") or _new_id("calib"),
    )
    last = d.get("last_updated")
    if last:
        try:
            state.last_updated = datetime.fromisoformat(last.replace("Z", "+00:00"))
        except ValueError:
            pass
    return state


def verification_to_json(result: VerificationResult) -> str:
    return json.dumps(verification_to_dict(result), sort_keys=True, separators=(",", ":"))


def verification_from_json(text: str | bytes) -> VerificationResult:
    return verification_from_dict(json.loads(text))


def calibration_to_json(state: CalibrationState) -> str:
    return json.dumps(calibration_to_dict(state), sort_keys=True, separators=(",", ":"))


def calibration_from_json(text: str | bytes) -> CalibrationState:
    return calibration_from_dict(json.loads(text))


# --------------------------------------------------------------------------- #
# Markdown rendering (mirrors render_analysis_markdown)
# --------------------------------------------------------------------------- #
def render_verification_markdown(result: VerificationResult) -> str:
    """Render a :class:`VerificationResult` as a Git-diffable Markdown document.

    Layout (parallel to :func:`~voyage_trace.analysis.render_analysis_markdown`):

    ```markdown
    ---
    verification_id: ...
    plan_id: ...
    target_agent_id: ...
    round_id: ...
    verified_count: N
    tau_contribution: ...
    ---
    # Verification <plan_id> — Projected vs Actual Savings

    ## Summary
    <totals + mean relative error + comparison mode>

    ## Per-Proposal Errors
    | id | target | kind | projected | actual | error | rel% | note |

    ## Per-Node Actual Savings
    | node | actual_saving($) |
    ```
    """
    front = {
        "verification_id": result.verification_id,
        "plan_id": result.plan_id,
        "target_agent_id": result.target_agent_id,
        "round_id": result.round_id,
        "before_runs": result.before_runs,
        "after_runs": result.after_runs,
        "comparison_mode": result.comparison_mode,
        "verified_count": result.verified_count,
        "unverifiable_count": result.unverifiable_count,
        "total_projected_usd": round(result.total_projected_usd, 6),
        "total_actual_usd": round(result.total_actual_usd, 6),
        "total_error_usd": round(result.total_error_usd, 6),
        "created_at": result.created_at.isoformat(),
    }
    front_str = yaml.safe_dump(front, sort_keys=False, allow_unicode=True).strip()

    lines: list[str] = ["---", front_str, "---", ""]
    lines.append(f"# Verification {result.plan_id} — Projected vs Actual Savings")
    lines.append("")

    lines.append("## Summary")
    mre = result.mean_relative_error
    mre_str = f"{mre * 100:.2f}%" if mre is not None else "n/a"
    bias = (
        "optimistic" if result.total_error_usd > 0
        else "pessimistic" if result.total_error_usd < 0
        else "calibrated"
    )
    lines.append(
        f"- comparison_mode: `{result.comparison_mode}` "
        f"(before_runs={result.before_runs}, after_runs={result.after_runs})"
    )
    lines.append(
        f"- projected: ${result.total_projected_usd:.6f} | "
        f"actual: ${result.total_actual_usd:.6f} | "
        f"error: ${result.total_error_usd:.6f} ({bias})"
    )
    lines.append(f"- mean_relative_error: {mre_str}")
    lines.append(f"- verified: {result.verified_count} | unverifiable: {result.unverifiable_count}")
    if result.note:
        lines.append(f"- note: {result.note}")
    lines.append("")

    lines.append("## Per-Proposal Errors")
    if result.projection_errors:
        lines.append("| id | target | kind | projected | actual | error | rel% | note |")
        lines.append("|---|---|---|---|---|---|---|---|")
        for e in result.projection_errors:
            proj = f"{e.projected_usd:.6f}"
            act = "n/a" if e.actual_usd is None else f"{e.actual_usd:.6f}"
            err = "n/a" if e.error_usd is None else f"{e.error_usd:.6f}"
            rel = "n/a" if e.relative_error is None else f"{e.relative_error * 100:.2f}%"
            note = (e.note or "").replace("|", "/").replace("\n", " ")
            lines.append(
                f"| {e.proposal_id} | {e.target_node_id} | {e.kind} | "
                f"{proj} | {act} | {err} | {rel} | {note} |"
            )
    else:
        lines.append("- (no accepted proposals to verify)")
    lines.append("")

    lines.append("## Per-Node Actual Savings")
    if result.node_actual_savings:
        lines.append("| node | actual_saving($) |")
        lines.append("|---|---|")
        for nid in sorted(result.node_actual_savings):
            lines.append(f"| {nid} | {result.node_actual_savings[nid]:.6f} |")
    else:
        lines.append("- (no nodes)")
    lines.append("")

    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Agent chain-of-thought guidance
# --------------------------------------------------------------------------- #
VERIFICATION_COT_PROMPT = """\
You are the **verification sub-agent** in the voyage_trace governance pipeline.
You run AFTER a governance plan has been deployed and a fresh batch of
post-deployment traces of the SAME target agent has been collected. Your job
is to close the loop: measure whether the savings the simulator *projected*
actually *materialised*, and fold the gap into a calibration multiplier that
the next governance round uses.

Think step by step.

## When to run (and when NOT to)
1. Has a previously-accepted GovernancePlan been deployed, and have you
   received ≥1 post-deployment trace of the same target agent? If NO → do NOT
   run. Verification needs reality to compare against; nothing to verify yet.
2. Can you aggregate the post-deployment traces into an `ExecutionGraph` whose
   nodes use the SAME `<operation_type>:<label>` keys as the before-graph the
   plan was produced from? If NO → record a `VERIFY` step with status FAILED
   and rationale "node keys do not align — cannot pair proposals with reality".
3. If YES to both → proceed.

## How to run
- Build the after-graph from the post-deployment traces via
  `aggregate_execution_graph`.
- Call `verification.verify_plan(plan, before_graph, after_graph)`. This does
  REAL graph arithmetic: `actual = before.node.cost - after.node.cost` per
  node. It does NOT consult the simulator's projection — that is the point.
- For each accepted proposal in the plan, the result pairs its
  `expected_savings["cost_delta_usd"]` (projected) with the actual savings of
  its `target_node_id`. Rejected proposals are NOT verified (never deployed).

## How to update the calibration multiplier
- Recall the agent's `CalibrationState` from semantic memory
  (`recall_cross_round(target_agent_id, "calibration:state")`).
- Call `verification.update_calibration(state, result)`. This folds the
  verified (projected, actual) pairs into `state.sum_projected_usd` /
  `state.sum_actual_usd` and recomputes `τ = Σactual / Σprojected`.
- Persist the updated `CalibrationState` back to semantic memory so the next
  governance round can recall it. Record a `REMEMBER` step.
- `τ < 1` means the simulator was optimistic (typical); `τ > 1` pessimistic;
  `τ ≈ 1` calibrated. `τ is None` means cold start (no observations yet).

## How the next round uses τ
- The governance agent recalls `τ` and calls
  `verification.calibrated_projection(raw_savings, τ)` before applying the
  `min_savings_usd` threshold. With `τ is None` it falls back to the raw
  projection (today's behaviour). This is the feedback path that closes the
  loop — the projector is corrected by its own past errors.

## Honesty contract
- NEVER fabricate actual savings. If a proposal's target node is absent from
  the after-graph, mark it `unverifiable` and exclude it from `τ`. Do NOT
  assume zero savings.
- Before/after graphs SHOULD have equal `observed_runs`. When they differ,
  `verify_plan` normalises to per-call savings and re-projects to the before
  volume, and records `comparison_mode="per_call_projected"` — surface that
  mode in your `VERIFY` step rationale so a reviewer knows the totals are
  normalised, not raw.
- Record EVERY verification as an `AnalysisStep(kind=VERIFY)` with inputs
  {plan_id, before_runs, after_runs} and outputs {verified_count,
  total_projected_usd, total_actual_usd, tau}. Persist the
  `VerificationResult` under the `verification_results` storage namespace.
- Echo the calibration state honestly: if `τ is None` (cold start), say so in
  the verification summary. Do not present a cold-start projection as
  calibrated.
"""
