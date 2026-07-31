"""Multi-agent architecture for the voyage_trace governance pipeline.

This module splits the analysis / optimisation process into four
specialised sub-agents coordinated by one orchestrator, plus a fifth
verification sub-agent that closes the projection→actual loop AFTER a plan
is deployed:

    ┌──────────────┐   payloads   ┌──────────────┐  traces  ┌──────────────┐
    │ IngestAgent  │ ───────────► │ ModelingAgent│ ───────► │SimulationAgent│
    │              │   traces     │ (+ AutoML)   │  graph   │              │
    └──────────────┘              └──────────────┘  props   └──────┬───────┘
                                                                        │ validated
                                                                        ▼
                                                            ┌──────────────────┐
                                                            │ GovernanceAgent  │
                                                            │  (decide+memory) │─── plan ──┐
                                                            └──────────────────┘           │
                                                                                          ▼
                                                              ┌────────────────────────────┐
   post-deployment traces ──────────────────────────────────► │ VerificationAgent          │
                                                              │  verify projected vs actual│
                                                              │  update calibration τ      │
                                                              └─────────────┬──────────────┘
                                                                            │ τ (cross-round)
                                                                            ▼
                                                              next GovernanceAgent.run() decides
                                                              on  calibrated_projection(P, τ)

Every sub-agent operates on a shared :class:`~voyage_trace.analysis.AnalysisRecord`
and appends :class:`~voyage_trace.analysis.AnalysisStep` objects describing what
it did and why. So the multi-agent trajectory *is* the AnalysisRecord — the
internal data format defined in :mod:`voyage_trace.analysis`.

Design notes
------------
* **Pure-Python, no live LLM.** Each agent is a plain class with a ``run``
  method. The role / CoT prompts (``*_ROLE``) are the same prompts a
  deepagents sub-agent would be seeded with — wiring them into real LLM
  sub-agents later is a mechanical step (pass ``role.cot_prompt`` as the
  sub-agent's system prompt and expose ``run``'s body as tools).
* **Sync core, async seam.** Ingest / Modelling / Simulation are pure CPU
  work and stay sync. Governance and Verification are ``async`` because they
  touch the async :class:`~voyage_trace.memory.manager.PartitionedMemory`.
  The :class:`Orchestrator` is ``async`` to match.
* **AutoML proposes, simulator disposes.** The modelling agent surfaces
  candidate :class:`~voyage_trace.analysis.OptimizationProposal` objects
  from AutoML; the simulation agent fills ``expected_savings``; the
  governance agent accepts only proposals the simulator validated as
  beneficial. No proposal reaches the plan unvalidated.
* **The simulator projects, reality disposes.** Verification is the closed
  loop: a deployed plan's projected savings are compared to the savings that
  materialised in post-deployment traces, and the gap is folded into a
  calibration multiplier ``τ`` that the next governance round applies to its
  raw projections (see :mod:`voyage_trace.verification`). With ``τ = None``
  (cold start) governance behaves exactly as before.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any

from .adapters import adapt
from .analysis import (
    AnalysisRecord,
    AnalysisStep,
    AnalysisStepKind,
    GovernancePlan,
    OptimizationProposal,
    ProposalDecision,
    StepStatus,
    render_analysis_markdown,
)
from .automl import (
    AUTOML_COT_PROMPT,
    AutoMLReport,
    inject_automl_into_graph_md,
    run_automl,
)
from .execution_graph import (
    ExecutionGraph,
    aggregate_execution_graph,
    build_execution_graph,
    render_markdown,
)
from .memory import MemoryScope
from .simulator import Modification, project_savings, simulate_graph
from .types import CanonicalTrace
from .verification import (
    VERIFICATION_COT_PROMPT,
    CalibrationState,
    VerificationResult,
    calibrated_projection,
    calibration_from_dict,
    update_calibration,
    verify_plan,
)

# Sentinel for "calibration_multiplier not provided by the caller". Distinct
# from ``None`` (which the caller may pass explicitly to force cold-start even
# when memory is wired).
_UNSET: Any = object()


# --------------------------------------------------------------------------- #
# Agent roles + chain-of-thought prompts
# --------------------------------------------------------------------------- #
@dataclass
class AgentRole:
    """Declarative description of one sub-agent's responsibility.

    ``cot_prompt`` is the system prompt a real LLM sub-agent would receive;
    it is also the documentation for what the sync ``run`` method does.
    """

    name: str
    description: str
    cot_prompt: str
    inputs: list[str] = field(default_factory=list)
    outputs: list[str] = field(default_factory=list)


INGEST_ROLE = AgentRole(
    name="ingest",
    description="Adapts raw payloads into normalised CanonicalTraces.",
    inputs=["raw_payloads"],
    outputs=["traces"],
    cot_prompt="""\
You are the **ingest sub-agent**. For each raw payload, pick a source
protocol (or let `adapters.adapt` infer it) and produce a normalised
`CanonicalTrace`. Record one `AnalysisStep(kind=INGEST)` per payload with
the inferred protocol and span count. If a payload fails to adapt, record
the step as `FAILED` with the error note and continue with the rest — never
abort the whole round on one bad payload.
""",
)

MODELING_ROLE = AgentRole(
    name="modeling",
    description="Builds the execution graph, runs AutoML, emits proposals.",
    inputs=["traces"],
    outputs=["execution_graph", "automl_report", "proposals"],
    cot_prompt=AUTOML_COT_PROMPT,
)

SIMULATION_ROLE = AgentRole(
    name="simulation",
    description="Validates each proposal with the simulator; fills expected_savings.",
    inputs=["traces", "proposals"],
    outputs=["validated_proposals"],
    cot_prompt="""\
You are the **simulation sub-agent**. For each proposal, run
`simulator.simulate(trace, [proposal.modification])` against a baseline
`replay(trace)` and fill `proposal.expected_savings` via
`simulator.project_savings`. Mark `validated=True` only if the simulation
produced no divergences AND projected savings are non-negative. Record one
`AnalysisStep(kind=VALIDATE)` per proposal. You NEVER decide acceptance —
that is the governance agent's job. You only report what the simulator said.
""",
)

GOVERNANCE_ROLE = AgentRole(
    name="governance",
    description="Decides accept/reject per proposal, writes the plan + memory.",
    inputs=["record", "proposals", "memory"],
    outputs=["governance_plan"],
    cot_prompt="""\
You are the **governance sub-agent**. Accept a proposal iff it is validated
AND its projected `cost_delta_usd >= min_savings_usd`. Reject otherwise, with
a one-line rationale. Build a `GovernancePlan` from the accepted subset,
write a 2-3 sentence summary, and record an `AnalysisStep(kind=DECIDE)`.
If a `PartitionedMemory` is available: recall similar past failures
(`recall_cross_round`) before deciding, remember the outcome
(`episodic().remember`) after, and persist any reusable fix as a
procedural template. Record `RECALL` / `REMEMBER` steps. Finish the
AnalysisRecord. You are the final authority — nothing reaches the operator
that you did not sign off on.

If a calibration multiplier `τ` is available (recalled from a prior
verification round), apply it via `verification.calibrated_projection` before
comparing to `min_savings_usd` — decide on the *calibrated* projection, not
the raw one. Record `τ` on each proposal's `calibration_applied` and on the
plan. With `τ is None` (cold start) behave exactly as before (raw projection).
""",
)

VERIFICATION_ROLE = AgentRole(
    name="verification",
    description="Verifies a deployed plan's projected savings vs post-deployment actuals; updates τ.",
    inputs=["plan", "before_graph", "after_traces", "memory"],
    outputs=["verification_result", "calibration_state"],
    cot_prompt=VERIFICATION_COT_PROMPT,
)


# --------------------------------------------------------------------------- #
# Shared context passed between agents (keeps run() signatures small)
# --------------------------------------------------------------------------- #
@dataclass
class ModelingOutput:
    """What the modelling agent hands to the simulation agent."""

    graph: ExecutionGraph
    graph_md: str  # execution-graph Markdown (AutoML-enriched when report is set)
    report: AutoMLReport | None
    automl_target: str | None = None


# --------------------------------------------------------------------------- #
# IngestAgent
# --------------------------------------------------------------------------- #
class IngestAgent:
    """Adapts raw payloads into normalised :class:`CanonicalTrace` objects."""

    role: AgentRole = INGEST_ROLE

    def run(
        self,
        payloads: list[Any],
        record: AnalysisRecord,
        *,
        source_protocol: str | None = None,
    ) -> list[CanonicalTrace]:
        traces: list[CanonicalTrace] = []
        for i, payload in enumerate(payloads):
            step = AnalysisStep(
                kind=AnalysisStepKind.INGEST,
                agent_role=self.role.name,
                rationale=f"adapt payload #{i} via adapters.adapt",
                inputs={"payload_index": i, "source_protocol": source_protocol or "auto"},
            )
            record.add_step(step)
            try:
                trace = adapt(payload, source_protocol=source_protocol)
                # Coalesce: every trace in one round targets the same agent.
                if not trace.agent_id:
                    trace.agent_id = record.target_agent_id
                step.outputs = {
                    "trace_id": trace.trace_id,
                    "span_count": trace.span_count,
                    "source_protocol": trace.source_protocol.value,
                }
                step.finish()
                traces.append(trace)
            except Exception as exc:  # noqa: BLE001 — one bad payload must not abort the round
                step.finish(StepStatus.FAILED, note=f"{type(exc).__name__}: {exc}")
        # Summary step so the timeline is grep-able. Pass the status to
        # ``finish()`` explicitly — ``finish()`` defaults to SUCCESS and would
        # otherwise clobber the FAILED status we want for empty ingest.
        summary_status = StepStatus.SUCCESS if traces else StepStatus.FAILED
        record.add_step(AnalysisStep(
            kind=AnalysisStepKind.INGEST,
            agent_role=self.role.name,
            rationale="ingest summary",
            inputs={"payload_count": len(payloads)},
            outputs={"trace_count": len(traces)},
        )).finish(summary_status)
        return traces


# --------------------------------------------------------------------------- #
# ModelingAgent
# --------------------------------------------------------------------------- #
class ModelingAgent:
    """Builds the execution graph, runs AutoML, and surfaces proposals.

    Implements the CoT in :data:`AUTOML_COT_PROMPT`: with <3 traces it
    builds the descriptive graph only; with ≥3 it enriches it with the
    AutoML view and turns each AutoML suggestion into an
    :class:`OptimizationProposal`.
    """

    role: AgentRole = MODELING_ROLE
    min_traces_for_automl: int = 3

    def run(
        self,
        traces: list[CanonicalTrace],
        record: AnalysisRecord,
        *,
        automl_target: str = "cost_usd",
    ) -> ModelingOutput:
        # --- descriptive model (always) ----------------------------------- #
        graph = aggregate_execution_graph(traces) if len(traces) > 1 else build_execution_graph(traces[0])
        graph_md = render_markdown(graph)
        record.add_step(AnalysisStep(
            kind=AnalysisStepKind.MODEL,
            agent_role=self.role.name,
            rationale="build execution graph + render Markdown",
            inputs={"trace_count": len(traces)},
            outputs={"nodes": len(graph.nodes), "edges": len(graph.edges),
                     "total_cost_usd": round(graph.total_cost_usd, 6)},
        )).finish()

        # --- explanatory model (AutoML, conditional) ---------------------- #
        report: AutoMLReport | None = None
        if len(traces) < self.min_traces_for_automl:
            record.add_step(AnalysisStep(
                kind=AnalysisStepKind.MODEL,
                agent_role=self.role.name,
                rationale=(
                    f"insufficient samples for AutoML ({len(traces)} < "
                    f"{self.min_traces_for_automl}); descriptive graph only"
                ),
                inputs={"trace_count": len(traces)},
                outputs={"automl": False},
            )).finish()
        else:
            report = run_automl(traces, target=automl_target)
            graph_md = inject_automl_into_graph_md(graph_md, report)
            step = record.add_step(AnalysisStep(
                kind=AnalysisStepKind.MODEL,
                agent_role=self.role.name,
                rationale=f"run AutoML target={automl_target}; enrich graph MD",
                inputs={"trace_count": len(traces), "target": automl_target},
                outputs={
                    "best_model": report.best_model.feature,
                    "r_squared": round(report.best_model.r_squared, 4),
                    "top_feature": report.top_feature,
                    "suggestion_count": len(report.suggested_modifications),
                },
            ))
            step.finish(StepStatus.FAILED if report.notes and any("No feature" in n for n in report.notes) else StepStatus.SUCCESS)

            # Turn each AutoML suggestion into a candidate proposal. AutoML
            # proposes; the simulation agent disposes.
            for mod, rationale in report.suggested_modifications:
                proposal = OptimizationProposal(
                    modification=mod,
                    rationale=rationale,
                )
                record.add_proposal(proposal)
                record.add_step(AnalysisStep(
                    kind=AnalysisStepKind.PROPOSE,
                    agent_role=self.role.name,
                    rationale=rationale,
                    inputs={"target_node_id": mod.target_node_id, "kind": mod.kind},
                    outputs={"proposal_id": proposal.proposal_id},
                )).finish()

        return ModelingOutput(graph=graph, graph_md=graph_md, report=report, automl_target=automl_target)


# --------------------------------------------------------------------------- #
# SimulationAgent
# --------------------------------------------------------------------------- #
class SimulationAgent:
    """Validates each proposal against the simulator.

    Proposals come from AutoML and target **aggregated graph node_ids**
    (``<operation_type>:<label>``), so validation must run on the
    aggregated :class:`~voyage_trace.execution_graph.ExecutionGraph` via
    :func:`~voyage_trace.simulator.simulate_graph` — *not* on a single
    trace (whose span_ids would not match the aggregated keys). The
    baseline is ``simulate_graph(graph, [])``; each proposal is
    ``simulate_graph(graph, [modification])``; ``expected_savings`` comes
    from :func:`~voyage_trace.simulator.project_savings`.
    """

    role: AgentRole = SIMULATION_ROLE

    def run(
        self,
        graph: ExecutionGraph,
        record: AnalysisRecord,
        proposals: list[OptimizationProposal],
    ) -> list[OptimizationProposal]:
        baseline = simulate_graph(graph, [])
        record.add_step(AnalysisStep(
            kind=AnalysisStepKind.SIMULATE,
            agent_role=self.role.name,
            rationale="baseline simulate_graph of aggregated template",
            inputs={"nodes": len(graph.nodes), "observed_runs": graph.observed_runs},
            outputs={"total_cost_usd": round(baseline.total_cost_usd, 6),
                     "ok": baseline.ok},
        )).finish()

        for proposal in proposals:
            mod = proposal.modification
            modified = simulate_graph(graph, [mod])
            savings = project_savings(baseline, modified)
            proposal.expected_savings = savings
            # Validated iff the simulator did not diverge AND the change is
            # not strictly worse on cost. (A 0-cost change that improves
            # latency is still "validated" — non-negative, not strictly
            # positive, so guardrails like cap_loops can pass.)
            cost_delta = savings.get("cost_delta_usd", 0.0)
            proposal.validated = (not modified.divergences) and cost_delta >= 0.0
            notes = []
            if modified.divergences:
                notes.append(f"{len(modified.divergences)} divergence(s)")
            if cost_delta < 0:
                notes.append(f"cost increased by ${-cost_delta:.6f}")
            proposal.validation_notes = "; ".join(notes) if notes else "ok"
            record.add_step(AnalysisStep(
                kind=AnalysisStepKind.VALIDATE,
                agent_role=self.role.name,
                rationale=f"simulate_graph {mod.kind} on {mod.target_node_id}",
                inputs={"proposal_id": proposal.proposal_id, "kind": mod.kind},
                outputs={"validated": proposal.validated,
                         "cost_delta_usd": round(cost_delta, 6),
                         "divergences": len(modified.divergences)},
            )).finish()
        return proposals


# --------------------------------------------------------------------------- #
# GovernanceAgent (async — touches PartitionedMemory)
# --------------------------------------------------------------------------- #
class GovernanceAgent:
    """Decides accept/reject, writes the :class:`GovernancePlan`, persists memory."""

    role: AgentRole = GOVERNANCE_ROLE

    async def run(
        self,
        record: AnalysisRecord,
        *,
        memory: Any | None = None,
        min_savings_usd: float = 0.0,
        calibration_multiplier: float | None = None,
    ) -> GovernancePlan:
        """Decide accept/reject and write the :class:`GovernancePlan`.

        When ``calibration_multiplier`` (``τ``) is provided, the
        accept/reject threshold is applied to the *calibrated* projection
        ``τ · projected_savings`` (via
        :func:`~voyage_trace.verification.calibrated_projection`) rather than
        the raw simulator projection. ``None`` means cold start: the raw
        projection is used unchanged (today's behaviour). The raw
        ``expected_savings`` on each proposal is never overwritten — ``τ`` is
        recorded on ``proposal.calibration_applied`` and ``plan.calibration_applied``
        so the raw vs calibrated decision is always auditable.
        """
        # Optional cross-round recall before deciding. This is the
        # "recall for reuse" path: past outcomes for the same failure shape
        # inform whether to trust a proposal.
        if memory is not None:
            await self._recall(memory, record)

        accepted: list[OptimizationProposal] = []
        rejected: list[OptimizationProposal] = []
        for proposal in record.proposals:
            raw_cost_delta = proposal.expected_savings.get("cost_delta_usd", 0.0)
            calibrated_delta = calibrated_projection(raw_cost_delta, calibration_multiplier)
            # Record τ on the proposal so the raw-vs-calibrated decision is
            # auditable per proposal, not just at the plan level.
            proposal.calibration_applied = calibration_multiplier
            if proposal.validated and calibrated_delta >= min_savings_usd:
                if calibration_multiplier is None:
                    rationale = (
                        f"validated; projected saving ${raw_cost_delta:.6f} >= ${min_savings_usd}"
                    )
                else:
                    rationale = (
                        f"validated; calibrated saving ${calibrated_delta:.6f} "
                        f"(τ={calibration_multiplier:.4f} × ${raw_cost_delta:.6f}) "
                        f">= ${min_savings_usd}"
                    )
                proposal.accept(rationale=rationale)
                accepted.append(proposal)
            else:
                if not proposal.validated:
                    reason = "not validated"
                elif calibration_multiplier is None:
                    reason = f"saving ${raw_cost_delta:.6f} < ${min_savings_usd}"
                else:
                    reason = (
                        f"calibrated saving ${calibrated_delta:.6f} "
                        f"(τ={calibration_multiplier:.4f}) < ${min_savings_usd}"
                    )
                proposal.reject(rationale=reason)
                rejected.append(proposal)
            record.add_step(AnalysisStep(
                kind=AnalysisStepKind.DECIDE,
                agent_role=self.role.name,
                rationale=proposal.decision_rationale,
                inputs={"proposal_id": proposal.proposal_id},
                outputs={
                    "decision": proposal.decision.value if proposal.decision else "—",
                    "calibrated_delta_usd": round(calibrated_delta, 6),
                },
            )).finish()

        # Compose the human-readable summary.
        summary_parts = [
            f"Round {record.round_id} over {record.target_agent_id}: "
            f"{len(accepted)} accepted / {len(rejected)} rejected proposal(s).",
        ]
        # Surface AutoML's headline signal if present.
        model_steps = [s for s in record.steps if s.kind == AnalysisStepKind.MODEL and s.outputs.get("best_model")]
        if model_steps:
            last = model_steps[-1]
            summary_parts.append(
                f"AutoML top feature: {last.outputs.get('top_feature')} "
                f"(R^2={last.outputs.get('r_squared')})."
            )
        # Echo any low-sample warning from AutoML (honesty contract).
        for s in record.steps:
            if s.kind == AnalysisStepKind.MODEL and s.status == StepStatus.FAILED:
                summary_parts.append("WARNING: AutoML found no explanatory signal above the mean baseline.")
                break
        # Surface the calibration state (closed-loop honesty contract).
        if calibration_multiplier is None:
            summary_parts.append("Calibration: cold-start (raw simulator projection used).")
        else:
            total_calibrated = sum(
                calibrated_projection(p.expected_savings.get("cost_delta_usd", 0.0), calibration_multiplier)
                for p in accepted
            )
            summary_parts.append(
                f"Calibration: τ={calibration_multiplier:.4f} applied "
                f"(total calibrated savings ${total_calibrated:.6f})."
            )

        plan = GovernancePlan(
            target_agent_id=record.target_agent_id,
            round_id=record.round_id,
            summary=" ".join(summary_parts),
            accepted_proposals=accepted,
            rejected_proposals=rejected,
            metrics={
                "total_proposals": len(record.proposals),
                "accepted": len(accepted),
                "rejected": len(rejected),
                "total_projected_savings_usd": round(
                    sum(p.expected_savings.get("cost_delta_usd", 0.0) for p in accepted), 6
                ),
            },
            analysis_record_id=record.record_id,
            calibration_applied=calibration_multiplier,
        )
        record.plan = plan
        record.finish()

        # Optional memory persistence after deciding.
        if memory is not None:
            await self._remember(memory, record, plan)

        return plan

    async def _recall(self, memory: Any, record: AnalysisRecord) -> None:
        """Recall past outcomes for the round's failure signatures."""
        # The failure signature is derived from any failed/error span seen
        # in the analysis. We use the target agent id as the scope.
        step = AnalysisStep(
            kind=AnalysisStepKind.RECALL,
            agent_role=self.role.name,
            rationale="recall_cross_round for prior governance outcomes",
            inputs={"target_agent_id": record.target_agent_id},
        )
        record.add_step(step)
        try:
            # Recall by a generic signature; concrete agents can specialise.
            hits = await memory.recall_cross_round(record.target_agent_id, "governance:outcome", limit=5)
            step.outputs = {"hits": len(hits) if hasattr(hits, "__len__") else 0}
            step.finish()
        except Exception as exc:  # noqa: BLE001 — memory is best-effort
            step.finish(StepStatus.FAILED, note=f"{type(exc).__name__}: {exc}")

    async def _remember(self, memory: Any, record: AnalysisRecord, plan: GovernancePlan) -> None:
        """Persist the round outcome into episodic memory for future recall."""
        step = AnalysisStep(
            kind=AnalysisStepKind.REMEMBER,
            agent_role=self.role.name,
            rationale="persist round outcome to episodic memory",
            inputs={"plan_id": plan.plan_id},
        )
        record.add_step(step)
        try:
            scope = memory.current()
            if scope is None:
                # Mount a scope on the fly if the caller didn't.
                await memory.mount(record.target_agent_id, record.round_id)
                scope = memory.current()
            await memory.episodic().remember(
                scope, plan.plan_id,
                {
                    "trace_id": record.record_id,
                    "agent_id": record.target_agent_id,
                    "failure_signature": "governance:outcome",
                    "outcome": "accepted" if plan.accepted_count else "no_action",
                    "accepted": plan.accepted_count,
                    "projected_savings_usd": plan.total_projected_savings_usd,
                    "round_id": record.round_id,
                },
            )
            step.outputs = {"remembered": True, "plan_id": plan.plan_id}
            step.finish()
        except Exception as exc:  # noqa: BLE001 — memory is best-effort
            step.finish(StepStatus.FAILED, note=f"{type(exc).__name__}: {exc}")


# --------------------------------------------------------------------------- #
# VerificationAgent (async — touches PartitionedMemory)
# --------------------------------------------------------------------------- #
class VerificationAgent:
    """Verifies a deployed plan's projected savings vs post-deployment actuals.

    Implements the CoT in :data:`VERIFICATION_COT_PROMPT`. Runs AFTER a
    governance plan has been deployed and a fresh batch of post-deployment
    traces of the SAME target agent has been collected. Closes the
    projection→actual loop by:

    1. Building the after-graph from post-deployment traces via
       :func:`~voyage_trace.execution_graph.aggregate_execution_graph`.
    2. Calling :func:`~voyage_trace.verification.verify_plan` — REAL graph
       arithmetic (``before.cost - after.cost`` per node), never consulting
       the simulator's projection.
    3. Recalling the agent's :class:`CalibrationState` from semantic memory
       (under the fixed key :attr:`CALIBRATION_KEY` in a
       :attr:`CALIBRATION_ROUND` pseudo-round namespace).
    4. Folding the result into the state via
       :func:`~voyage_trace.verification.update_calibration`.
    5. Persisting the updated :class:`CalibrationState` back to semantic
       memory so the next governance round can recall it.
    6. Stamping each verified proposal's ``actual_savings`` and ``verified``
       flag on the plan so the closed loop is visible at the plan level too.

    Records ``VERIFY`` and (when memory is wired) ``RECALL`` / ``REMEMBER``
    steps on the provided :class:`AnalysisRecord`.
    """

    role: AgentRole = VERIFICATION_ROLE

    #: Fixed key under which the per-agent CalibrationState is stored.
    CALIBRATION_KEY = "calibration_state"
    #: Fixed pseudo-round namespace so the calibration state lives at a
    #: known location independent of any governance round_id.
    CALIBRATION_ROUND = "_calibration"

    async def run(
        self,
        plan: GovernancePlan,
        before_graph: ExecutionGraph,
        after_traces: list[CanonicalTrace],
        record: AnalysisRecord,
        *,
        memory: Any | None = None,
        round_id: str = "",
        note: str = "",
    ) -> tuple[VerificationResult, CalibrationState]:
        """Verify ``plan`` against post-deployment ``after_traces``.

        Returns the :class:`VerificationResult` and the updated
        :class:`CalibrationState`. The result is also stamped back onto
        ``plan`` (each accepted proposal's ``actual_savings`` / ``verified``
        and the plan's ``verification_id``) so the caller's reference to
        the plan reflects the verification without a separate lookup.
        """
        # --- 1. build the after-graph ------------------------------------ #
        if not after_traces:
            raise ValueError(
                "VerificationAgent requires ≥1 post-deployment trace; "
                "got 0. Nothing to verify against."
            )
        after_graph = (
            aggregate_execution_graph(after_traces)
            if len(after_traces) > 1
            else build_execution_graph(after_traces[0])
        )

        # --- 2. verify the plan (REAL graph arithmetic) ------------------ #
        result = verify_plan(
            plan, before_graph, after_graph,
            round_id=round_id or plan.round_id,
            note=note,
        )

        # Stamp each verified proposal's actual_savings + verified flag on
        # the plan so the closed loop is visible at the plan level too.
        # Only accepted proposals were deployed; rejected ones are left
        # untouched (verified=False, actual_savings={}).
        for proposal in plan.accepted_proposals:
            target = proposal.modification.target_node_id
            actual = result.node_actual_savings.get(target)
            if actual is not None:
                proposal.actual_savings = {"cost_delta_usd": float(actual)}
                proposal.verified = True

        # Link the verification back to the plan.
        plan.verification_id = result.verification_id

        record.add_step(AnalysisStep(
            kind=AnalysisStepKind.VERIFY,
            agent_role=self.role.name,
            rationale=(
                f"verify_plan(plan={plan.plan_id}, before_runs={result.before_runs}, "
                f"after_runs={result.after_runs}, mode={result.comparison_mode})"
            ),
            inputs={
                "plan_id": plan.plan_id,
                "before_runs": result.before_runs,
                "after_runs": result.after_runs,
            },
            outputs={
                "verification_id": result.verification_id,
                "verified_count": result.verified_count,
                "unverifiable_count": result.unverifiable_count,
                "total_projected_usd": round(result.total_projected_usd, 6),
                "total_actual_usd": round(result.total_actual_usd, 6),
                "total_error_usd": round(result.total_error_usd, 6),
                "mean_relative_error": (
                    round(result.mean_relative_error, 6)
                    if result.mean_relative_error is not None
                    else None
                ),
            },
        )).finish()

        # --- 3. recall calibration state ---------------------------------- #
        state: CalibrationState | None = None
        if memory is not None:
            state = await self._recall_calibration(
                memory, plan.target_agent_id, record
            )
        if state is None:
            state = CalibrationState(target_agent_id=plan.target_agent_id)

        # --- 4. fold the result into the state ---------------------------- #
        update_calibration(state, result)

        # --- 5. persist the updated state --------------------------------- #
        if memory is not None:
            await self._persist_calibration(memory, state, record)

        return result, state

    async def _recall_calibration(
        self,
        memory: Any,
        target_agent_id: str,
        record: AnalysisRecord,
    ) -> CalibrationState | None:
        """Recall the per-agent CalibrationState from semantic memory."""
        step = AnalysisStep(
            kind=AnalysisStepKind.RECALL,
            agent_role=self.role.name,
            rationale=(
                f"recall calibration τ for {target_agent_id} from semantic memory"
            ),
            inputs={
                "target_agent_id": target_agent_id,
                "key": self.CALIBRATION_KEY,
                "namespace_round": self.CALIBRATION_ROUND,
            },
        )
        record.add_step(step)
        try:
            scope = MemoryScope(
                target_agent_id=target_agent_id,
                round_id=self.CALIBRATION_ROUND,
                partition="semantic",
            )
            raw = await memory.semantic().recall(scope, self.CALIBRATION_KEY)
            if raw is None:
                step.outputs = {"found": False, "tau": None}
                step.finish(note="cold start — no prior calibration state")
                return None
            state = calibration_from_dict(raw)
            step.outputs = {
                "found": True,
                "tau": state.tau,
                "n_observations": state.n_observations,
                "n_plans_verified": state.n_plans_verified,
            }
            step.finish()
            return state
        except Exception as exc:  # noqa: BLE001 — memory is best-effort
            step.finish(StepStatus.FAILED, note=f"{type(exc).__name__}: {exc}")
            return None

    async def _persist_calibration(
        self,
        memory: Any,
        state: CalibrationState,
        record: AnalysisRecord,
    ) -> None:
        """Persist the updated CalibrationState back to semantic memory."""
        step = AnalysisStep(
            kind=AnalysisStepKind.REMEMBER,
            agent_role=self.role.name,
            rationale=(
                f"persist updated calibration τ={state.tau} to semantic memory"
            ),
            inputs={
                "target_agent_id": state.target_agent_id,
                "key": self.CALIBRATION_KEY,
                "namespace_round": self.CALIBRATION_ROUND,
            },
        )
        record.add_step(step)
        try:
            scope = MemoryScope(
                target_agent_id=state.target_agent_id,
                round_id=self.CALIBRATION_ROUND,
                partition="semantic",
            )
            await memory.semantic().remember(
                scope, self.CALIBRATION_KEY, state.to_dict()
            )
            step.outputs = {
                "persisted": True,
                "tau": state.tau,
                "n_observations": state.n_observations,
                "n_plans_verified": state.n_plans_verified,
            }
            step.finish()
        except Exception as exc:  # noqa: BLE001 — memory is best-effort
            step.finish(StepStatus.FAILED, note=f"{type(exc).__name__}: {exc}")


# --------------------------------------------------------------------------- #
# Orchestrator
# --------------------------------------------------------------------------- #
class Orchestrator:
    """Coordinates the sub-agents and threads the AnalysisRecord.

    This is the public entry point for "run one governance round end to
    end". It owns the :class:`AnalysisRecord` (the internal data format)
    and hands it to each sub-agent in turn; the final
    :class:`GovernancePlan` and the populated record are returned to the
    caller and (optionally) persisted to storage.
    """

    def __init__(
        self,
        *,
        ingest: IngestAgent | None = None,
        modeling: ModelingAgent | None = None,
        simulation: SimulationAgent | None = None,
        governance: GovernanceAgent | None = None,
        verification: VerificationAgent | None = None,
    ) -> None:
        self.ingest = ingest or IngestAgent()
        self.modeling = modeling or ModelingAgent()
        self.simulation = simulation or SimulationAgent()
        self.governance = governance or GovernanceAgent()
        self.verification = verification or VerificationAgent()

    async def run(
        self,
        payloads: list[Any],
        *,
        target_agent_id: str,
        round_id: str,
        source_protocol: str | None = None,
        automl_target: str = "cost_usd",
        memory: Any | None = None,
        min_savings_usd: float = 0.0,
        calibration_multiplier: Any = _UNSET,
    ) -> tuple[AnalysisRecord, GovernancePlan]:
        """Run one governance round end to end.

        When ``memory`` is provided AND ``calibration_multiplier`` was not
        explicitly passed, the orchestrator recalls the per-agent calibration
        multiplier ``τ`` from semantic memory (written by a prior
        :meth:`verify_round`) and passes it to the governance agent so
        accept/reject decisions are made on *calibrated* projections. This
        is the feedback path that closes the loop. An explicit
        ``calibration_multiplier`` overrides the recall (useful for tests);
        pass ``None`` explicitly to force the cold-start path even when
        memory is wired.
        """
        record = AnalysisRecord(target_agent_id=target_agent_id, round_id=round_id)

        # Closed loop: recall τ from semantic memory unless the caller
        # explicitly pinned it (including pinning to None for cold-start).
        # ``None`` (cold start) is the honest default when no prior
        # verification has run.
        if calibration_multiplier is _UNSET:
            tau: float | None = None
            if memory is not None:
                tau = await self._recall_tau(memory, target_agent_id, record)
        else:
            tau = calibration_multiplier  # type: ignore[assignment]

        traces = self.ingest.run(
            payloads, record, source_protocol=source_protocol
        )
        if not traces:
            # Nothing to govern — finish the record with an empty plan so
            # the trajectory is still recorded honestly.
            plan = GovernancePlan(
                target_agent_id=target_agent_id,
                round_id=round_id,
                summary="No traces ingested; empty round.",
                analysis_record_id=record.record_id,
            )
            record.plan = plan
            record.finish()
            return record, plan

        modeling_out = self.modeling.run(
            traces, record, automl_target=automl_target
        )
        # The simulation agent validates whatever proposals the modelling
        # agent surfaced (may be empty when AutoML was skipped). It validates
        # against the aggregated graph so the aggregated node_ids targeted by
        # AutoML proposals resolve correctly.
        self.simulation.run(modeling_out.graph, record, record.proposals)
        plan = await self.governance.run(
            record, memory=memory, min_savings_usd=min_savings_usd,
            calibration_multiplier=tau,
        )
        return record, plan

    async def verify_round(
        self,
        plan: GovernancePlan,
        before_graph: ExecutionGraph,
        after_payloads: list[Any],
        *,
        target_agent_id: str | None = None,
        round_id: str = "",
        source_protocol: str | None = None,
        memory: Any | None = None,
        note: str = "",
    ) -> tuple[AnalysisRecord, VerificationResult, CalibrationState]:
        """Verify a deployed ``plan`` against post-deployment ``after_payloads``.

        This is the second half of the closed loop. It ingests the
        post-deployment payloads into traces, builds the after-graph, and
        delegates to :meth:`VerificationAgent.run` to compare projected vs
        actual savings and update the calibration state ``τ``.

        A fresh :class:`AnalysisRecord` is created for the verification
        round (distinct from the record that produced the plan) so the
        verification trajectory is independently auditable. The
        :class:`VerificationResult` is also stamped back onto ``plan``.

        Parameters
        ----------
        plan
            The previously-deployed :class:`GovernancePlan` whose projected
            savings are being verified.
        before_graph
            The aggregated :class:`ExecutionGraph` the plan was produced
            from (the "before" state). Must use the same
            ``<operation_type>:<label>`` node keys the after-graph will
            produce.
        after_payloads
            Raw post-deployment payloads (same format :meth:`run` accepts).
            They MUST be traces of the same target agent the plan targets.
        """
        agent_id = target_agent_id or plan.target_agent_id
        ver_round_id = round_id or f"verify-{plan.round_id}"
        record = AnalysisRecord(
            target_agent_id=agent_id, round_id=ver_round_id,
        )

        # Ingest the post-deployment payloads (reuses IngestAgent so the
        # verification round has the same trace-normalisation path as a
        # governance round — no shortcutting).
        after_traces = self.ingest.run(
            after_payloads, record, source_protocol=source_protocol
        )
        if not after_traces:
            record.finish()
            raise ValueError(
                "verify_round ingested 0 post-deployment traces; "
                "cannot verify against an empty after-state."
            )

        result, state = await self.verification.run(
            plan, before_graph, after_traces, record,
            memory=memory, round_id=ver_round_id, note=note,
        )
        record.finish()
        return record, result, state

    async def _recall_tau(
        self,
        memory: Any,
        target_agent_id: str,
        record: AnalysisRecord,
    ) -> float | None:
        """Recall the per-agent ``τ`` from semantic memory.

        Uses the same fixed key / pseudo-round namespace as
        :meth:`VerificationAgent._persist_calibration`. Returns ``None``
        (cold start) when no calibration state exists yet. Failures are
        best-effort: a ``None`` is returned and a FAILED step is recorded
        rather than aborting the governance round.
        """
        step = AnalysisStep(
            kind=AnalysisStepKind.RECALL,
            agent_role="orchestrator",
            rationale=(
                f"recall calibration τ for {target_agent_id} to calibrate "
                f"governance projections"
            ),
            inputs={
                "target_agent_id": target_agent_id,
                "key": VerificationAgent.CALIBRATION_KEY,
                "namespace_round": VerificationAgent.CALIBRATION_ROUND,
            },
        )
        record.add_step(step)
        try:
            scope = MemoryScope(
                target_agent_id=target_agent_id,
                round_id=VerificationAgent.CALIBRATION_ROUND,
                partition="semantic",
            )
            raw = await memory.semantic().recall(
                scope, VerificationAgent.CALIBRATION_KEY
            )
            if raw is None:
                step.outputs = {"found": False, "tau": None}
                step.finish(note="cold start — no prior calibration state")
                return None
            state = calibration_from_dict(raw)
            step.outputs = {
                "found": True,
                "tau": state.tau,
                "n_observations": state.n_observations,
            }
            step.finish()
            return state.tau
        except Exception as exc:  # noqa: BLE001 — memory is best-effort
            step.finish(StepStatus.FAILED, note=f"{type(exc).__name__}: {exc}")
            return None

    # Convenience: run + render the analysis trajectory as Markdown.
    async def run_with_markdown(self, **kwargs: Any) -> tuple[AnalysisRecord, GovernancePlan, str]:
        record, plan = await self.run(**kwargs)
        return record, plan, render_analysis_markdown(record)


# --------------------------------------------------------------------------- #
# Helper: run an async orchestrator from sync code (for non-async callers)
# --------------------------------------------------------------------------- #
def run_sync(**kwargs: Any) -> tuple[AnalysisRecord, GovernancePlan]:
    """Synchronous wrapper around :meth:`Orchestrator.run`.

    Useful for scripts and tests that don't want to deal with asyncio
    directly. Raises if called from inside a running event loop — in that
    case call ``await Orchestrator().run(...)`` instead.
    """
    return asyncio.run(Orchestrator().run(**kwargs))
