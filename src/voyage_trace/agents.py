"""Multi-agent architecture for the voyage_trace governance pipeline.

This module splits the analysis / optimisation process into four
specialised sub-agents coordinated by one orchestrator:

    ┌──────────────┐   payloads   ┌──────────────┐  traces  ┌──────────────┐
    │ IngestAgent  │ ───────────► │ ModelingAgent│ ───────► │SimulationAgent│
    │              │   traces     │ (+ AutoML)   │  graph   │              │
    └──────────────┘              └──────────────┘  props   └──────┬───────┘
                                                                        │ validated
                                                                        ▼
                                                            ┌──────────────────┐
                                                            │ GovernanceAgent  │
                                                            │  (decide+memory) │
                                                            └──────────────────┘

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
  work and stay sync. Governance is ``async`` because it touches the
  async :class:`~voyage_trace.memory.manager.PartitionedMemory`. The
  :class:`Orchestrator` is ``async`` to match.
* **AutoML proposes, simulator disposes.** The modelling agent surfaces
  candidate :class:`~voyage_trace.analysis.OptimizationProposal` objects
  from AutoML; the simulation agent fills ``expected_savings``; the
  governance agent accepts only proposals the simulator validated as
  beneficial. No proposal reaches the plan unvalidated.
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
from .simulator import Modification, project_savings, simulate_graph
from .types import CanonicalTrace


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
""",
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
            step.finish(StepStatus.FAILED if report.notes and "No feature" in report.notes[0] else StepStatus.SUCCESS)

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
    ) -> GovernancePlan:
        # Optional cross-round recall before deciding. This is the
        # "recall for reuse" path: past outcomes for the same failure shape
        # inform whether to trust a proposal.
        if memory is not None:
            await self._recall(memory, record)

        accepted: list[OptimizationProposal] = []
        rejected: list[OptimizationProposal] = []
        for proposal in record.proposals:
            cost_delta = proposal.expected_savings.get("cost_delta_usd", 0.0)
            if proposal.validated and cost_delta >= min_savings_usd:
                proposal.accept(
                    rationale=f"validated; projected saving ${cost_delta:.6f} >= ${min_savings_usd}"
                )
                accepted.append(proposal)
            else:
                reason = "not validated" if not proposal.validated else (
                    f"saving ${cost_delta:.6f} < ${min_savings_usd}"
                )
                proposal.reject(rationale=reason)
                rejected.append(proposal)
            record.add_step(AnalysisStep(
                kind=AnalysisStepKind.DECIDE,
                agent_role=self.role.name,
                rationale=proposal.decision_rationale,
                inputs={"proposal_id": proposal.proposal_id},
                outputs={"decision": proposal.decision.value if proposal.decision else "—"},
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
# Orchestrator
# --------------------------------------------------------------------------- #
class Orchestrator:
    """Coordinates the four sub-agents and threads the AnalysisRecord.

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
    ) -> None:
        self.ingest = ingest or IngestAgent()
        self.modeling = modeling or ModelingAgent()
        self.simulation = simulation or SimulationAgent()
        self.governance = governance or GovernanceAgent()

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
    ) -> tuple[AnalysisRecord, GovernancePlan]:
        record = AnalysisRecord(target_agent_id=target_agent_id, round_id=round_id)

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
            record, memory=memory, min_savings_usd=min_savings_usd
        )
        return record, plan

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
