"""Research subagent — gathers findings on a topic via search + summarise.

Pattern origin: common research-agent pattern (OpenHands / SMTense /
"research-agent" GitHub archetypes). Re-expressed as a deepagents
:class:`SubAgent` with two custom tools (``search`` and ``summarise``),
so the model drives a small tool loop while the trace observer captures
each step.

The tools are deterministic pure-Python stubs in this sample — production
deployments replace them with real search APIs without touching the agent
spec.
"""

from __future__ import annotations

from typing import Any

from langchain_core.tools import tool

from .builder import SubAgentSpec
from .tracing import TraceObserver

RESEARCH_AGENT_ID = "research-agent"
RESEARCH_SYSTEM_PROMPT = """\
You are a research sub-agent. To answer the user's question you MUST follow
these steps in order:

1. Call the `search` tool with the user's question as `query`.
2. Call the `summarise` tool, passing the concatenated search hits as
   `findings`.
3. Return the `summarise` output verbatim as your final answer.

Do not answer from memory, and do not use any other tool (no glob / ls /
read_file). Your only tools are `search` and `summarise`.
"""


# --------------------------------------------------------------------------- #
# Tools — deterministic sample implementations (replace in production).
# --------------------------------------------------------------------------- #
@tool
def search(query: str) -> str:
    """Search the knowledge base for ``query`` and return up to 3 hits.

    This sample implementation returns deterministic canned hits so the agent
    loop is reproducible. In production, swap the body for a real search call
    (Tavily / Brave / etc.); the tool signature does not change.
    """
    return (
        f"[1] {query}: doc-alpha — overview of {query}.\n"
        f"[2] {query}: doc-beta — comparison of {query} alternatives.\n"
        f"[3] {query}: doc-gamma — recent benchmarks for {query}."
    )


@tool
def summarise(findings: str) -> str:
    """Summarise the search ``findings`` into 3 concise bullet points.

    Pure-Python deterministic transformation — no LLM call. The agent calls
    this tool with the concatenated search hits; the tool returns the bullets.
    """
    lines = [ln for ln in findings.splitlines() if ln.strip()]
    bullets: list[str] = []
    for ln in lines[:3]:
        # Strip the leading "[N] topic: " prefix and turn into a bullet.
        text = ln.split(":", 1)[-1].strip(" —") if ":" in ln else ln
        bullets.append(f"- {text}")
    if not bullets:
        return "- (no findings)"
    return "\n".join(bullets)


RESEARCH_TOOLS = [search, summarise]

RESEARCH_SPEC = SubAgentSpec(
    agent_id=RESEARCH_AGENT_ID,
    agent_name="ResearchAgent",
    description=(
        "Researches a topic by calling search + summarise, then returns "
        "a 3-bullet summary. Use for any factual question."
    ),
    system_prompt=RESEARCH_SYSTEM_PROMPT,
    tools=RESEARCH_TOOLS,
    orchestrator_prompt=(
        "You are a research orchestrator. Delegate research questions to "
        "the `research-agent` subagent via the `task` tool."
    ),
)

# Public API preserved (thin wrappers over the shared builder).
build_research_subagent_spec = RESEARCH_SPEC.build_subagent_dict


def build_research_agent(
    *,
    model: Any | None = None,
    observer: TraceObserver | None = None,
    config_path: str | None = None,
) -> tuple[Any, TraceObserver]:
    """Build a runnable research agent + its trace observer (see
    :meth:`SubAgentSpec.build_agent`)."""
    return RESEARCH_SPEC.build_agent(
        model=model, observer=observer, config_path=config_path
    )
