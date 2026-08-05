"""Knowledge-base QA agent — retrieves from a KB and answers grounded.

Pattern origin: Dify / MaxKB / FastGPT-style KB-QA employee. Re-expressed
as a deepagents :class:`SubAgent` with two tools: ``retrieve`` (KB lookup)
and ``answer_or_escalate`` (compose an answer or escalate to a human).

The retrieval tool ships with a tiny in-memory KB so the agent loop runs
end-to-end in tests. Production deployments swap the retriever for a real
vector store without touching the agent spec.
"""

from __future__ import annotations

from typing import Any

from langchain_core.tools import tool

from .builder import SubAgentSpec
from .tracing import TraceObserver

KB_QA_AGENT_ID = "kb-qa-agent"
KB_QA_SYSTEM_PROMPT = """\
You are a knowledge-base QA sub-agent. To answer the user's question you
MUST follow these steps in order:

1. Call the `retrieve` tool with the user's question as `question`.
2. Inspect the retrieve output:
   - If it is NOT "no hits", call `answer_or_escalate` with
     `mode="answer"` and the retrieved text as `text`.
   - If it IS "no hits", call `answer_or_escalate` with
     `mode="escalate"` and a short note as `text`.
3. Return the `answer_or_escalate` output as your final answer.

Do not use `glob`, `ls`, `read_file`, or any other tool — your only tools
are `retrieve` and `answer_or_escalate`.
"""


# A tiny in-memory KB — production swaps this for a real retriever.
_KB: dict[str, str] = {
    "return policy": "Returns accepted within 30 days with receipt.",
    "shipping time": "Standard shipping: 3-5 business days.",
    "warranty": "1-year limited warranty on all electronics.",
}


@tool
def retrieve(question: str) -> str:
    """Look up ``question`` in the KB; return matching chunk(s) or 'no hits'.

    Deterministic substring match over the sample KB. Production deployments
    replace the body with a vector-store retriever; the tool signature is
    unchanged so the agent spec does not need edits.
    """
    q = question.lower().strip()
    hits = [v for k, v in _KB.items() if k in q or any(w in q for w in k.split())]
    return "\n".join(hits) if hits else "no hits"


@tool
def answer_or_escalate(mode: str, text: str) -> str:
    """Compose a final answer (``mode="answer"``) or escalate
    (``mode="escalate"``) to a human.

    Returns the formatted final reply; the caller surfaces it to the user.
    """
    mode = (mode or "").lower().strip()
    if mode == "escalate":
        return f"[ESCALATED to human] {text}"
    if mode == "answer":
        return f"[ANSWER] {text}"
    return f"[UNKNOWN MODE {mode!r}] {text}"


KB_QA_TOOLS = [retrieve, answer_or_escalate]

KB_QA_SPEC = SubAgentSpec(
    agent_id=KB_QA_AGENT_ID,
    agent_name="KBQAAgent",
    description=(
        "Answers customer questions from the knowledge base, or "
        "escalates to a human when no relevant KB chunk is found."
    ),
    system_prompt=KB_QA_SYSTEM_PROMPT,
    tools=KB_QA_TOOLS,
    orchestrator_prompt=(
        "You are a customer-support orchestrator. Delegate factual "
        "questions to the `kb-qa-agent` subagent via the `task` tool."
    ),
)

build_kb_qa_subagent_spec = KB_QA_SPEC.build_subagent_dict


def build_kb_qa_agent(
    *,
    model: Any | None = None,
    observer: TraceObserver | None = None,
    config_path: str | None = None,
) -> tuple[Any, TraceObserver]:
    """Build a runnable KB-QA agent + its trace observer (see
    :meth:`SubAgentSpec.build_agent`)."""
    return KB_QA_SPEC.build_agent(
        model=model, observer=observer, config_path=config_path
    )
