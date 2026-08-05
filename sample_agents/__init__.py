"""Sample digital-employee agents built on deepagents + voyage_trace tracing.

This package ships three reference agents — research, code-review, and
knowledge-base QA — re-expressed as deepagents :class:`SubAgent` specs with
custom tools. Each agent is wired to a low-intrusion
:class:`~sample_agents.tracing.TraceObserver` that captures every model +
tool call as a :class:`~voyage_trace.types.CanonicalTrace` without altering
the agent's behaviour.

Public surface
--------------
* :class:`TraceObserver`, :func:`attach` — the low-intrusion trace bridge.
* :class:`ScriptedChatModel` — deterministic chat model for end-to-end tests
  without a live LLM.
* :class:`LLMConfig`, :class:`LLMConfigSet`, :func:`load_config` — config-file
  LLM setup (DeepSeek API key read from env, never from source).
* :func:`build_research_agent`, :func:`build_code_review_agent`,
  :func:`build_kb_qa_agent` — the three sample agents.
* :data:`RESEARCH_AGENT_ID`, :data:`CODE_REVIEW_AGENT_ID`,
  :data:`KB_QA_AGENT_ID` — their canonical ids.

See ``docs/sample-agents.md`` for the design notes and the low-intrusion
trace contract.
"""

from __future__ import annotations

from .llm_config import LLMConfig, LLMConfigSet, load_config
from .testing import ScriptedChatModel
from .tracing import TraceObserver, attach
from .builder import SubAgentSpec
from .research_agent import RESEARCH_AGENT_ID
from .code_review_agent import CODE_REVIEW_AGENT_ID
from .kb_qa_agent import KB_QA_AGENT_ID
from .research_agent import build_research_agent, build_research_subagent_spec
from .code_review_agent import build_code_review_agent, build_code_review_subagent_spec
from .kb_qa_agent import build_kb_qa_agent, build_kb_qa_subagent_spec

__all__ = [
    "TraceObserver",
    "attach",
    "ScriptedChatModel",
    "LLMConfig",
    "LLMConfigSet",
    "load_config",
    "SubAgentSpec",
    "RESEARCH_AGENT_ID",
    "CODE_REVIEW_AGENT_ID",
    "KB_QA_AGENT_ID",
    "build_research_agent",
    "build_research_subagent_spec",
    "build_code_review_agent",
    "build_code_review_subagent_spec",
    "build_kb_qa_agent",
    "build_kb_qa_subagent_spec",
]
