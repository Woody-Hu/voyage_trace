"""Shared spec + builder for the sample subagents.

The three sample agents (research / code-review / KB-QA) share an identical
shape: a :class:`SubAgent` spec with custom tools, mounted on a parent
orchestrator via the ``task`` tool, with a single
:class:`~sample_agents.tracing.TraceObserver` capturing both the parent and
the subagent. This module factors that shape into one dataclass + one
builder so each agent module only declares *what is different* (id, prompt,
tools).

Public API is unchanged: ``build_research_agent`` etc. still exist as
thin wrappers — the refactor is purely structural.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from langchain_core.tools import BaseTool

from .llm_config import LLMConfig, load_config
from .tracing import TraceObserver, attach


@dataclass(frozen=True)
class SubAgentSpec:
    """Declarative description of one sample subagent.

    Each sample-agent module instantiates this once and calls
    :meth:`build_agent` / :meth:`build_subagent_dict` on it — no per-agent
    builder code is duplicated.
    """

    agent_id: str
    agent_name: str
    description: str
    system_prompt: str
    tools: list[BaseTool | Callable] = field(default_factory=list)
    orchestrator_prompt: str = (
        "You are an orchestrator. Delegate to the `{agent_id}` subagent via "
        "the `task` tool."
    )

    def build_subagent_dict(
        self,
        *,
        model: Any | None = None,
        config: LLMConfig | None = None,
        observer: TraceObserver | None = None,
    ) -> dict[str, Any]:
        """Return a :class:`SubAgent`-shaped dict for ``create_deep_agent``.

        When ``observer`` is provided it is attached to the subagent's own
        middleware chain so the subagent's internal model + tool calls are
        captured by the same :class:`TraceObserver` instance as the parent.
        """
        spec: dict[str, Any] = {
            "name": self.agent_id,
            "description": self.description,
            "system_prompt": self.system_prompt,
            "tools": list(self.tools),
        }
        if model is not None:
            spec["model"] = model
        elif config is not None:
            # Deferred model construction; consumed by build_agent below.
            spec["_llm_config"] = config
        if observer is not None:
            spec["middleware"] = [observer]
        return spec

    def build_agent(
        self,
        *,
        model: Any | None = None,
        observer: TraceObserver | None = None,
        config_path: str | None = None,
    ) -> tuple[Any, TraceObserver]:
        """Build a runnable parent agent + its trace observer.

        Returns ``(agent, observer)``. ``agent`` is a deepagents
        ``CompiledStateGraph``; ``observer`` captures the trace.

        The same observer is attached to BOTH the parent and the subagent, so
        one :class:`~voyage_trace.types.CanonicalTrace` captures the full
        delegation chain. When ``model`` is ``None`` it is built from the
        loaded LLM config (``config_path`` or default discovery); a missing
        key raises rather than silently falling back to a "free" model.
        """
        from deepagents import create_deep_agent

        if observer is None:
            observer = attach(self.agent_id, agent_name=self.agent_name)

        if model is None:
            cfg_set = load_config(config_path)
            cfg = cfg_set.for_agent(self.agent_id)
            model = cfg.build_chat_model(config_path=cfg_set.path)

        spec = self.build_subagent_dict(model=model, observer=observer)
        agent = create_deep_agent(
            model=model,
            subagents=[spec],  # type: ignore[arg-type]
            middleware=[observer],
            system_prompt=self.orchestrator_prompt.format(agent_id=self.agent_id),
        )
        return agent, observer
