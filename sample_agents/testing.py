"""Test utilities for sample_agents — a deterministic chat model + scripted tools.

These helpers exist so the sample-agent tests can run end-to-end against the
**real** deepagents stack (no mocks of ``create_deep_agent``, ``SubAgent``,
or the middleware chain) without needing a live LLM. The
:class:`ScriptedChatModel` plays a pre-recorded list of
:class:`~langchain_core.messages.AIMessage` responses, supports
:meth:`bind_tools` (returning ``self`` so deepagents' tool-binding succeeds),
and never produces non-deterministic output — so the resulting
:class:`~voyage_trace.types.CanonicalTrace` is byte-stable across runs.

Honesty contract: this is a **test double**, not a fake "LLM that always
says success". The script author decides what the model "says", including
the tool calls it makes and whether it returns errors. The trace observer
then captures whatever the scripted model actually did — there is no
shortcut from "I want a tool span" to "I emit a tool span".
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult


class ScriptedChatModel(BaseChatModel):
    """A chat model that replays a pre-scripted list of :class:`AIMessage` s.

    Parameters
    ----------
    script:
        The list of :class:`AIMessage` responses to play in order. Each
        ``invoke`` consumes the next message; running out raises
        :class:`StopIteration` (so the test fails loudly rather than
        silently looping).
    model_name:
        Recorded on emitted spans as ``metadata.model``. Defaults to
        ``"scripted-test-model"``.
    """

    script: list[AIMessage]
    _cursor: int = 0
    model_name: str = "scripted-test-model"

    def __init__(self, *, script: list[AIMessage] | None = None,
                 model_name: str = "scripted-test-model") -> None:
        # BaseChatModel is a pydantic model — use model_validate-style init.
        super().__init__(script=list(script or []), model_name=model_name)
        self._cursor = 0

    @property
    def _llm_type(self) -> str:
        return "scripted-test"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        if self._cursor >= len(self.script):
            raise StopIteration(
                f"ScriptedChatModel exhausted its script of {len(self.script)} messages "
                "— the agent looped more times than the test author scripted. "
                "Either extend the script or fix the agent to terminate."
            )
        msg = self.script[self._cursor]
        self._cursor += 1
        return ChatResult(generations=[ChatGeneration(message=msg)])

    def bind_tools(self, tools: Any, **kwargs: Any) -> "ScriptedChatModel":
        """Return ``self`` — the script ignores tool schemas.

        deepagents calls ``model.bind_tools(tools)`` before each model
        invocation. A real model uses this to advertise the tool schema to
        the LLM; our scripted model already knows what to say, so we just
        return self.
        """
        return self

    @classmethod
    def from_texts(cls, *texts: str) -> "ScriptedChatModel":
        """Convenience: build a script from plain string responses."""
        return cls(script=[AIMessage(content=t) for t in texts])

    @classmethod
    def from_iter(cls, items: Iterator[AIMessage | str]) -> "ScriptedChatModel":
        script: list[AIMessage] = []
        for item in items:
            script.append(item if isinstance(item, AIMessage) else AIMessage(content=item))
        return cls(script=script)

    @classmethod
    def from_tool_calls(
        cls, *steps: tuple[str, list[dict[str, Any]] | None],
    ) -> "ScriptedChatModel":
        """Build a script of alternating text/tool_call turns.

        Each ``step`` is ``(text, tool_calls_or_None)``: when
        ``tool_calls`` is not None the scripted AIMessage carries them; when
        it is ``None`` the message is plain text (the agent's final reply).
        """
        script: list[AIMessage] = []
        for text, tool_calls in steps:
            if tool_calls is None:
                script.append(AIMessage(content=text))
            else:
                script.append(AIMessage(content=text, tool_calls=tool_calls))
        return cls(script=script)

    def reset(self) -> None:
        """Rewind the cursor to the start (for re-running a script)."""
        self._cursor = 0
