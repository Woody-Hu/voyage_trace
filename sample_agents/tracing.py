"""Low-intrusion trace observer for deepagents.

This module is the **bridge** between deepagents and voyage_trace's
:class:`CanonicalTrace` protocol. It provides one middleware class —
:class:`TraceObserver` — that wraps every model call and every tool call a
deepagents agent makes, emits one :class:`~voyage_trace.types.TraceSpan` per
call, and lets the caller harvest the assembled :class:`CanonicalTrace` at
the end. The observer is **additive only**: it returns
``handler(request)`` verbatim, never mutates state, never raises into the
agent loop, and is composable with every other deepagents middleware.

Design contract (the "low-intrusion" guarantees)
-------------------------------------------------
1. **No behaviour change.** ``wrap_model_call`` and ``wrap_tool_call``
   forward the request to ``handler`` and return its result unchanged. The
   agent's outputs are byte-identical with and without the observer.
2. **No silent failures.** Anything the observer cannot interpret is
   recorded as an empty span (or skipped), never raised into the agent
   loop. A bug in the observer must not corrupt a real agent run.
3. **One span per call.** Each model invocation is one CHAT span; each
   tool invocation is one EXECUTE_TOOL span. The two span kinds share a
   trace id, so the resulting :class:`CanonicalTrace` is a flat list of
   agent steps that flows straight through :func:`aggregate_execution_graph`
   / :func:`run_automl`.
4. **Real token counts when available.** When the underlying model fills
   ``usage_metadata`` (OpenAI / Anthropic / DeepSeek do), the observer
   reads ``input_tokens`` / ``output_tokens`` from it. When it doesn't, the
   span records zeros rather than fabricating numbers.
5. **No SDK lock-in.** The observer imports only deepagents' public
   ``AgentMiddleware`` base and langchain's message types — no LangSmith,
   no Langfuse, no OTel. The trace is a plain :class:`CanonicalTrace` that
   can be pushed to any backend via :mod:`voyage_trace.integrations`.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

from langchain.agents.middleware import AgentMiddleware
from langchain.agents.middleware.types import ModelRequest, ModelResponse, ToolCallRequest
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage

from voyage_trace.protocol import normalise
from voyage_trace.types import (
    CanonicalTrace,
    OperationType,
    SourceProtocol,
    SpanStatus,
    TraceSpan,
)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _coerce_str(value: Any, *, limit: int = 4000) -> str:
    """Best-effort string coercion with a size cap (keeps traces small)."""
    if value is None:
        return ""
    if isinstance(value, str):
        text = value
    elif isinstance(value, (list, tuple)):
        text = " ".join(_coerce_str(v, limit=limit) for v in value)
    elif isinstance(value, dict):
        text = str(value)
    else:
        text = str(value)
    return text if len(text) <= limit else text[:limit] + "…"


def _extract_usage(msg: AIMessage) -> tuple[int, int]:
    """Read (input_tokens, output_tokens) from an AIMessage's usage_metadata.

    Returns (0, 0) when the model did not populate usage — never fabricates.
    """
    usage = getattr(msg, "usage_metadata", None) or {}
    return int(usage.get("input_tokens", 0) or 0), int(usage.get("output_tokens", 0) or 0)


def _last_ai_message(messages: list[BaseMessage] | None) -> AIMessage | None:
    if not messages:
        return None
    for msg in reversed(messages):
        if isinstance(msg, AIMessage):
            return msg
    return None


def _tool_call_status(result: Any) -> tuple[SpanStatus, str | None]:
    """Inspect a tool result to decide status + error text."""
    # ToolMessage carries an explicit ``status`` ("success" / "error").
    if isinstance(result, ToolMessage):
        status_str = str(getattr(result, "status", "") or "").lower()
        if status_str == "error":
            return SpanStatus.FAILED, _coerce_str(result.content, limit=500)
        return SpanStatus.SUCCESS, None
    # Command objects (deepagents returns them for handoffs).
    if hasattr(result, "update"):
        return SpanStatus.SUCCESS, None
    return SpanStatus.SUCCESS, None


def _model_name(model: Any) -> str:
    """Best-effort model identifier extraction across LangChain model shapes.

    LangChain chat models expose the model identifier as ``model_name`` or
    ``model`` (newer versions), not ``.name`` (which is often the class name
    or empty). Reading only ``.name`` collapsed every chat span into a single
    ``chat:chat`` node in the aggregated graph, starving AutoML of rows.
    """
    for attr in ("model_name", "model", "name"):
        val = getattr(model, attr, None)
        if isinstance(val, str) and val:
            return val
    return ""


class TraceObserver(AgentMiddleware):
    """A no-op deepagents middleware that captures one :class:`CanonicalTrace`.

    Drop an instance into ``create_deep_agent(middleware=[TraceObserver(...)])``
    and the agent's every model + tool call is recorded as a span. Call
    :meth:`finalize` after the agent finishes to get the assembled trace.
    """

    def __init__(
        self,
        *,
        agent_id: str | None = None,
        agent_name: str = "",
        session_id: str = "",
        trace_id: str | None = None,
        source_protocol: SourceProtocol = SourceProtocol.CUSTOM,
    ) -> None:
        self.agent_id = agent_id or f"agent-{uuid.uuid4().hex[:8]}"
        self.agent_name = agent_name
        self.session_id = session_id
        self.trace_id = trace_id or f"trace-{uuid.uuid4().hex[:12]}"
        self.source_protocol = source_protocol
        # Internal span buffer — appended to in wrap_*_call hooks.
        self._spans: list[TraceSpan] = []
        self._counter = 0

    # -- public API ------------------------------------------------------- #
    @property
    def spans(self) -> list[TraceSpan]:
        """The captured spans so far (a snapshot, safe to iterate)."""
        return list(self._spans)

    def finalize(self) -> CanonicalTrace:
        """Build and normalise the :class:`CanonicalTrace` from captured spans.

        Calling :meth:`finalize` does not stop further capture — the observer
        keeps recording into ``self._spans``. Calling it again returns a fresh
        trace with whatever has been captured up to that point.
        """
        trace = CanonicalTrace(
            trace_id=self.trace_id,
            agent_id=self.agent_id,
            agent_name=self.agent_name,
            session_id=self.session_id,
            source_protocol=self.source_protocol,
            spans=list(self._spans),
            metadata={"observer": "deepagents.TraceObserver"},
        )
        return normalise(trace)

    def reset(self, *, trace_id: str | None = None) -> None:
        """Clear captured spans and (optionally) start a new trace id.

        Useful when reusing the same observer instance across multiple
        invocations: call ``reset(trace_id=...)`` between agent runs.
        """
        self._spans.clear()
        self._counter = 0
        if trace_id:
            self.trace_id = trace_id

    # -- middleware hooks ------------------------------------------------- #
    def _next_id(self, prefix: str) -> str:
        self._counter += 1
        return f"{prefix}-{self._counter:04d}"

    def wrap_model_call(
        self,
        request: ModelRequest,  # type: ignore[valid-type]
        handler,
    ):
        start = _utcnow()
        try:
            response: ModelResponse = handler(request)  # type: ignore[valid-type]
        except Exception:  # noqa: BLE001 — never propagate observer failures
            # The exception is allowed to propagate to the agent (it is the
            # agent's own failure, not ours); we just don't emit a span.
            raise
        end = _utcnow()
        try:
            self._record_model_call(request, response, start, end)
        except Exception:  # noqa: BLE001 — observer must never crash the agent
            pass
        return response

    async def awrap_model_call(
        self,
        request: ModelRequest,  # type: ignore[valid-type]
        handler,
    ):
        start = _utcnow()
        try:
            response: ModelResponse = await handler(request)  # type: ignore[valid-type]
        except Exception:  # noqa: BLE001
            raise
        end = _utcnow()
        try:
            self._record_model_call(request, response, start, end)
        except Exception:  # noqa: BLE001
            pass
        return response

    def wrap_tool_call(
        self,
        request: ToolCallRequest,  # type: ignore[valid-type]
        handler,
    ):
        start = _utcnow()
        try:
            result = handler(request)  # type: ignore[valid-type]
        except Exception:  # noqa: BLE001
            # Tool raised — record the failure then re-raise.
            end = _utcnow()
            try:
                self._record_tool_call(request, None, start, end, failed=True)
            except Exception:  # noqa: BLE001
                pass
            raise
        end = _utcnow()
        try:
            self._record_tool_call(request, result, start, end, failed=False)
        except Exception:  # noqa: BLE001
            pass
        return result

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,  # type: ignore[valid-type]
        handler,
    ):
        start = _utcnow()
        try:
            result = await handler(request)  # type: ignore[valid-type]
        except Exception:  # noqa: BLE001
            end = _utcnow()
            try:
                self._record_tool_call(request, None, start, end, failed=True)
            except Exception:  # noqa: BLE001
                pass
            raise
        end = _utcnow()
        try:
            self._record_tool_call(request, result, start, end, failed=False)
        except Exception:  # noqa: BLE001
            pass
        return result

    # -- internal recording ---------------------------------------------- #
    def _record_model_call(
        self,
        request: ModelRequest,  # type: ignore[valid-type]
        response: ModelResponse,  # type: ignore[valid-type]
        start: datetime,
        end: datetime,
    ) -> None:
        last_ai = _last_ai_message(getattr(response, "result", None))
        in_tok, out_tok = _extract_usage(last_ai) if last_ai else (0, 0)
        model_name = _model_name(getattr(request, "model", None))
        # Truncate inputs to the last user message — full transcripts make
        # the trace huge without adding analytical value (we already have
        # the graph for the call sequence).
        inputs_snapshot: dict[str, Any] = {
            "model": model_name,
            "messages": [
                {"role": type(m).__name__, "content": _coerce_str(m.content, limit=500)}
                for m in (getattr(request, "messages", None) or [])[-3:]
            ],
        }
        outputs_snapshot: dict[str, Any] = {
            "content": _coerce_str(getattr(last_ai, "content", ""), limit=1000),
        }
        if getattr(last_ai, "tool_calls", None):
            outputs_snapshot["tool_calls"] = [
                {"name": tc.get("name", ""), "args": tc.get("args", {})}
                for tc in last_ai.tool_calls
            ]
        span = TraceSpan(
            trace_id=self.trace_id,
            span_id=self._next_id("chat"),
            parent_span_id=None,
            session_id=self.session_id,
            agent_id=self.agent_id,
            agent_name=self.agent_name,
            operation_type=OperationType.CHAT,
            status=SpanStatus.SUCCESS,
            start_time=start,
            end_time=end,
            inputs=inputs_snapshot,
            outputs=outputs_snapshot,
            input_tokens=in_tok,
            output_tokens=out_tok,
            # ``name`` aligns with the convention every adapter / synthetic
            # trace uses (``metadata["name"]``); ``aggregate_execution_graph``
            # buckets nodes by it, so without it every chat span collapses
            # into one node and AutoML loses all signal.
            metadata={
                "name": model_name or "chat",
                "model": model_name,
                "kind": "deepagents.model_call",
            },
            source_protocol=self.source_protocol,
        )
        self._spans.append(span)

    def _record_tool_call(
        self,
        request: ToolCallRequest,  # type: ignore[valid-type]
        result: Any,
        start: datetime,
        end: datetime,
        *,
        failed: bool,
    ) -> None:
        tc = getattr(request, "tool_call", {}) or {}
        name = str(tc.get("name", "tool"))
        args = tc.get("args", {}) or {}
        # The id from the model's tool_call request — keeps the span linked
        # back to the AIMessage that produced it (after protocol normalisation
        # parents are recomputed; we leave parent_span_id None and let the
        # protocol derive dotted_order from start_time).
        tool_call_id = tc.get("id")
        status, err = (SpanStatus.FAILED, "tool raised") if failed else _tool_call_status(result)
        outputs_snapshot: dict[str, Any]
        if result is None:
            outputs_snapshot = {"error": err or "tool failed"}
        elif isinstance(result, ToolMessage):
            outputs_snapshot = {"content": _coerce_str(result.content, limit=1000)}
        elif hasattr(result, "update"):
            # Command: extract the first ToolMessage in the update.
            msgs = (getattr(result, "update", {}) or {}).get("messages", [])
            first_msg = next(iter(msgs), None) if msgs else None
            outputs_snapshot = {
                "content": _coerce_str(getattr(first_msg, "content", ""), limit=1000),
                "kind": "command",
            }
        else:
            outputs_snapshot = {"content": _coerce_str(result, limit=1000)}
        span = TraceSpan(
            trace_id=self.trace_id,
            span_id=self._next_id(f"tool_{name}"),
            parent_span_id=None,
            session_id=self.session_id,
            agent_id=self.agent_id,
            agent_name=self.agent_name,
            operation_type=OperationType.EXECUTE_TOOL,
            status=status,
            start_time=start,
            end_time=end,
            inputs={"name": name, "args": args, "tool_call_id": tool_call_id},
            outputs=outputs_snapshot,
            error=err,
            # ``name`` aligns with the convention every adapter / synthetic
            # trace uses; ``aggregate_execution_graph`` buckets nodes by it.
            # ``tool`` is kept for backwards compatibility with existing
            # tests that assert ``metadata["tool"]``.
            metadata={
                "name": name,
                "tool": name,
                "kind": "deepagents.tool_call",
            },
            source_protocol=self.source_protocol,
        )
        self._spans.append(span)


def attach(
    agent_id: str,
    *,
    agent_name: str = "",
    session_id: str = "",
    trace_id: str | None = None,
) -> TraceObserver:
    """Convenience constructor — returns a fresh :class:`TraceObserver`.

    Kept as a function (not a method) so it composes with the
    ``middleware=[...]`` argument to ``create_deep_agent`` directly:

        observer = attach("research-agent")
        agent = create_deep_agent(model=..., middleware=[observer])
        agent.invoke(...)
        trace = observer.finalize()
    """
    return TraceObserver(
        agent_id=agent_id,
        agent_name=agent_name,
        session_id=session_id,
        trace_id=trace_id,
    )
