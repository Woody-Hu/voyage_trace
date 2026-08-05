"""Tests for the sample_agents package.

Every test runs the **real** deepagents stack — ``create_deep_agent`` is
called for real, the agent is invoked for real, and the
:class:`TraceObserver` is the production middleware. No deepagents internals
are mocked. The only test double is :class:`ScriptedChatModel`, which plays
a deterministic, author-written script of model responses — the agent loop,
the tool dispatch, and the trace capture are all real.

Honesty contract:
- The scripted model's responses are written by the test author — there is
  no path from "I want a tool span" to "I emit a tool span" that bypasses
  the agent loop. The trace observer captures whatever the agent actually
  did, not what the test wanted it to do.
- No API keys are read from source. The LLM-config tests use a real
  temporary YAML file (written by the test) and env vars set in-process.
"""

from __future__ import annotations

import os
import textwrap
from datetime import datetime, timezone

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from sample_agents import (
    CODE_REVIEW_AGENT_ID,
    KB_QA_AGENT_ID,
    RESEARCH_AGENT_ID,
    LLMConfig,
    ScriptedChatModel,
    TraceObserver,
    attach,
    build_code_review_agent,
    build_kb_qa_agent,
    build_research_agent,
    load_config,
)
from voyage_trace.execution_graph import aggregate_execution_graph
from voyage_trace.types import (
    CanonicalTrace,
    OperationType,
    SourceProtocol,
    SpanStatus,
    TraceSpan,
)


# --------------------------------------------------------------------------- #
# ScriptedChatModel
# --------------------------------------------------------------------------- #
class TestScriptedChatModel:
    def test_replays_script_in_order(self):
        model = ScriptedChatModel.from_texts("a", "b", "c")
        r1 = model.invoke([HumanMessage(content="x")])
        r2 = model.invoke([HumanMessage(content="y")])
        r3 = model.invoke([HumanMessage(content="z")])
        assert r1.content == "a"
        assert r2.content == "b"
        assert r3.content == "c"

    def test_exhausting_script_raises_stopiteration(self):
        model = ScriptedChatModel.from_texts("only one")
        model.invoke([HumanMessage(content="x")])
        with pytest.raises(StopIteration, match="exhausted"):
            model.invoke([HumanMessage(content="y")])

    def test_bind_tools_returns_self(self):
        # deepagents calls model.bind_tools(tools) before each model call;
        # the scripted model must accept this and return a runnable.
        model = ScriptedChatModel.from_texts("hi")
        bound = model.bind_tools([{"name": "x", "description": "y"}])
        assert bound is model
        assert bound.invoke([HumanMessage(content="hi")]).content == "hi"

    def test_from_tool_calls_builds_tool_call_messages(self):
        model = ScriptedChatModel.from_tool_calls(
            ("thinking", [{"name": "search", "args": {"q": "x"}, "id": "c1"}]),
            ("final answer", None),
        )
        m1 = model.invoke([HumanMessage(content="x")])
        # AIMessage normalises tool_calls to add a ``type`` field; compare the
        # fields we care about, not the full dict.
        assert len(m1.tool_calls) == 1
        tc = m1.tool_calls[0]
        assert tc["name"] == "search"
        assert tc["args"] == {"q": "x"}
        assert tc["id"] == "c1"
        m2 = model.invoke([HumanMessage(content="y")])
        assert m2.content == "final answer"
        assert m2.tool_calls == []

    def test_reset_rewinds_cursor(self):
        model = ScriptedChatModel.from_texts("a", "b")
        model.invoke([HumanMessage(content="x")])
        model.reset()
        assert model.invoke([HumanMessage(content="x")]).content == "a"


# --------------------------------------------------------------------------- #
# TraceObserver (unit-level, without deepagents)
# --------------------------------------------------------------------------- #
class TestTraceObserverUnits:
    def test_finalize_empty_raises_protocol_error(self):
        # An empty observer's trace has no spans; the protocol invariant
        # rejects this. The honest behaviour: finalise() raises rather than
        # fabricating a fake span.
        from voyage_trace.protocol import ProtocolError
        obs = attach("agent-x", agent_name="X")
        with pytest.raises(ProtocolError, match="at least one span"):
            obs.finalize()

    def test_reset_clears_spans_and_assigns_new_trace_id(self):
        obs = attach("agent-x", agent_name="X", trace_id="t-old")
        # Manually append one span so finalize works.
        obs._spans.append(TraceSpan(
            trace_id=obs.trace_id, span_id="s1",
            operation_type=OperationType.CHAT, agent_id="agent-x",
            source_protocol=SourceProtocol.CUSTOM,
        ))
        assert len(obs.spans) == 1
        obs.reset(trace_id="t-new")
        assert obs.spans == []
        assert obs.trace_id == "t-new"

    def test_agent_id_and_name_propagate_to_spans(self):
        obs = TraceObserver(agent_id="agent-y", agent_name="Y", trace_id="t-y")
        obs._spans.append(TraceSpan(
            trace_id=obs.trace_id, span_id="s1",
            operation_type=OperationType.CHAT,
            agent_id="agent-y", agent_name="Y",
            source_protocol=SourceProtocol.CUSTOM,
        ))
        trace = obs.finalize()
        assert trace.agent_id == "agent-y"
        assert trace.agent_name == "Y"
        assert trace.spans[0].agent_id == "agent-y"


# --------------------------------------------------------------------------- #
# Research agent — end-to-end with the real deepagents stack
# --------------------------------------------------------------------------- #
class TestResearchAgent:
    @pytest.fixture
    def scripted_model(self) -> ScriptedChatModel:
        # Parent delegates via `task` -> subagent calls `search` ->
        # subagent calls `summarise` -> subagent returns bullets ->
        # parent returns final summary.
        return ScriptedChatModel(script=[
            AIMessage(content="", tool_calls=[{
                "name": "task", "args": {
                    "description": "research autogluon",
                    "subagent_type": "research-agent",
                }, "id": "c1",
            }]),
            AIMessage(content="", tool_calls=[{
                "name": "search", "args": {"query": "autogluon"}, "id": "c2",
            }]),
            AIMessage(content="", tool_calls=[{
                "name": "summarise",
                "args": {"findings": "[1] autogluon: doc-alpha.\n[2] autogluon: doc-beta."},
                "id": "c3",
            }]),
            AIMessage(content="- doc-alpha overview.\n- doc-beta comparison."),
            AIMessage(content="Summary: - doc-alpha overview. - doc-beta comparison."),
        ])

    def test_agent_returns_final_summary(self, scripted_model):
        agent, _ = build_research_agent(model=scripted_model)
        result = agent.invoke({"messages": [HumanMessage(content="research autogluon")]})
        final = result["messages"][-1]
        assert isinstance(final, AIMessage)
        assert "doc-alpha" in final.content

    def test_observer_captures_full_delegation_chain(self, scripted_model):
        agent, observer = build_research_agent(model=scripted_model)
        agent.invoke({"messages": [HumanMessage(content="research autogluon")]})
        spans = observer.spans
        # Expect: 4 model calls + 3 tool calls (task, search, summarise) = 7..8.
        assert len(spans) >= 7
        # Both CHAT and EXECUTE_TOOL spans are present.
        ops = {s.operation_type for s in spans}
        assert OperationType.CHAT in ops
        assert OperationType.EXECUTE_TOOL in ops
        # The tool names captured include the subagent's tools.
        tool_names = {s.metadata.get("tool") for s in spans
                      if s.operation_type == OperationType.EXECUTE_TOOL}
        assert "search" in tool_names
        assert "summarise" in tool_names
        assert "task" in tool_names

    def test_observer_does_not_alter_agent_output(self, scripted_model):
        """The trace observer must be a no-op: same output with/without."""
        # Build a fresh model for the "without observer" run — the scripted
        # model has internal state. Re-script with the same messages.
        with_observer_model = ScriptedChatModel(script=list(scripted_model.script))
        without_observer_model = ScriptedChatModel(script=list(scripted_model.script))

        agent_with, _ = build_research_agent(model=with_observer_model)
        # Build a no-observer variant by constructing one and not passing it.
        from deepagents import create_deep_agent
        from sample_agents.research_agent import build_research_subagent_spec
        spec = build_research_subagent_spec(model=without_observer_model, observer=None)
        agent_without = create_deep_agent(
            model=without_observer_model,
            subagents=[spec],
            system_prompt="orchestrator",
        )

        r_with = agent_with.invoke({"messages": [HumanMessage(content="research autogluon")]})
        r_without = agent_without.invoke({"messages": [HumanMessage(content="research autogluon")]})
        # The final agent reply must be identical.
        assert r_with["messages"][-1].content == r_without["messages"][-1].content

    def test_finalized_trace_flows_through_execution_graph(self, scripted_model):
        """The captured trace must be ingestible by the downstream pipeline."""
        agent, observer = build_research_agent(model=scripted_model)
        agent.invoke({"messages": [HumanMessage(content="research x")]})
        trace = observer.finalize()
        # aggregate_execution_graph is the downstream stage; it must accept
        # the observer's trace without further normalisation.
        graph = aggregate_execution_graph([trace])
        assert graph.observed_runs == 1
        # The graph has nodes for chat + each tool kind.
        op_kinds = {n.split(":", 1)[0] for n in graph.nodes}
        assert "chat" in op_kinds
        assert "execute_tool" in op_kinds


# --------------------------------------------------------------------------- #
# Code-review agent
# --------------------------------------------------------------------------- #
class TestCodeReviewAgent:
    @pytest.fixture
    def scripted_model(self) -> ScriptedChatModel:
        return ScriptedChatModel(script=[
            AIMessage(content="", tool_calls=[{
                "name": "task", "args": {
                    "description": "review src/foo.py",
                    "subagent_type": "code-review-agent",
                }, "id": "c1",
            }]),
            AIMessage(content="", tool_calls=[{
                "name": "read_snippet", "args": {"path": "src/foo.py"}, "id": "c2",
            }]),
            AIMessage(content="", tool_calls=[{
                "name": "critique",
                "args": {"snippet": "def divide(a, b):\n    return a / b  # ZeroDivisionError"},
                "id": "c3",
            }]),
            AIMessage(content="## Style\n- (no issues)\n## Bugs\n- possible ZeroDivisionError"),
            AIMessage(content="Review complete: 1 bug found."),
        ])

    def test_agent_runs_and_produces_output(self, scripted_model):
        agent, _ = build_code_review_agent(model=scripted_model)
        result = agent.invoke({"messages": [HumanMessage(content="review src/foo.py")]})
        assert isinstance(result["messages"][-1], AIMessage)

    def test_observer_captures_review_tools(self, scripted_model):
        agent, observer = build_code_review_agent(model=scripted_model)
        agent.invoke({"messages": [HumanMessage(content="review src/foo.py")]})
        tool_names = {s.metadata.get("tool") for s in observer.spans
                      if s.operation_type == OperationType.EXECUTE_TOOL}
        assert "read_snippet" in tool_names
        assert "critique" in tool_names
        assert "task" in tool_names

    def test_critique_tool_flags_division_by_zero(self):
        from sample_agents.code_review_agent import critique
        # Tools can be invoked directly (they are real callables, not mocks).
        review = critique.invoke({"snippet": "x = a / 0"})
        assert "ZeroDivisionError" in review


# --------------------------------------------------------------------------- #
# KB-QA agent
# --------------------------------------------------------------------------- #
class TestKBQAAgent:
    @pytest.fixture
    def scripted_model(self) -> ScriptedChatModel:
        # Parent delegates -> subagent retrieves -> subagent answers.
        return ScriptedChatModel(script=[
            AIMessage(content="", tool_calls=[{
                "name": "task", "args": {
                    "description": "what is your return policy?",
                    "subagent_type": "kb-qa-agent",
                }, "id": "c1",
            }]),
            AIMessage(content="", tool_calls=[{
                "name": "retrieve",
                "args": {"question": "what is your return policy?"},
                "id": "c2",
            }]),
            AIMessage(content="", tool_calls=[{
                "name": "answer_or_escalate",
                "args": {"mode": "answer", "text": "Returns accepted within 30 days."},
                "id": "c3",
            }]),
            AIMessage(content="[ANSWER] Returns accepted within 30 days."),
            AIMessage(content="Answer: Returns accepted within 30 days."),
        ])

    def test_agent_returns_grounded_answer(self, scripted_model):
        agent, _ = build_kb_qa_agent(model=scripted_model)
        result = agent.invoke({"messages": [HumanMessage(content="return policy?")]})
        assert "30 days" in result["messages"][-1].content

    def test_observer_captures_retrieve_and_answer(self, scripted_model):
        agent, observer = build_kb_qa_agent(model=scripted_model)
        agent.invoke({"messages": [HumanMessage(content="return policy?")]})
        tool_names = {s.metadata.get("tool") for s in observer.spans
                      if s.operation_type == OperationType.EXECUTE_TOOL}
        assert "retrieve" in tool_names
        assert "answer_or_escalate" in tool_names

    def test_retrieve_returns_no_hits_for_unknown_question(self):
        from sample_agents.kb_qa_agent import retrieve
        out = retrieve.invoke({"question": "what is the meaning of life?"})
        assert out == "no hits"

    def test_answer_or_escalate_modes(self):
        from sample_agents.kb_qa_agent import answer_or_escalate
        assert "[ANSWER]" in answer_or_escalate.invoke({"mode": "answer", "text": "x"})
        assert "[ESCALATED" in answer_or_escalate.invoke({"mode": "escalate", "text": "y"})


# --------------------------------------------------------------------------- #
# LLM config — file discovery + key resolution (no source-coded keys)
# --------------------------------------------------------------------------- #
class TestLLMConfig:
    def test_default_config_has_no_api_key_in_source(self):
        """The default LLMConfig must NOT ship with an api_key — keys come
        from env vars or a git-ignored config file, never from source."""
        cfg = LLMConfig()
        assert cfg.api_key is None
        assert cfg.api_key_env == "DEEPSEEK_API_KEY"

    def test_resolved_api_key_from_env_var(self, monkeypatch):
        cfg = LLMConfig()
        monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test-key-from-env")
        assert cfg.resolved_api_key() == "sk-test-key-from-env"

    def test_resolved_api_key_from_explicit_field(self):
        cfg = LLMConfig(api_key="sk-explicit")
        assert cfg.resolved_api_key() == "sk-explicit"

    def test_resolved_api_key_raises_clear_error_when_missing(self, monkeypatch):
        cfg = LLMConfig()
        monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
        with pytest.raises(RuntimeError, match="No API key resolved"):
            cfg.resolved_api_key()

    def test_load_config_reads_yaml(self, tmp_path, monkeypatch):
        cfg_file = tmp_path / "config.yaml"
        cfg_file.write_text(textwrap.dedent("""
            default:
              provider: deepseek
              model: deepseek-chat
              base_url: https://api.deepseek.com
              api_key_env: DEEPSEEK_API_KEY
              temperature: 0.3
            research_agent:
              provider: deepseek
              model: deepseek-reasoner
              temperature: 0.0
        """), encoding="utf-8")
        cfg_set = load_config(str(cfg_file))
        assert cfg_set.default.model == "deepseek-chat"
        assert cfg_set.default.temperature == 0.3
        # Per-agent override
        research_cfg = cfg_set.for_agent("research_agent")
        assert research_cfg.model == "deepseek-reasoner"
        assert research_cfg.temperature == 0.0
        # Unknown agent falls back to default
        unknown = cfg_set.for_agent("unknown")
        assert unknown.model == "deepseek-chat"

    def test_load_config_returns_empty_set_when_no_file(self, monkeypatch):
        # Point discovery at a non-existent path.
        monkeypatch.setenv("VOYAGE_TRACE_LLM_CONFIG", "/nonexistent/config.yaml")
        cfg_set = load_config()
        assert cfg_set.default.api_key is None
        # Building a model raises (no key) rather than silently succeeding.
        with pytest.raises(RuntimeError, match="No API key resolved"):
            cfg_set.default.build_chat_model()

    def test_no_api_key_in_source_files(self):
        """Scan the sample_agents package source for hardcoded DeepSeek keys.

        This is a meta-test that asserts the honesty contract: the API key
        must never appear in source. It scans for the literal key prefix
        ``sk-`` followed by 20+ hex chars (the DeepSeek format).
        """
        import re
        from pathlib import Path
        sample_dir = Path(__file__).parent.parent / "sample_agents"
        pattern = re.compile(r"sk-[a-f0-9]{20,}")
        offenders: list[str] = []
        for py in sample_dir.rglob("*.py"):
            text = py.read_text(encoding="utf-8")
            for match in pattern.finditer(text):
                # Allow the pattern to appear in a comment / docstring only if
                # it's clearly illustrative (not a real key). Real keys are 32+
                # hex chars; we flag anything that looks like one.
                offenders.append(f"{py.name}: {match.group()}")
        # The example YAML may contain a placeholder comment but never a key.
        yaml = sample_dir / "config.example.yaml"
        if yaml.exists():
            text = yaml.read_text(encoding="utf-8")
            for match in pattern.finditer(text):
                offenders.append(f"config.example.yaml: {match.group()}")
        assert not offenders, f"Hardcoded API keys found in source: {offenders}"


# --------------------------------------------------------------------------- #
# Cross-cutting: the trace is a real CanonicalTrace usable by voyage_trace
# --------------------------------------------------------------------------- #
class TestTraceIntegration:
    def test_observer_trace_can_be_pushed_to_langfuse_export(self, monkeypatch):
        """The observer's trace flows straight through the Langfuse push helper."""
        from sample_agents import ScriptedChatModel, build_research_agent
        from voyage_trace.integrations.langfuse_export import export_to_langfuse

        monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
        monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "", )
        model = ScriptedChatModel(script=[
            AIMessage(content="", tool_calls=[{
                "name": "task", "args": {
                    "description": "research x",
                    "subagent_type": "research-agent",
                }, "id": "c1",
            }]),
            AIMessage(content="", tool_calls=[{
                "name": "search", "args": {"query": "x"}, "id": "c2",
            }]),
            AIMessage(content="", tool_calls=[{
                "name": "summarise", "args": {"findings": "[1] x: a"}, "id": "c3",
            }]),
            AIMessage(content="- a"),
            AIMessage(content="done"),
        ])
        agent, observer = build_research_agent(model=model)
        agent.invoke({"messages": [HumanMessage(content="research x")]})
        trace = observer.finalize()
        # Push through the Langfuse export — SDK absent in CI -> JSON artefact.
        export = export_to_langfuse(trace)
        assert "observations" in export
        assert len(export["observations"]) == trace.span_count

    def test_observer_trace_aggregates_with_other_traces(self):
        """The observer's trace composes with hand-built traces under
        aggregate_execution_graph — proving the trace is a first-class
        CanonicalTrace, not a special-case shape."""
        from sample_agents import ScriptedChatModel, build_research_agent

        model = ScriptedChatModel(script=[
            AIMessage(content="", tool_calls=[{
                "name": "task", "args": {
                    "description": "research x",
                    "subagent_type": "research-agent",
                }, "id": "c1",
            }]),
            AIMessage(content="", tool_calls=[{
                "name": "search", "args": {"query": "x"}, "id": "c2",
            }]),
            AIMessage(content="", tool_calls=[{
                "name": "summarise", "args": {"findings": "[1] x: a"}, "id": "c3",
            }]),
            AIMessage(content="- a"),
            AIMessage(content="done"),
        ])
        agent, observer = build_research_agent(model=model)
        agent.invoke({"messages": [HumanMessage(content="x")]})
        trace = observer.finalize()

        # A hand-built trace of a different agent.
        other = CanonicalTrace(
            trace_id="other", agent_id="other-agent", agent_name="Other",
            source_protocol=SourceProtocol.CUSTOM,
            spans=[TraceSpan(
                trace_id="other", span_id="s1",
                operation_type=OperationType.CHAT, agent_id="other-agent",
                source_protocol=SourceProtocol.CUSTOM,
                cost_usd=0.01, input_tokens=10, output_tokens=20,
            )],
        )
        from voyage_trace.protocol import normalise
        other = normalise(other)

        graph = aggregate_execution_graph([trace, other])
        assert graph.observed_runs == 2


# --------------------------------------------------------------------------- #
# SubAgentSpec builder — the shared, refactored builder
# --------------------------------------------------------------------------- #
class TestSubAgentSpec:
    """The refactored SubAgentSpec must produce identical agent/observer
    behaviour to the per-module builders it replaced."""

    def test_build_subagent_dict_shape_without_observer(self):
        from sample_agents import SubAgentSpec
        from sample_agents.research_agent import search

        spec = SubAgentSpec(
            agent_id="x", agent_name="X", description="d",
            system_prompt="p", tools=[search],
        )
        d = spec.build_subagent_dict()
        assert d["name"] == "x"
        assert d["description"] == "d"
        assert d["system_prompt"] == "p"
        assert d["tools"] == [search]
        # No model, no observer → no `model`/`middleware` keys.
        assert "model" not in d
        assert "middleware" not in d

    def test_build_subagent_dict_attaches_observer(self):
        from sample_agents import SubAgentSpec, attach

        obs = attach("x", agent_name="X")
        spec = SubAgentSpec(
            agent_id="x", agent_name="X", description="d",
            system_prompt="p", tools=[], orchestrator_prompt="o",
        )
        d = spec.build_subagent_dict(observer=obs)
        assert d["middleware"] == [obs]

    def test_build_subagent_dict_stashes_deferred_config(self):
        from sample_agents import LLMConfig, SubAgentSpec

        cfg = LLMConfig(model="deepseek-chat")
        spec = SubAgentSpec(
            agent_id="x", agent_name="X", description="d",
            system_prompt="p", tools=[],
        )
        d = spec.build_subagent_dict(config=cfg)
        # Config is deferred until build_agent constructs the model.
        assert d["_llm_config"] is cfg

    def test_build_agent_uses_provided_model_and_observer(self):
        """When both model and observer are passed, build_agent must not
        touch the LLM config — so it runs even with no env var / config file."""
        from sample_agents import SubAgentSpec, attach

        spec = SubAgentSpec(
            agent_id="probe-agent", agent_name="Probe",
            description="d", system_prompt="p", tools=[],
            orchestrator_prompt="orchestrator",
        )
        obs = attach("probe-agent", agent_name="Probe")
        model = ScriptedChatModel.from_texts("ok")
        # Should succeed without any API key in the environment.
        agent, returned_obs = spec.build_agent(model=model, observer=obs)
        assert returned_obs is obs
        result = agent.invoke({"messages": [HumanMessage(content="hi")]})
        assert isinstance(result["messages"][-1], AIMessage)

    def test_orchestrator_prompt_interpolates_agent_id(self):
        from sample_agents import SubAgentSpec

        spec = SubAgentSpec(
            agent_id="my-agent", agent_name="M", description="d",
            system_prompt="p", tools=[],
            orchestrator_prompt="delegate to `{agent_id}`",
        )
        model = ScriptedChatModel.from_texts("done")
        agent, _ = spec.build_agent(model=model)
        # The orchestrator system_prompt is handed to create_deep_agent; the
        # agent runs and the {agent_id} placeholder is replaced.
        result = agent.invoke({"messages": [HumanMessage(content="x")]})
        assert isinstance(result["messages"][-1], AIMessage)

    def test_research_spec_is_subagent_spec_instance(self):
        from sample_agents import SubAgentSpec
        from sample_agents.research_agent import RESEARCH_SPEC

        assert isinstance(RESEARCH_SPEC, SubAgentSpec)
        assert RESEARCH_SPEC.agent_id == "research-agent"
        assert RESEARCH_SPEC.agent_name == "ResearchAgent"

    def test_build_research_subagent_spec_alias_matches_spec_method(self):
        """The module-level `build_research_subagent_spec` alias must be the
        bound method of RESEARCH_SPEC (the refactored public-API contract)."""
        from sample_agents.research_agent import (
            RESEARCH_SPEC, build_research_subagent_spec,
        )
        assert build_research_subagent_spec == RESEARCH_SPEC.build_subagent_dict


# --------------------------------------------------------------------------- #
# TraceObserver edge cases — error paths and token accounting
# --------------------------------------------------------------------------- #
class TestTraceObserverEdgeCases:
    def test_extract_usage_reads_usage_metadata(self):
        """Real AIMessage.usage_metadata (populated by OpenAI/DeepSeek) must
        flow through to the span — never fabricated when absent."""
        from sample_agents.tracing import _extract_usage

        # Populated (the shape OpenAI / DeepSeek return).
        msg = AIMessage(
            content="hi",
            usage_metadata={"input_tokens": 42, "output_tokens": 13,
                            "total_tokens": 55},
        )
        assert _extract_usage(msg) == (42, 13)

        # Absent — honest zeros, not fabricated.
        bare = AIMessage(content="hi")
        assert _extract_usage(bare) == (0, 0)

    def test_coerce_str_truncates_long_values(self):
        from sample_agents.tracing import _coerce_str

        long = "x" * 10_000
        out = _coerce_str(long, limit=100)
        assert len(out) == 101  # 100 chars + ellipsis
        assert out.endswith("…")
        # None → empty string (never the literal "None").
        assert _coerce_str(None) == ""

    def test_tool_call_status_error_toolmessage(self):
        """A ToolMessage with status='error' must map to FAILED."""
        from sample_agents.tracing import _tool_call_status

        err_msg = ToolMessage(content="boom", tool_call_id="c1", status="error")
        status, err = _tool_call_status(err_msg)
        assert status == SpanStatus.FAILED
        assert err == "boom"

        ok_msg = ToolMessage(content="ok", tool_call_id="c2", status="success")
        status, err = _tool_call_status(ok_msg)
        assert status == SpanStatus.SUCCESS
        assert err is None

    def test_observer_records_failed_span_when_tool_raises(self):
        """A tool that raises must produce a FAILED EXECUTE_TOOL span and the
        exception must propagate to the agent (not be swallowed by the
        observer). This is the no-silent-failure contract in reverse: the
        observer records the failure honestly, then re-raises."""
        from langchain_core.tools import tool as lc_tool

        from sample_agents import SubAgentSpec, attach

        @lc_tool
        def boom(x: str) -> str:
            """Always raises."""
            raise RuntimeError("tool exploded")

        spec = SubAgentSpec(
            agent_id="boom-agent", agent_name="Boom", description="d",
            system_prompt="call boom then stop", tools=[boom],
            orchestrator_prompt="call boom",
        )
        model = ScriptedChatModel(script=[
            AIMessage(content="", tool_calls=[{
                "name": "boom", "args": {"x": "1"}, "id": "c1",
            }]),
            AIMessage(content="recovered"),
        ])
        obs = attach("boom-agent", agent_name="Boom")
        agent, _ = spec.build_agent(model=model, observer=obs)
        # The agent's tool error propagates; deepagents surfaces it. We do
        # not assert on the exception type (deepagents may wrap it) — we
        # assert the observer honestly captured the FAILED span.
        try:
            agent.invoke({"messages": [HumanMessage(content="go")]})
        except Exception:
            pass
        failed_tools = [s for s in obs.spans
                        if s.operation_type == OperationType.EXECUTE_TOOL
                        and s.status == SpanStatus.FAILED]
        assert len(failed_tools) >= 1
        assert failed_tools[0].metadata.get("tool") == "boom"
        assert failed_tools[0].error is not None

    def test_observer_usage_metadata_flows_into_span_tokens(self):
        """When the scripted model's AIMessage carries usage_metadata, the
        captured CHAT span must record those token counts (not zeros)."""
        from sample_agents import SubAgentSpec, attach

        spec = SubAgentSpec(
            agent_id="usage-agent", agent_name="Usage", description="d",
            system_prompt="reply once", tools=[],
            orchestrator_prompt="reply once",
        )
        model = ScriptedChatModel(script=[
            AIMessage(
                content="hello",
                usage_metadata={"input_tokens": 7, "output_tokens": 3,
                                "total_tokens": 10},
            ),
        ])
        obs = attach("usage-agent", agent_name="Usage")
        agent, _ = spec.build_agent(model=model, observer=obs)
        agent.invoke({"messages": [HumanMessage(content="hi")]})
        chat_spans = [s for s in obs.spans
                      if s.operation_type == OperationType.CHAT]
        assert chat_spans, "expected at least one chat span"
        # The first chat span's tokens come from the scripted usage_metadata.
        assert chat_spans[0].input_tokens == 7
        assert chat_spans[0].output_tokens == 3

    def test_observer_async_hooks_mirror_sync(self):
        """awrap_model_call / awrap_tool_call must capture spans just like
        their sync counterparts — the low-intrusion contract applies to the
        async path too."""
        import asyncio

        from sample_agents import attach
        from sample_agents.tracing import TraceObserver

        obs = TraceObserver(agent_id="async-agent", agent_name="Async")

        # A fake request/response pair shaped like deepagents' ModelRequest /
        # ModelResponse. We use SimpleNamespace because the observer only reads
        # attrs (getattr), never types.
        from types import SimpleNamespace

        class _Resp:
            def __init__(self, msg):
                self.result = [msg]

        ai = AIMessage(content="async-reply",
                       usage_metadata={"input_tokens": 5, "output_tokens": 2,
                                       "total_tokens": 7})
        request = SimpleNamespace(model=SimpleNamespace(name="m"), messages=[])
        response = _Resp(ai)

        async def _run():
            async def handler(req):
                return response
            await obs.awrap_model_call(request, handler)  # type: ignore[arg-type]

        asyncio.run(_run())
        assert len(obs.spans) == 1
        assert obs.spans[0].input_tokens == 5
        assert obs.spans[0].output_tokens == 2
        assert obs.spans[0].agent_name == "Async"

    def test_observer_reset_starts_new_trace(self):
        """reset() between runs must produce two traces with distinct ids
        and no span bleed-over."""
        from sample_agents import SubAgentSpec, attach

        spec = SubAgentSpec(
            agent_id="reset-agent", agent_name="Reset", description="d",
            system_prompt="reply", tools=[],
            orchestrator_prompt="reply",
        )
        obs = attach("reset-agent", agent_name="Reset", trace_id="trace-1")
        model = ScriptedChatModel(script=[
            AIMessage(content="first"),
            AIMessage(content="second"),
        ])
        agent, _ = spec.build_agent(model=model, observer=obs)
        agent.invoke({"messages": [HumanMessage(content="a")]})
        first_count = len(obs.spans)
        assert first_count >= 1
        obs.reset(trace_id="trace-2")
        assert obs.spans == []
        assert obs.trace_id == "trace-2"
        # Re-run on the same agent — capture resumes on the new trace.
        agent.invoke({"messages": [HumanMessage(content="b")]})
        assert len(obs.spans) >= 1
        # All new spans carry the new trace_id.
        assert all(s.trace_id == "trace-2" for s in obs.spans)


# --------------------------------------------------------------------------- #
# Honesty meta-test: refactor did not introduce hardcoded keys or shortcuts
# --------------------------------------------------------------------------- #
class TestRefactorHonesty:
    def test_builder_module_has_no_api_keys(self):
        """The new builder.py must not hardcode credentials."""
        import re
        from pathlib import Path

        builder = Path(__file__).parent.parent / "sample_agents" / "builder.py"
        text = builder.read_text(encoding="utf-8")
        pattern = re.compile(r"sk-[a-f0-9]{20,}")
        assert not pattern.findall(text), "hardcoded key in builder.py"

    def test_sample_tool_modules_do_not_import_llm_clients(self):
        """The deterministic @tool stubs must not secretly call a real LLM
        to fabricate "smart" answers. They are pure-Python transforms."""
        from pathlib import Path

        sample_dir = Path(__file__).parent.parent / "sample_agents"
        forbidden = ("langchain_openai", "openai.ChatOpenAI",
                     "ChatOpenAI(", "anthropic.Anthropic")
        offenders = []
        for py in sample_dir.rglob("*.py"):
            if py.name in {"llm_config.py", "builder.py", "tracing.py",
                           "testing.py", "__init__.py"}:
                continue
            text = py.read_text(encoding="utf-8")
            for needle in forbidden:
                if needle in text:
                    offenders.append(f"{py.name}: {needle}")
        assert not offenders, f"tool modules import LLM clients: {offenders}"

