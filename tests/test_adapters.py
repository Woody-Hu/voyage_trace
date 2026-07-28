"""Tests for voyage_trace.adapters — source-protocol trace adapters.

Each adapter is tested with real exported-JSON payloads (the kind a backend
would actually produce), not mocks. The tests verify field mapping, protocol
invariant enforcement after adapt(), and the auto-inference logic.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest

from voyage_trace.adapters import adapt, _infer_protocol
from voyage_trace.adapters.base import AdapterError
from voyage_trace.types import OperationType, SourceProtocol, SpanStatus


# --------------------------------------------------------------------------- #
# LangSmith adapter
# --------------------------------------------------------------------------- #
class TestLangSmithAdapter:
    @pytest.fixture
    def langsmith_payload(self) -> dict:
        """A realistic LangSmith run export with a parent chain + child."""
        return {
            "runs": [
                {
                    "id": "run-root",
                    "trace_id": "ls-trace-001",
                    "run_type": "chain",
                    "name": "AgentRun",
                    "status": "success",
                    "start_time": "2025-07-27T12:00:00+00:00",
                    "end_time": "2025-07-27T12:00:05+00:00",
                    "inputs": {"input": "hello"},
                    "outputs": {"output": "world"},
                    "prompt_tokens": 10,
                    "completion_tokens": 20,
                    "total_cost": 0.001,
                    "session_id": "session-1",
                    "extra": {"metadata": {"agent_id": "agent-ls"}},
                    "dotted_order": "20250727T120000Z0000000000000001",
                },
                {
                    "id": "run-child",
                    "trace_id": "ls-trace-001",
                    "parent_run_id": "run-root",
                    "run_type": "llm",
                    "name": "LLM",
                    "status": "success",
                    "start_time": "2025-07-27T12:00:01+00:00",
                    "end_time": "2025-07-27T12:00:03+00:00",
                    "inputs": {"messages": []},
                    "outputs": {"content": "response"},
                    "prompt_tokens": 50,
                    "completion_tokens": 100,
                    "total_cost": 0.002,
                    "extra": {"metadata": {"agent_id": "agent-ls"}},
                    "dotted_order": "20250727T120000Z0000000000000001.20250727T120001Z0000000000000002",
                },
            ]
        }

    def test_adapts_to_canonical_trace(self, langsmith_payload):
        trace = adapt(langsmith_payload, source_protocol="langsmith")
        assert trace.trace_id == "ls-trace-001"
        assert trace.agent_id == "agent-ls"
        assert trace.source_protocol == SourceProtocol.LANGSMITH
        assert trace.span_count == 2

    def test_run_type_mapped_to_operation_type(self, langsmith_payload):
        trace = adapt(langsmith_payload, source_protocol="langsmith")
        ops = [s.operation_type for s in trace.spans]
        assert OperationType.INVOKE_AGENT in ops
        assert OperationType.CHAT in ops

    def test_tokens_and_cost_preserved(self, langsmith_payload):
        trace = adapt(langsmith_payload, source_protocol="langsmith")
        llm_span = next(s for s in trace.spans if s.span_id == "run-child")
        assert llm_span.input_tokens == 50
        assert llm_span.output_tokens == 100
        assert llm_span.cost_usd == 0.002

    def test_error_status_mapped_to_failed(self):
        payload = {
            "id": "r1",
            "trace_id": "t1",
            "run_type": "chain",
            "name": "Fail",
            "error": "something went wrong",
            "start_time": "2025-07-27T12:00:00+00:00",
            "end_time": "2025-07-27T12:00:01+00:00",
            "extra": {"metadata": {}},
        }
        trace = adapt(payload, source_protocol="langsmith")
        assert trace.spans[0].status == SpanStatus.FAILED
        assert trace.spans[0].error == "something went wrong"

    def test_empty_runs_rejected(self):
        with pytest.raises(AdapterError, match="no runs"):
            adapt({"runs": []}, source_protocol="langsmith")

    def test_accepts_json_string(self, langsmith_payload):
        trace = adapt(json.dumps(langsmith_payload), source_protocol="langsmith")
        assert trace.trace_id == "ls-trace-001"


# --------------------------------------------------------------------------- #
# Langfuse adapter
# --------------------------------------------------------------------------- #
class TestLangfuseAdapter:
    @pytest.fixture
    def langfuse_payload(self) -> dict:
        return {
            "trace": {
                "id": "lf-trace-001",
                "name": "MyAgent",
                "session_id": "sess-1",
                "user_id": "user-1",
                "metadata": {"agent_id": "agent-lf"},
            },
            "observations": [
                {
                    "id": "obs-root",
                    "trace_id": "lf-trace-001",
                    "type": "span",
                    "name": "agent_run",
                    "start_time": "2025-07-27T12:00:00+00:00",
                    "end_time": "2025-07-27T12:00:05+00:00",
                    "input": {"q": "hello"},
                    "output": {"a": "world"},
                    "level": "DEFAULT",
                    "metadata": {},
                    "usage": {"input": 5, "output": 10},
                    "calculated_total_cost": 0.01,
                },
                {
                    "id": "obs-child",
                    "trace_id": "lf-trace-001",
                    "parent_id": "obs-root",
                    "type": "generation",
                    "name": "llm_call",
                    "start_time": "2025-07-27T12:00:01+00:00",
                    "end_time": "2025-07-27T12:00:03+00:00",
                    "input": {"prompt": "hi"},
                    "output": {"text": "resp"},
                    "level": "DEFAULT",
                    "metadata": {},
                    "usage": {"input": 20, "output": 40},
                    "calculated_total_cost": 0.02,
                },
            ],
        }

    def test_adapts_to_canonical_trace(self, langfuse_payload):
        trace = adapt(langfuse_payload, source_protocol="langfuse")
        assert trace.trace_id == "lf-trace-001"
        assert trace.agent_id == "agent-lf"
        assert trace.source_protocol == SourceProtocol.LANGFUSE
        assert trace.span_count == 2

    def test_observation_type_mapped(self, langfuse_payload):
        trace = adapt(langfuse_payload, source_protocol="langfuse")
        root = next(s for s in trace.spans if s.span_id == "obs-root")
        child = next(s for s in trace.spans if s.span_id == "obs-child")
        assert root.operation_type == OperationType.INVOKE_AGENT
        assert child.operation_type == OperationType.CHAT

    def test_error_level_mapped_to_failed(self, langfuse_payload):
        langfuse_payload["observations"][0]["level"] = "ERROR"
        trace = adapt(langfuse_payload, source_protocol="langfuse")
        assert trace.spans[0].status == SpanStatus.FAILED

    def test_dotted_order_derived_from_parent_tree(self, langfuse_payload):
        trace = adapt(langfuse_payload, source_protocol="langfuse")
        root = next(s for s in trace.spans if s.parent_span_id is None)
        child = next(s for s in trace.spans if s.parent_span_id == "obs-root")
        assert child.dotted_order.startswith(root.dotted_order + ".")


# --------------------------------------------------------------------------- #
# OTel adapter
# --------------------------------------------------------------------------- #
class TestOTELAdapter:
    @pytest.fixture
    def otel_payload(self) -> list:
        return [
            {
                "trace_id": "otel-trace-001",
                "span_id": "span-aaa",
                "name": "chat_call",
                "start_time": "2025-07-27T12:00:00+00:00",
                "end_time": "2025-07-27T12:00:02+00:00",
                "attributes": {
                    "gen_ai.operation.name": "chat",
                    "gen_ai.agent.id": "agent-otel",
                    "gen_ai.agent.name": "OTelAgent",
                    "gen_ai.usage.input_tokens": 30,
                    "gen_ai.usage.output_tokens": 60,
                    "gen_ai.conversation.id": "conv-1",
                },
                "status": {"code": "OK"},
            },
            {
                "trace_id": "otel-trace-001",
                "span_id": "span-bbb",
                "parent_span_id": "span-aaa",
                "name": "tool_call",
                "start_time": "2025-07-27T12:00:01+00:00",
                "end_time": "2025-07-27T12:00:02+00:00",
                "attributes": {
                    "gen_ai.operation.name": "execute_tool",
                    "gen_ai.usage.input_tokens": 5,
                    "gen_ai.usage.output_tokens": 10,
                },
                "status": {"code": "ERROR", "message": "tool failed"},
            },
        ]

    def test_adapts_to_canonical_trace(self, otel_payload):
        trace = adapt(otel_payload, source_protocol="otel")
        assert trace.trace_id == "otel-trace-001"
        assert trace.agent_id == "agent-otel"
        assert trace.source_protocol == SourceProtocol.OTEL
        assert trace.span_count == 2

    def test_operation_name_mapped(self, otel_payload):
        trace = adapt(otel_payload, source_protocol="otel")
        ops = {s.operation_type for s in trace.spans}
        assert OperationType.CHAT in ops
        assert OperationType.EXECUTE_TOOL in ops

    def test_error_status_mapped(self, otel_payload):
        trace = adapt(otel_payload, source_protocol="otel")
        failed = next(s for s in trace.spans if s.span_id == "span-bbb")
        assert failed.status == SpanStatus.FAILED
        assert failed.error == "tool failed"

    def test_unix_nano_timestamps_parsed(self):
        payload = [
            {
                "trace_id": "t1",
                "span_id": "s1",
                "name": "test",
                "start_time_unix_nano": 1753622400000000000,
                "end_time_unix_nano": 1753622402000000000,
                "attributes": {"gen_ai.operation.name": "chat"},
                "status": {"code": "OK"},
            }
        ]
        trace = adapt(payload, source_protocol="otel")
        span = trace.spans[0]
        assert span.start_time is not None
        assert span.end_time is not None
        assert span.duration_seconds == 2.0

    def test_otlp_resource_spans_tree(self):
        payload = {
            "resourceSpans": [
                {
                    "scopeSpans": [
                        {
                            "spans": [
                                {
                                    "trace_id": "t1",
                                    "span_id": "s1",
                                    "name": "test",
                                    "start_time": "2025-07-27T12:00:00+00:00",
                                    "end_time": "2025-07-27T12:00:01+00:00",
                                    "attributes": {"gen_ai.operation.name": "chat"},
                                    "status": {"code": "OK"},
                                }
                            ]
                        }
                    ]
                }
            ]
        }
        trace = adapt(payload, source_protocol="otel")
        assert trace.span_count == 1


# --------------------------------------------------------------------------- #
# A2A adapter
# --------------------------------------------------------------------------- #
class TestA2AAdapter:
    @pytest.fixture
    def a2a_payload(self) -> dict:
        return {
            "id": "a2a-task-001",
            "contextId": "ctx-1",
            "status": {"state": "completed", "timestamp": "2025-07-27T12:00:03+00:00"},
            "history": [
                {
                    "state": "submitted",
                    "timestamp": "2025-07-27T12:00:00+00:00",
                    "message": {"role": "user", "content": "do task"},
                },
                {
                    "state": "working",
                    "timestamp": "2025-07-27T12:00:01+00:00",
                    "message": {"role": "agent", "content": "working..."},
                },
                {
                    "state": "completed",
                    "timestamp": "2025-07-27T12:00:03+00:00",
                    "message": {"role": "agent", "content": "done"},
                },
            ],
            "metadata": {"agent_id": "agent-a2a"},
        }

    def test_adapts_to_canonical_trace(self, a2a_payload):
        trace = adapt(a2a_payload, source_protocol="a2a")
        assert trace.trace_id == "a2a-task-001"
        assert trace.agent_id == "agent-a2a"
        assert trace.source_protocol == SourceProtocol.A2A
        # 3 history entries + status dict appended = 4 spans
        assert trace.span_count == 4

    def test_spans_chained_parent_to_child(self, a2a_payload):
        trace = adapt(a2a_payload, source_protocol="a2a")
        spans = trace.sorted_spans()
        for i in range(1, len(spans)):
            assert spans[i].parent_span_id == spans[i - 1].span_id

    def test_state_mapped_to_status(self, a2a_payload):
        trace = adapt(a2a_payload, source_protocol="a2a")
        statuses = {s.status for s in trace.spans}
        assert SpanStatus.SUBMITTED in statuses
        assert SpanStatus.WORKING in statuses
        assert SpanStatus.SUCCESS in statuses

    def test_user_message_becomes_inputs(self, a2a_payload):
        trace = adapt(a2a_payload, source_protocol="a2a")
        first = trace.sorted_spans()[0]
        assert "message" in first.inputs
        assert first.inputs["message"]["role"] == "user"

    def test_agent_message_becomes_outputs(self, a2a_payload):
        trace = adapt(a2a_payload, source_protocol="a2a")
        spans = trace.sorted_spans()
        working_span = next(s for s in spans if s.status == SpanStatus.WORKING)
        assert working_span.outputs is not None
        assert "message" in working_span.outputs


# --------------------------------------------------------------------------- #
# MCP adapter
# --------------------------------------------------------------------------- #
class TestMCPAdapter:
    @pytest.fixture
    def mcp_jsonrpc_payload(self) -> dict:
        return {
            "trace_id": "mcp-trace-001",
            "messages": [
                {
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "tools/call",
                    "params": {
                        "name": "search",
                        "arguments": {"query": "test"},
                        "_meta": {
                            "trace_id": "mcp-trace-001",
                            "server_name": "search-server",
                            "start_time": "2025-07-27T12:00:00+00:00",
                        },
                    },
                },
                {
                    "jsonrpc": "2.0",
                    "id": 1,
                    "result": {"content": "search results"},
                    "_meta": {"end_time": "2025-07-27T12:00:01+00:00"},
                },
            ],
        }

    def test_adapts_jsonrpc_to_canonical_trace(self, mcp_jsonrpc_payload):
        trace = adapt(mcp_jsonrpc_payload, source_protocol="mcp")
        assert trace.trace_id == "mcp-trace-001"
        assert trace.agent_id == "search-server"
        assert trace.source_protocol == SourceProtocol.MCP
        assert trace.span_count == 1

    def test_method_mapped_to_operation_type(self, mcp_jsonrpc_payload):
        trace = adapt(mcp_jsonrpc_payload, source_protocol="mcp")
        assert trace.spans[0].operation_type == OperationType.EXECUTE_TOOL

    def test_response_result_becomes_outputs(self, mcp_jsonrpc_payload):
        trace = adapt(mcp_jsonrpc_payload, source_protocol="mcp")
        assert trace.spans[0].outputs == {"content": "search results"}

    def test_error_response_sets_failed_status(self, mcp_jsonrpc_payload):
        mcp_jsonrpc_payload["messages"][1] = {
            "jsonrpc": "2.0",
            "id": 1,
            "error": {"code": -1, "message": "tool error"},
        }
        trace = adapt(mcp_jsonrpc_payload, source_protocol="mcp")
        assert trace.spans[0].status == SpanStatus.FAILED

    def test_resources_method_mapped_to_retrieval(self):
        payload = {
            "trace_id": "mcp-t2",
            "messages": [
                {
                    "jsonrpc": "2.0",
                    "id": 2,
                    "method": "resources/read",
                    "params": {"_meta": {"trace_id": "mcp-t2"}},
                },
                {
                    "jsonrpc": "2.0",
                    "id": 2,
                    "result": {"contents": []},
                },
            ],
        }
        trace = adapt(payload, source_protocol="mcp")
        assert trace.spans[0].operation_type == OperationType.RETRIEVAL


# --------------------------------------------------------------------------- #
# Raw adapter
# --------------------------------------------------------------------------- #
class TestRawAdapter:
    def test_canonical_trace_passthrough(self, simple_trace):
        trace = adapt(simple_trace, source_protocol="custom")
        assert trace.trace_id == simple_trace.trace_id
        assert trace.span_count == simple_trace.span_count

    def test_canonical_dict_parsed(self):
        payload = {
            "trace_id": "raw-001",
            "agent_id": "agent-raw",
            "spans": [
                {
                    "trace_id": "raw-001",
                    "span_id": "s1",
                    "operation_type": "chat",
                    "start_time": "2025-07-27T12:00:00+00:00",
                    "end_time": "2025-07-27T12:00:01+00:00",
                }
            ],
        }
        trace = adapt(payload, source_protocol="custom")
        assert trace.trace_id == "raw-001"
        assert trace.span_count == 1

    def test_single_span_dict_wrapped(self):
        payload = {
            "trace_id": "raw-002",
            "span_id": "s1",
            "id": "s1",
            "operation_type": "chat",
        }
        trace = adapt(payload, source_protocol="custom")
        assert trace.span_count == 1
        assert trace.spans[0].span_id == "s1"

    def test_id_aliased_to_span_id(self):
        payload = {
            "trace_id": "raw-003",
            "id": "my-span",
            "parent_id": None,
        }
        trace = adapt(payload, source_protocol="custom")
        assert trace.spans[0].span_id == "my-span"

    def test_list_of_spans_wrapped(self):
        payload = [
            {"trace_id": "raw-004", "span_id": "s1"},
            {"trace_id": "raw-004", "span_id": "s2", "parent_span_id": "s1"},
        ]
        trace = adapt(payload, source_protocol="custom")
        assert trace.span_count == 2

    def test_empty_list_rejected(self):
        with pytest.raises(AdapterError, match="empty"):
            adapt([], source_protocol="custom")


# --------------------------------------------------------------------------- #
# Protocol inference
# --------------------------------------------------------------------------- #
class TestProtocolInference:
    def test_langsmith_inferred_from_run_type(self):
        assert _infer_protocol({"run_type": "chain"}) == SourceProtocol.LANGSMITH

    def test_langfuse_inferred_from_observations(self):
        assert _infer_protocol({"observations": []}) == SourceProtocol.LANGFUSE

    def test_otel_inferred_from_resource_spans(self):
        assert _infer_protocol({"resourceSpans": []}) == SourceProtocol.OTEL

    def test_a2a_inferred_from_history(self):
        assert _infer_protocol({"history": [], "contextId": "x"}) == SourceProtocol.A2A

    def test_mcp_inferred_from_jsonrpc(self):
        assert _infer_protocol({"jsonrpc": "2.0", "method": "test"}) == SourceProtocol.MCP

    def test_custom_fallback(self):
        assert _infer_protocol({"unknown": "shape"}) == SourceProtocol.CUSTOM

    def test_list_with_run_type_infers_langsmith(self):
        assert _infer_protocol([{"run_type": "llm"}]) == SourceProtocol.LANGSMITH

    def test_list_with_attributes_infers_otel(self):
        assert _infer_protocol([{"attributes": {}}]) == SourceProtocol.OTEL

    def test_json_string_decoded_before_inference(self):
        payload = json.dumps({"resourceSpans": []})
        assert _infer_protocol(payload) == SourceProtocol.OTEL

    def test_adapt_without_protocol_uses_inference(self):
        payload = {
            "trace_id": "auto-001",
            "agent_id": "a1",
            "spans": [{"trace_id": "auto-001", "span_id": "s1"}],
        }
        trace = adapt(payload)
        assert trace.source_protocol == SourceProtocol.CUSTOM
