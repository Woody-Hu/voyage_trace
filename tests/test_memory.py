"""Tests for voyage_trace.memory — four-partition memory system.

Tests use a real InMemoryStorage backend (not a mock) to verify:
- Episodic: remember/recall/search/recall_similar
- Semantic: confidence threshold filtering
- Procedural: version auto-increment + latest()
- Working: snapshot/clear lifecycle
- PartitionedMemory: mount/unmount, cross-round recall, async context manager
"""

from __future__ import annotations

import pytest

from voyage_trace.memory import (
    EpisodicMemory,
    MemoryScope,
    PartitionedMemory,
    ProceduralMemory,
    SemanticMemory,
    WorkingMemory,
)
from voyage_trace.storage import InMemoryStorage


# --------------------------------------------------------------------------- #
# Episodic memory
# --------------------------------------------------------------------------- #
class TestEpisodicMemory:
    async def test_remember_and_recall(self, storage: InMemoryStorage):
        em = EpisodicMemory(storage)
        scope = MemoryScope(target_agent_id="agent-A", round_id="r1")
        record = {
            "trace_id": "t1",
            "agent_id": "agent-A",
            "failure_signature": "loop:web_search",
            "findings": [{"type": "loop", "severity": "high"}],
            "outcome": "capped at 3 iterations",
        }
        await em.remember(scope, "ep-1", record)
        result = await em.recall(scope, "ep-1")
        assert result is not None
        assert result["failure_signature"] == "loop:web_search"
        assert result["outcome"] == "capped at 3 iterations"

    async def test_recall_missing_returns_none(self, storage: InMemoryStorage):
        em = EpisodicMemory(storage)
        scope = MemoryScope(target_agent_id="agent-A", round_id="r1")
        assert await em.recall(scope, "nonexistent") is None

    async def test_search_by_failure_signature(self, storage: InMemoryStorage):
        em = EpisodicMemory(storage)
        scope = MemoryScope(target_agent_id="agent-A", round_id="r1")
        for i in range(3):
            await em.remember(scope, f"ep-{i}", {
                "trace_id": f"t{i}",
                "agent_id": "agent-A",
                "failure_signature": "loop:web_search",
                "outcome": f"fix-{i}",
            })
        results = await em.search(scope, {"failure_signature": "loop:web_search"})
        assert len(results) == 3

    async def test_recall_similar_across_rounds(self, storage: InMemoryStorage):
        em = EpisodicMemory(storage)
        # Write to two different rounds.
        for round_id in ("r1", "r2", "r3"):
            scope = MemoryScope(target_agent_id="agent-A", round_id=round_id)
            await em.remember(scope, f"ep-{round_id}", {
                "trace_id": f"t-{round_id}",
                "agent_id": "agent-A",
                "failure_signature": "loop:web_search",
                "outcome": f"fix-{round_id}",
            })
        # Cross-round recall.
        wildcard_scope = MemoryScope(target_agent_id="agent-A", round_id="*")
        results = await em.recall_similar(wildcard_scope, "loop:web_search")
        assert len(results) == 3

    async def test_forget(self, storage: InMemoryStorage):
        em = EpisodicMemory(storage)
        scope = MemoryScope(target_agent_id="agent-A", round_id="r1")
        await em.remember(scope, "ep-1", {"trace_id": "t1", "failure_signature": "f1"})
        assert await em.forget(scope, "ep-1") is True
        assert await em.recall(scope, "ep-1") is None


# --------------------------------------------------------------------------- #
# Semantic memory
# --------------------------------------------------------------------------- #
class TestSemanticMemory:
    async def test_remember_and_search(self, storage: InMemoryStorage):
        sm = SemanticMemory(storage)
        scope = MemoryScope(target_agent_id="*", round_id="global")
        await sm.remember(scope, "rule-1", {
            "rule_id": "rule-1",
            "rule_text": "agents that call web_search >3x are looping",
            "confidence": 0.85,
        })
        results = await sm.search(scope, {"rule_id": "rule-1"})
        assert len(results) == 1
        assert results[0]["confidence"] == 0.85

    async def test_confidence_min_filter(self, storage: InMemoryStorage):
        sm = SemanticMemory(storage)
        scope = MemoryScope(target_agent_id="*", round_id="global")
        for i, conf in enumerate([0.3, 0.6, 0.9]):
            await sm.remember(scope, f"rule-{i}", {
                "rule_id": f"rule-{i}",
                "confidence": conf,
            })
        results = await sm.search(scope, {"confidence_min": 0.5})
        assert len(results) == 2
        for r in results:
            assert r["confidence"] >= 0.5

    async def test_confidence_max_filter(self, storage: InMemoryStorage):
        sm = SemanticMemory(storage)
        scope = MemoryScope(target_agent_id="*", round_id="global")
        for i, conf in enumerate([0.3, 0.6, 0.9]):
            await sm.remember(scope, f"rule-{i}", {
                "rule_id": f"rule-{i}",
                "confidence": conf,
            })
        results = await sm.search(scope, {"confidence_max": 0.7})
        assert len(results) == 2
        for r in results:
            assert r["confidence"] <= 0.7

    async def test_cross_agent_search_with_wildcard(self, storage: InMemoryStorage):
        sm = SemanticMemory(storage)
        for agent in ("agent-A", "agent-B"):
            scope = MemoryScope(target_agent_id=agent, round_id="r1")
            await sm.remember(scope, f"rule-{agent}", {
                "rule_id": f"rule-{agent}",
                "confidence": 0.8,
            })
        wildcard = MemoryScope(target_agent_id="*", round_id="*")
        results = await sm.search(wildcard, {})
        assert len(results) == 2


# --------------------------------------------------------------------------- #
# Procedural memory
# --------------------------------------------------------------------------- #
class TestProceduralMemory:
    async def test_auto_increment_version(self, storage: InMemoryStorage):
        pm = ProceduralMemory(storage)
        scope = MemoryScope(target_agent_id="agent-A", round_id="r1")
        await pm.remember(scope, "guardrail-1", {
            "template_id": "g1",
            "kind": "guardrail",
            "content": "v1 content",
        })
        await pm.remember(scope, "guardrail-1", {
            "template_id": "g1",
            "kind": "guardrail",
            "content": "v2 content",
        })
        latest = await pm.latest(scope, "guardrail-1")
        assert latest is not None
        assert latest["version"] == 2
        assert latest["content"] == "v2 content"

    async def test_latest_returns_none_when_empty(self, storage: InMemoryStorage):
        pm = ProceduralMemory(storage)
        scope = MemoryScope(target_agent_id="agent-A", round_id="r1")
        assert await pm.latest(scope, "nonexistent") is None

    async def test_old_version_preserved(self, storage: InMemoryStorage):
        pm = ProceduralMemory(storage)
        scope = MemoryScope(target_agent_id="agent-A", round_id="r1")
        await pm.remember(scope, "fix-1", {"content": "v1"})
        await pm.remember(scope, "fix-1", {"content": "v2"})
        v1 = await pm.recall(scope, "fix-1#v1")
        assert v1 is not None
        assert v1["content"] == "v1"
        v2 = await pm.recall(scope, "fix-1#v2")
        assert v2 is not None
        assert v2["content"] == "v2"

    async def test_explicit_version_not_overwritten(self, storage: InMemoryStorage):
        pm = ProceduralMemory(storage)
        scope = MemoryScope(target_agent_id="agent-A", round_id="r1")
        await pm.remember(scope, "tmpl", {"content": "explicit", "version": 5})
        latest = await pm.latest(scope, "tmpl")
        assert latest is not None
        assert latest["version"] == 5

    async def test_search_by_kind(self, storage: InMemoryStorage):
        pm = ProceduralMemory(storage)
        scope = MemoryScope(target_agent_id="agent-A", round_id="r1")
        await pm.remember(scope, "g1", {"kind": "guardrail", "content": "g"})
        await pm.remember(scope, "p1", {"kind": "prompt", "content": "p"})
        results = await pm.search(scope, {"kind": "guardrail"})
        assert len(results) == 1
        assert results[0]["content"] == "g"


# --------------------------------------------------------------------------- #
# Working memory
# --------------------------------------------------------------------------- #
class TestWorkingMemory:
    async def test_remember_and_recall(self, storage: InMemoryStorage):
        wm = WorkingMemory(storage)
        scope = MemoryScope(target_agent_id="agent-A", round_id="r1")
        await wm.remember(scope, "item-1", {
            "item_id": "i1",
            "kind": "trace",
            "payload": {"data": 1},
        })
        result = await wm.recall(scope, "item-1")
        assert result is not None
        assert result["payload"]["data"] == 1

    async def test_snapshot_returns_all_items(self, storage: InMemoryStorage):
        wm = WorkingMemory(storage)
        scope = MemoryScope(target_agent_id="agent-A", round_id="r1")
        for i in range(3):
            await wm.remember(scope, f"item-{i}", {
                "item_id": f"i{i}",
                "kind": "finding",
                "payload": {"n": i},
            })
        snapshot = await wm.snapshot(scope)
        assert len(snapshot["items"]) == 3
        assert snapshot["target_agent_id"] == "agent-A"
        assert snapshot["round_id"] == "r1"

    async def test_clear_removes_all_items(self, storage: InMemoryStorage):
        wm = WorkingMemory(storage)
        scope = MemoryScope(target_agent_id="agent-A", round_id="r1")
        for i in range(3):
            await wm.remember(scope, f"item-{i}", {"item_id": f"i{i}", "kind": "trace"})
        count = await wm.clear(scope)
        assert count == 3
        snapshot = await wm.snapshot(scope)
        assert len(snapshot["items"]) == 0

    async def test_isolation_between_scopes(self, storage: InMemoryStorage):
        wm = WorkingMemory(storage)
        scope_a = MemoryScope(target_agent_id="agent-A", round_id="r1")
        scope_b = MemoryScope(target_agent_id="agent-B", round_id="r1")
        await wm.remember(scope_a, "item", {"item_id": "a", "kind": "trace"})
        await wm.remember(scope_b, "item", {"item_id": "b", "kind": "trace"})
        a = await wm.recall(scope_a, "item")
        b = await wm.recall(scope_b, "item")
        assert a["item_id"] == "a"
        assert b["item_id"] == "b"


# --------------------------------------------------------------------------- #
# PartitionedMemory manager
# --------------------------------------------------------------------------- #
class TestPartitionedMemory:
    async def test_mount_and_current(self, storage: InMemoryStorage):
        pm = PartitionedMemory(storage)
        assert pm.current() is None
        scope = await pm.mount("agent-A", "r1")
        assert pm.current() is not None
        assert pm.current().target_agent_id == "agent-A"
        assert pm.current().round_id == "r1"

    async def test_unmount_clears_working_memory(self, storage: InMemoryStorage):
        pm = PartitionedMemory(storage)
        scope = await pm.mount("agent-A", "r1")
        await pm.working().remember(scope, "item", {"item_id": "i1", "kind": "trace"})
        popped = await pm.unmount()
        assert popped is not None
        assert pm.current() is None
        # Working memory should be cleared.
        result = await pm.working().recall(scope, "item")
        assert result is None

    async def test_nested_scopes(self, storage: InMemoryStorage):
        pm = PartitionedMemory(storage)
        s1 = await pm.mount("agent-A", "r1")
        s2 = await pm.mount("agent-A", "r2")
        assert pm.current().round_id == "r2"
        await pm.unmount()
        assert pm.current().round_id == "r1"
        await pm.unmount()
        assert pm.current() is None

    async def test_partition_accessors(self, storage: InMemoryStorage):
        pm = PartitionedMemory(storage)
        assert isinstance(pm.episodic(), EpisodicMemory)
        assert isinstance(pm.semantic(), SemanticMemory)
        assert isinstance(pm.procedural(), ProceduralMemory)
        assert isinstance(pm.working(), WorkingMemory)

    async def test_recall_cross_round(self, storage: InMemoryStorage):
        pm = PartitionedMemory(storage)
        for round_id in ("r1", "r2"):
            scope = await pm.mount("agent-A", round_id)
            await pm.episodic().remember(scope, f"ep-{round_id}", {
                "trace_id": f"t-{round_id}",
                "agent_id": "agent-A",
                "failure_signature": "loop:tool",
                "outcome": f"fix-{round_id}",
            })
            await pm.unmount()
        results = await pm.recall_cross_round("agent-A", "loop:tool")
        assert len(results) == 2

    async def test_async_context_manager(self, storage: InMemoryStorage):
        pm = PartitionedMemory(storage)
        async with pm.use("agent-A", "r1"):
            assert pm.current() is not None
            assert pm.current().target_agent_id == "agent-A"
            await pm.working().remember(pm.current(), "item", {"item_id": "i1", "kind": "trace"})
        assert pm.current() is None

    async def test_sync_context_manager(self, storage: InMemoryStorage):
        pm = PartitionedMemory(storage)
        with pm.use("agent-A", "r1"):
            assert pm.current() is not None
        # Sync exit only pops scope, doesn't clear working memory.
        assert pm.current() is None

    async def test_unmount_empty_returns_none(self, storage: InMemoryStorage):
        pm = PartitionedMemory(storage)
        assert await pm.unmount() is None
