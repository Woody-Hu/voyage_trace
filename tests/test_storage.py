"""Tests for voyage_trace.storage — InMemoryStorage real async operations.

InMemoryStorage is a genuine backend (not a mock): it has real concurrency
semantics (asyncio.Lock), real upsert behaviour, and real metadata query.
These tests exercise every method of the WorkspaceStorage ABC.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone

import pytest

from voyage_trace.storage import InMemoryStorage, StorageRecord
from voyage_trace.storage.base import WorkspaceStorage


class TestInMemoryStorageBasics:
    async def test_put_and_get(self, storage: InMemoryStorage):
        rec = await storage.put("traces", "t1", b'{"data": 1}')
        assert rec.namespace == "traces"
        assert rec.key == "t1"
        assert rec.value == b'{"data": 1}'
        assert rec.created_at is not None
        assert rec.updated_at is not None

        fetched = await storage.get("traces", "t1")
        assert fetched is not None
        assert fetched.value == b'{"data": 1}'

    async def test_get_missing_returns_none(self, storage: InMemoryStorage):
        assert await storage.get("traces", "nonexistent") is None

    async def test_put_upsert_preserves_created_at(self, storage: InMemoryStorage):
        rec1 = await storage.put("ns", "k1", b"v1")
        await asyncio.sleep(0.001)  # ensure timestamp differs
        rec2 = await storage.put("ns", "k1", b"v2")
        assert rec2.created_at == rec1.created_at
        assert rec2.updated_at > rec1.updated_at
        assert rec2.value == b"v2"

    async def test_put_with_metadata(self, storage: InMemoryStorage):
        await storage.put("ns", "k1", b"v", metadata={"agent_id": "a1", "source": "otel"})
        rec = await storage.get("ns", "k1")
        assert rec is not None
        assert rec.metadata["agent_id"] == "a1"
        assert rec.metadata["source"] == "otel"

    async def test_delete_existing(self, storage: InMemoryStorage):
        await storage.put("ns", "k1", b"v")
        assert await storage.delete("ns", "k1") is True
        assert await storage.get("ns", "k1") is None

    async def test_delete_missing_returns_false(self, storage: InMemoryStorage):
        assert await storage.delete("ns", "nonexistent") is False


class TestInMemoryStorageList:
    async def test_list_all_keys_in_namespace(self, storage: InMemoryStorage):
        for i in range(5):
            await storage.put("ns", f"k{i}", b"v")
        keys = await storage.list("ns")
        assert sorted(keys) == ["k0", "k1", "k2", "k3", "k4"]

    async def test_list_with_prefix(self, storage: InMemoryStorage):
        await storage.put("ns", "trace/001", b"v")
        await storage.put("ns", "trace/002", b"v")
        await storage.put("ns", "graph/001", b"v")
        keys = await storage.list("ns", prefix="trace/")
        assert sorted(keys) == ["trace/001", "trace/002"]

    async def test_list_respects_limit(self, storage: InMemoryStorage):
        for i in range(10):
            await storage.put("ns", f"k{i:02d}", b"v")
        keys = await storage.list("ns", limit=3)
        assert len(keys) == 3

    async def test_list_empty_namespace(self, storage: InMemoryStorage):
        assert await storage.list("empty-ns") == []


class TestInMemoryStorageQuery:
    async def test_query_by_metadata_equality(self, storage: InMemoryStorage):
        await storage.put("ns", "k1", b"v", metadata={"agent_id": "a1"})
        await storage.put("ns", "k2", b"v", metadata={"agent_id": "a2"})
        await storage.put("ns", "k3", b"v", metadata={"agent_id": "a1"})
        results = await storage.query("ns", {"agent_id": "a1"})
        assert len(results) == 2
        for r in results:
            assert r.metadata["agent_id"] == "a1"

    async def test_query_multiple_filters(self, storage: InMemoryStorage):
        await storage.put("ns", "k1", b"v", metadata={"a": 1, "b": 2})
        await storage.put("ns", "k2", b"v", metadata={"a": 1, "b": 3})
        results = await storage.query("ns", {"a": 1, "b": 2})
        assert len(results) == 1
        assert results[0].key == "k1"

    async def test_query_no_matches(self, storage: InMemoryStorage):
        await storage.put("ns", "k1", b"v", metadata={"a": 1})
        results = await storage.query("ns", {"a": 999})
        assert results == []

    async def test_query_limit(self, storage: InMemoryStorage):
        for i in range(10):
            await storage.put("ns", f"k{i}", b"v", metadata={"group": "x"})
        results = await storage.query("ns", {"group": "x"}, limit=3)
        assert len(results) == 3


class TestInMemoryStorageNamespaces:
    async def test_namespaces_returns_sorted(self, storage: InMemoryStorage):
        await storage.put("traces", "k1", b"v")
        await storage.put("graphs", "k1", b"v")
        await storage.put("memory", "k1", b"v")
        ns = await storage.namespaces()
        assert ns == ["graphs", "memory", "traces"]

    async def test_empty_storage_returns_empty_list(self, storage: InMemoryStorage):
        assert await storage.namespaces() == []


class TestInMemoryStorageConcurrency:
    async def test_concurrent_writes_to_different_keys(self, storage: InMemoryStorage):
        async def write(key: str):
            await storage.put("ns", key, b"v")

        await asyncio.gather(*[write(f"k{i}") for i in range(20)])
        keys = await storage.list("ns")
        assert len(keys) == 20

    async def test_concurrent_upsert_same_key(self, storage: InMemoryStorage):
        async def upsert(val: int):
            await storage.put("ns", "shared", str(val).encode())

        await asyncio.gather(*[upsert(i) for i in range(10)])
        rec = await storage.get("ns", "shared")
        assert rec is not None
        assert rec.value in [str(i).encode() for i in range(10)]


class TestStorageRecordText:
    def test_text_decodes_utf8(self):
        rec = StorageRecord(
            namespace="ns",
            key="k",
            value=b'{"hello": "world"}',
        )
        assert rec.text == '{"hello": "world"}'


class TestWorkspaceStorageABC:
    def test_cannot_instantiate_abc(self):
        with pytest.raises(TypeError):
            WorkspaceStorage()  # type: ignore[abstract]

    async def test_close_default_noop(self, storage: InMemoryStorage):
        await storage.close()

    async def test_async_context_manager(self, storage: InMemoryStorage):
        async with storage as s:
            assert s is storage
            await s.put("ns", "k", b"v")
