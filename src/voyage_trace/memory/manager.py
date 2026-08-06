"""PartitionedMemory — the manager for all four memory partitions.

Holds one instance of each partition (episodic, semantic, procedural,
working) and maintains a stack of "active" :class:`MemoryScope` objects.
:meth:`mount` pushes a scope (the "plug-in" half of dynamic
plug/unplug); :meth:`unmount` pops it and clears that scope's working
memory (the "unplug" half).

The manager is usable as both a sync and an async context manager. For
the async flavour (recommended — every storage operation is async)::

    async with pm.use("agent-A", "round-1"):
        await pm.episodic().remember(pm.current(), "f1", {...})

The sync flavour pushes/pops the scope stack but cannot run the async
working-memory clear, so prefer ``async with`` whenever the working
partition has been touched.
"""

from __future__ import annotations

from typing import Any

from ..storage.base import WorkspaceStorage
from .base import MemoryScope
from .episodic import EpisodicMemory
from .procedural import ProceduralMemory
from .semantic import SemanticMemory
from .working import WorkingMemory


class PartitionedMemory:
    """Manager for the four memory partitions + active-scope stack.

    The active-scope stack enables *dynamic plug/unplug*: a governance
    round ``mount``\\ s a ``(target_agent_id, round_id)`` scope on entry
    and ``unmount``\\ s it (clearing working memory) on exit. Nested
    scopes are supported — the stack top is always the "current" scope.
    """

    def __init__(self, storage: WorkspaceStorage) -> None:
        self.storage = storage
        self._episodic = EpisodicMemory(storage)
        self._semantic = SemanticMemory(storage)
        self._procedural = ProceduralMemory(storage)
        self._working = WorkingMemory(storage)
        self._scope_stack: list[MemoryScope] = []
        # Configured by .use() for the next with/async-with block.
        self._pending_mount: tuple[str, str] | None = None

    # -- dynamic plug / unplug ------------------------------------------- #

    async def mount(self, target_agent_id: str, round_id: str) -> MemoryScope:
        """Create a new scope and push it onto the active-scope stack.

        This is the "plug-in" half of dynamic plug/unplug: after
        ``mount``, :meth:`current` returns this scope and every partition
        accessor (``episodic()`` / ``semantic()`` / …) can be called
        against it.
        """
        scope = MemoryScope(
            target_agent_id=target_agent_id,
            round_id=round_id,
            partition="",
        )
        self._scope_stack.append(scope)
        return scope

    async def unmount(self) -> MemoryScope | None:
        """Pop the active scope and clear its working memory.

        This is the "unplug" half: the popped scope's working memory is
        wiped so the next round starts clean, while episodic / semantic /
        procedural records persist for future recall. Returns the popped
        scope, or ``None`` if the stack was empty.
        """
        if not self._scope_stack:
            return None
        scope = self._scope_stack.pop()
        await self._working.clear(scope)
        return scope

    def current(self) -> MemoryScope | None:
        """Return the active scope (stack top) without popping."""
        return self._scope_stack[-1] if self._scope_stack else None

    # -- partition accessors --------------------------------------------- #

    def episodic(self) -> EpisodicMemory:
        return self._episodic

    def semantic(self) -> SemanticMemory:
        return self._semantic

    def procedural(self) -> ProceduralMemory:
        return self._procedural

    def working(self) -> WorkingMemory:
        return self._working

    # -- cross-round recall ---------------------------------------------- #

    async def recall_cross_round(
        self,
        target_agent_id: str,
        failure_signature: str,
        limit: int = 5,
    ) -> list[dict[str, Any]]:
        """Recall episodic records for ``failure_signature`` across all rounds.

        This is the primary "recall for reuse" path: when the meta-agent
        encounters a failure in a new round, it can look up every past
        governance outcome for the same failure signature on the same
        target agent — regardless of which round produced it.
        """
        scope = MemoryScope(
            target_agent_id=target_agent_id,
            round_id="*",
            partition="",
        )
        return await self._episodic.recall_similar(scope, failure_signature, limit=limit)

    # -- context-manager support ----------------------------------------- #
    #
    # ``use()`` stages a (target_agent_id, round_id) pair; the subsequent
    # ``with`` / ``async with`` entry mounts it and the exit unmounts.
    # Sync ``__exit__`` cannot run the async working-memory clear, so it
    # only pops the scope — prefer ``async with`` whenever the working
    # partition has been touched.

    def use(self, target_agent_id: str, round_id: str) -> "PartitionedMemory":
        """Configure the scope for the next ``with`` / ``async with`` block."""
        self._pending_mount = (target_agent_id, round_id)
        return self

    def __enter__(self) -> "PartitionedMemory":
        if self._pending_mount is not None:
            target, round_id = self._pending_mount
            self._scope_stack.append(MemoryScope(target, round_id, ""))
            self._pending_mount = None
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        # Sync unmount: pop the scope only. Working-memory clear is async
        # and cannot be awaited here — use ``async with`` if you need it.
        if self._scope_stack:
            self._scope_stack.pop()
        return False

    async def __aenter__(self) -> "PartitionedMemory":
        if self._pending_mount is not None:
            target, round_id = self._pending_mount
            await self.mount(target, round_id)
            self._pending_mount = None
        return self

    async def __aexit__(self, exc_type, exc, tb) -> bool:
        await self.unmount()
        return False
