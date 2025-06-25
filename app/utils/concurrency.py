"""Concurrency utilities for async thread-safe operations"""

import asyncio
import time
from typing import Optional, Dict, Any
from contextlib import asynccontextmanager
from app.core.exceptions import ConcurrencyError


class AsyncReadWriteLock:
    def __init__(self):
        self._readers = 0
        self._writers = 0
        self._read_condition = asyncio.Condition()
        self._write_condition = asyncio.Condition()
        self._lock = asyncio.Lock()  # Protects the counters

    @asynccontextmanager
    async def read_lock(self, timeout: Optional[float] = None):
        """Acquire a read lock"""
        acquired = await self._acquire_read(timeout)
        if not acquired:
            raise ConcurrencyError(
                "Failed to acquire read lock", "read", int(timeout) if timeout else None
            )
        try:
            yield
        finally:
            await self._release_read()

    @asynccontextmanager
    async def write_lock(self, timeout: Optional[float] = None):
        """Acquire a write lock"""
        acquired = await self._acquire_write(timeout)
        if not acquired:
            raise ConcurrencyError(
                "Failed to acquire write lock",
                "write",
                int(timeout) if timeout else None,
            )
        try:
            yield
        finally:
            await self._release_write()

    async def _acquire_read(self, timeout: Optional[float] = None) -> bool:
        """Acquire a read lock with optional timeout"""
        async with self._read_condition:
            start_time = time.time()
            while self._writers > 0:
                if timeout is not None:
                    elapsed = time.time() - start_time
                    if elapsed >= timeout:
                        return False
                    remaining = timeout - elapsed
                    try:
                        await asyncio.wait_for(
                            self._read_condition.wait(), timeout=remaining
                        )
                    except asyncio.TimeoutError:
                        return False
                else:
                    await self._read_condition.wait()

            async with self._lock:
                self._readers += 1
            return True

    async def _release_read(self):
        """Release a read lock"""
        async with self._lock:
            self._readers -= 1
            readers_count = self._readers

        if readers_count == 0:
            async with self._write_condition:
                self._write_condition.notify_all()

    async def _acquire_write(self, timeout: Optional[float] = None) -> bool:
        """Acquire a write lock with optional timeout"""
        async with self._write_condition:
            start_time = time.time()
            while self._writers > 0 or self._readers > 0:
                if timeout is not None:
                    elapsed = time.time() - start_time
                    if elapsed >= timeout:
                        return False
                    remaining = timeout - elapsed
                    try:
                        await asyncio.wait_for(
                            self._write_condition.wait(), timeout=remaining
                        )
                    except asyncio.TimeoutError:
                        return False
                else:
                    await self._write_condition.wait()

            async with self._lock:
                self._writers += 1
            return True

    async def _release_write(self):
        """Release a write lock"""
        async with self._lock:
            self._writers -= 1

        async with self._write_condition:
            self._write_condition.notify_all()

        async with self._read_condition:
            self._read_condition.notify_all()


class AsyncLockManager:
    """Manages async locks for different resources to prevent deadlocks"""

    def __init__(self):
        self._locks: Dict[str, AsyncReadWriteLock] = {}
        self._lock_creation_lock = asyncio.Lock()

    async def get_lock(self, resource_id: str) -> AsyncReadWriteLock:
        """Get or create a lock for the given resource"""
        if resource_id not in self._locks:
            async with self._lock_creation_lock:

                # Double-check pattern
                if resource_id not in self._locks:
                    self._locks[resource_id] = AsyncReadWriteLock()
        return self._locks[resource_id]

    async def remove_lock(self, resource_id: str):
        """Remove a lock for the given resource (cleanup)"""
        async with self._lock_creation_lock:
            self._locks.pop(resource_id, None)

    async def read_lock(self, resource_id: str, timeout: Optional[float] = None):
        """Async context manager for read lock on a resource"""
        lock = await self.get_lock(resource_id)
        return lock.read_lock(timeout)

    async def write_lock(self, resource_id: str, timeout: Optional[float] = None):
        """Async context manager for write lock on a resource"""
        lock = await self.get_lock(resource_id)
        return lock.write_lock(timeout)


class ConcurrentDataStructure:
    """Thread safe data structure wrapper with async locks"""

    def __init__(self):
        self._data: Dict[str, Any] = {}
        self._lock = AsyncReadWriteLock()

    async def get(self, key: str):
        """Thread-safe get operation"""
        async with self._lock.read_lock():
            return self._data.get(key)

    async def set(self, key: str, value):
        """Thread-safe set operation"""
        async with self._lock.write_lock():
            self._data[key] = value

    async def delete(self, key: str) -> bool:
        """Thread-safe delete operation"""
        async with self._lock.write_lock():
            if key in self._data:
                del self._data[key]
                return True
            return False

    async def keys(self):
        """Thread-safe keys operation"""
        async with self._lock.read_lock():
            return list(self._data.keys())

    async def values(self):
        """Thread-safe values operation"""
        async with self._lock.read_lock():
            return list(self._data.values())

    async def items(self):
        """Thread-safe items operation"""
        async with self._lock.read_lock():
            return list(self._data.items())

    async def contains(self, key: str) -> bool:
        """Thread-safe contains operation"""
        async with self._lock.read_lock():
            return key in self._data

    async def size(self) -> int:
        """Thread-safe size operation"""
        async with self._lock.read_lock():
            return len(self._data)


# Global async lock manager instance
lock_manager = AsyncLockManager()


# Convenience functions for common patterns
async def with_read_lock(resource_id: str, operation, timeout: Optional[float] = None):
    """Execute an operation with a read lock"""
    async with await lock_manager.read_lock(resource_id, timeout):
        return await operation()


async def with_write_lock(resource_id: str, operation, timeout: Optional[float] = None):
    """Execute an operation with a write lock"""
    async with await lock_manager.write_lock(resource_id, timeout):
        return await operation()
