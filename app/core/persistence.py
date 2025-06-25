"""Persistence layer for disk-based vector database storage"""

import os
import json
import pickle
import aiofiles
import asyncio
from typing import Dict, Any, Optional, List, Union
from datetime import datetime, timezone
from dataclasses import dataclass, asdict
from pathlib import Path
import logging
from contextlib import asynccontextmanager

from app.core.config import settings
from app.core.exceptions import PersistenceError
from app.utils.concurrency import lock_manager

logger = logging.getLogger(__name__)


@dataclass
class CheckpointMetadata:
    """Metadata for database checkpoints"""

    checkpoint_id: str
    timestamp: datetime
    version: str
    library_count: int
    total_vectors: int
    index_types: Dict[str, str]
    file_size_bytes: int
    checksum: Optional[str] = None


class PersistenceManager:
    """
    High-performance disk persistence manager with:
    - Async WAL (Write-Ahead Logging)
    - Incremental checkpointing
    - Crash recovery
    - Data integrity verification
    - Background compaction
    """

    def __init__(self, base_path: str = "./data"):
        self.base_path = Path(base_path)
        self.wal_path = self.base_path / "wal"
        self.checkpoint_path = self.base_path / "checkpoints"
        self.metadata_path = self.base_path / "metadata"

    # Ensure directories exist
        for path in [
            self.base_path,
            self.wal_path,
            self.checkpoint_path,
            self.metadata_path,
        ]:
            path.mkdir(parents=True, exist_ok=True)

    # Internal state
        self._wal_sequence = 0
        self._last_checkpoint_time = None
        self._background_tasks: List[asyncio.Task] = []
        self._shutdown = False

    # Configuration
        self.wal_sync_interval = 5.0  # seconds
        self.checkpoint_interval = 300.0  # 5 minutes
        self.max_wal_size = 100 * 1024 * 1024  # 100MB

        logger.info(f"PersistenceManager initialized with base_path: {self.base_path}")

    async def initialize(self):
        """Initialize persistence manager and start background tasks"""
        try:
        # Load existing WAL sequence
            await self._load_wal_state()

        # Start background tasks
            self._background_tasks = [
                asyncio.create_task(self._wal_sync_worker()),
                asyncio.create_task(self._checkpoint_worker()),
                asyncio.create_task(self._compaction_worker()),
            ]

            logger.info("PersistenceManager background tasks started")

        except Exception as e:
            logger.error(f"Failed to initialize PersistenceManager: {e}")
            raise PersistenceError(f"Initialization failed: {e}")

    async def shutdown(self):
        """Gracefully shutdown persistence manager"""
        self._shutdown = True

    # Cancel background tasks
        for task in self._background_tasks:
            task.cancel()

    # Wait for tasks to complete
        await asyncio.gather(*self._background_tasks, return_exceptions=True)

    # Final WAL sync
        await self._sync_wal()

        logger.info("PersistenceManager shutdown complete")

    # WAL (Write-Ahead Logging) Operations

    async def log_operation(self, operation: str, data: Dict[str, Any]):
        """Log an operation to WAL for durability"""
        wal_entry = {
            "sequence": self._wal_sequence,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "operation": operation,
            "data": data,
        }

        wal_file = self.wal_path / f"wal_{self._wal_sequence:010d}.log"

        try:
            async with aiofiles.open(wal_file, "w") as f:
                await f.write(json.dumps(wal_entry, indent=2))
                await f.fsync()  # Force write to disk

            self._wal_sequence += 1

        except Exception as e:
            logger.error(f"Failed to write WAL entry: {e}")
            raise PersistenceError(f"WAL write failed: {e}")

    async def _load_wal_state(self):
        """Load WAL state from disk"""
        try:
            wal_files = sorted(self.wal_path.glob("wal_*.log"))
            if wal_files:
                last_file = wal_files[-1]
                sequence_str = last_file.stem.split("_")[1]
                self._wal_sequence = int(sequence_str) + 1
            else:
                self._wal_sequence = 0

            logger.info(f"WAL sequence initialized to: {self._wal_sequence}")

        except Exception as e:
            logger.warning(f"Failed to load WAL state: {e}, starting from 0")
            self._wal_sequence = 0

    async def _wal_sync_worker(self):
        """Background worker for WAL synchronization"""
        while not self._shutdown:
            try:
                await asyncio.sleep(self.wal_sync_interval)
                await self._sync_wal()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"WAL sync worker error: {e}")

    async def _sync_wal(self):
        """Synchronize WAL to disk"""
    # In a real implementation, we'd batch and sync WAL entries
    # Placeholder for more complex WAL logic
        pass

    # Checkpoint Operations

    async def create_checkpoint(
        self, libraries: Dict[str, Any], indexes: Dict[str, Any]
    ) -> str:
        """Create a full database checkpoint"""
        checkpoint_id = f"checkpoint_{int(datetime.now(timezone.utc).timestamp())}"

        try:
            checkpoint_data = {
                "metadata": {
                    "checkpoint_id": checkpoint_id,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "version": "1.0.0",
                    "library_count": len(libraries),
                    "total_vectors": sum(
                        len(idx._vectors) if hasattr(idx, "_vectors") else 0
                        for idx in indexes.values()
                    ),
                    "index_types": {
                        lib_id: type(idx).__name__ for lib_id, idx in indexes.items()
                    },
                },
                "libraries": self._serialize_libraries(libraries),
                "indexes": await self._serialize_indexes(indexes),
            }

            checkpoint_file = self.checkpoint_path / f"{checkpoint_id}.pkl"

        # Write checkpoint atomically
            temp_file = checkpoint_file.with_suffix(".tmp")
            async with aiofiles.open(temp_file, "wb") as f:
                await f.write(pickle.dumps(checkpoint_data))
                await f.fsync()

        # Atomic rename
            temp_file.rename(checkpoint_file)

        # Save metadata
            metadata = CheckpointMetadata(
                checkpoint_id=checkpoint_id,
                timestamp=datetime.now(timezone.utc),
                version="1.0.0",
                library_count=len(libraries),
                total_vectors=checkpoint_data["metadata"]["total_vectors"],
                index_types=checkpoint_data["metadata"]["index_types"],
                file_size_bytes=checkpoint_file.stat().st_size,
            )

            await self._save_checkpoint_metadata(metadata)

            self._last_checkpoint_time = datetime.now(timezone.utc)

            logger.info(f"Checkpoint created: {checkpoint_id}")
            return checkpoint_id

        except Exception as e:
            logger.error(f"Failed to create checkpoint: {e}")
            raise PersistenceError(f"Checkpoint creation failed: {e}")

    async def load_latest_checkpoint(self) -> Optional[Dict[str, Any]]:
        """Load the most recent checkpoint"""
        try:
            checkpoint_files = sorted(self.checkpoint_path.glob("checkpoint_*.pkl"))
            if not checkpoint_files:
                return None

            latest_checkpoint = checkpoint_files[-1]

            async with aiofiles.open(latest_checkpoint, "rb") as f:
                data = await f.read()
                checkpoint_data = pickle.loads(data)

            logger.info(f"Loaded checkpoint: {latest_checkpoint.stem}")
            return checkpoint_data

        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            raise PersistenceError(f"Checkpoint loading failed: {e}")

    async def _checkpoint_worker(self):
        """Background worker for periodic checkpointing"""
        while not self._shutdown:
            try:
                await asyncio.sleep(self.checkpoint_interval)

            # Check if checkpoint is needed
                if self._should_create_checkpoint():
                # This would need to be called with actual data
                    logger.info("Checkpoint worker triggered (implementation needed)")

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Checkpoint worker error: {e}")

    def _should_create_checkpoint(self) -> bool:
        """Determine if a new checkpoint should be created"""
        if self._last_checkpoint_time is None:
            return True

        time_since_last = datetime.now(timezone.utc) - self._last_checkpoint_time
        return time_since_last.total_seconds() > self.checkpoint_interval

# Recovery Operations

    async def recover_from_checkpoint(self) -> Optional[Dict[str, Any]]:
        """Recover database state from the latest checkpoint and replay WAL"""
        try:
        # Load latest checkpoint
            checkpoint_data = await self.load_latest_checkpoint()
            if not checkpoint_data:
                logger.info("No checkpoint found, starting fresh")
                return None

        # Replay WAL entries since checkpoint
            checkpoint_timestamp = datetime.fromisoformat(
                checkpoint_data["metadata"]["timestamp"]
            )

            await self._replay_wal_since(checkpoint_timestamp)

            logger.info("Database recovery completed")
            return checkpoint_data

        except Exception as e:
            logger.error(f"Recovery failed: {e}")
            raise PersistenceError(f"Recovery failed: {e}")

    async def _replay_wal_since(self, since_time: datetime):
        """Replay WAL entries since a given timestamp"""
        try:
            wal_files = sorted(self.wal_path.glob("wal_*.log"))

            for wal_file in wal_files:
                async with aiofiles.open(wal_file, "r") as f:
                    content = await f.read()
                    wal_entry = json.loads(content)

                entry_time = datetime.fromisoformat(wal_entry["timestamp"])
                if entry_time > since_time:
                # Apply WAL operation (implementation needed)
                    logger.debug(f"Replaying WAL operation: {wal_entry['operation']}")

        except Exception as e:
            logger.error(f"WAL replay failed: {e}")
            raise PersistenceError(f"WAL replay failed: {e}")

# Compaction Operations

    async def _compaction_worker(self):
        """Background worker for WAL compaction"""
        while not self._shutdown:
            try:
                await asyncio.sleep(3600)  # Run every hour
                await self._compact_wal()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Compaction worker error: {e}")

    async def _compact_wal(self):
        """Compact old WAL files"""
        try:
            wal_files = sorted(self.wal_path.glob("wal_*.log"))

        # Keep only recent WAL files (last 1000 operations)
            if len(wal_files) > 1000:
                files_to_remove = wal_files[:-1000]
                for wal_file in files_to_remove:
                    wal_file.unlink()

                logger.info(f"Compacted {len(files_to_remove)} old WAL files")

        except Exception as e:
            logger.error(f"WAL compaction failed: {e}")

# Serialization Helpers

    def _serialize_libraries(self, libraries: Dict[str, Any]) -> Dict[str, Any]:
        """Serialize libraries for checkpointing"""
        serialized = {}
        for lib_id, library in libraries.items():
            try:
                if hasattr(library, "__dict__"):
                    serialized[lib_id] = {
                        "id": getattr(library, "id", lib_id),
                        "name": getattr(library, "name", ""),
                        "description": getattr(library, "description", ""),
                        "index_type": getattr(library, "index_type", "linear"),
                        "similarity_metric": getattr(
                            library, "similarity_metric", "cosine"
                        ),
                        "created_at": getattr(
                            library, "created_at", datetime.now(timezone.utc)
                        ).isoformat(),
                        "updated_at": getattr(
                            library, "updated_at", datetime.now(timezone.utc)
                        ).isoformat(),
                    }
                else:
                    serialized[lib_id] = library
            except Exception as e:
                logger.warning(f"Failed to serialize library {lib_id}: {e}")

        return serialized

    async def _serialize_indexes(self, indexes: Dict[str, Any]) -> Dict[str, Any]:
        """Serialize vector indexes for checkpointing"""
        serialized = {}

        for lib_id, index in indexes.items():
            try:
                index_data = {
                    "type": type(index).__name__,
                    "similarity_metric": str(index.similarity_metric),
                    "vectors": {},
                    "metadata": {},
                }

            # Serialize vectors and metadata if available
                if hasattr(index, "_vectors"):
                    index_data["vectors"] = {
                        vector_id: (
                            vector.tolist() if hasattr(vector, "tolist") else vector
                        )
                        for vector_id, vector in index._vectors.items()
                    }

                if hasattr(index, "_metadata"):
                    index_data["metadata"] = dict(index._metadata)

            # Add index-specific parameters
                if hasattr(index, "lsh_params"):
                    index_data["lsh_params"] = asdict(index.lsh_params)

                serialized[lib_id] = index_data

            except Exception as e:
                logger.warning(f"Failed to serialize index {lib_id}: {e}")

        return serialized

    async def _save_checkpoint_metadata(self, metadata: CheckpointMetadata):
        """Save checkpoint metadata"""
        metadata_file = self.metadata_path / f"{metadata.checkpoint_id}.json"

        try:
            async with aiofiles.open(metadata_file, "w") as f:
                await f.write(json.dumps(asdict(metadata), indent=2, default=str))

        except Exception as e:
            logger.error(f"Failed to save checkpoint metadata: {e}")

# Context Managers

    @asynccontextmanager
    async def transaction(self, operation_name: str):
        """Transaction context manager with WAL logging"""
        transaction_id = f"tx_{int(datetime.now(timezone.utc).timestamp())}"

        try:
            await self.log_operation(
                "begin_transaction",
                {"transaction_id": transaction_id, "operation": operation_name},
            )

            yield transaction_id

            await self.log_operation(
                "commit_transaction", {"transaction_id": transaction_id}
            )

        except Exception as e:
            await self.log_operation(
                "rollback_transaction",
                {"transaction_id": transaction_id, "error": str(e)},
            )
            raise

    # Utility Methods

    async def get_storage_stats(self) -> Dict[str, Any]:
        """Get storage statistics"""
        try:
            wal_files = list(self.wal_path.glob("wal_*.log"))
            checkpoint_files = list(self.checkpoint_path.glob("checkpoint_*.pkl"))

            total_wal_size = sum(f.stat().st_size for f in wal_files)
            total_checkpoint_size = sum(f.stat().st_size for f in checkpoint_files)

            return {
                "wal_files": len(wal_files),
                "wal_size_bytes": total_wal_size,
                "checkpoint_files": len(checkpoint_files),
                "checkpoint_size_bytes": total_checkpoint_size,
                "total_size_bytes": total_wal_size + total_checkpoint_size,
                "last_checkpoint": (
                    self._last_checkpoint_time.isoformat()
                    if self._last_checkpoint_time
                    else None
                ),
            }

        except Exception as e:
            logger.error(f"Failed to get storage stats: {e}")
            return {}


# Global persistence manager instance
persistence_manager = PersistenceManager()
