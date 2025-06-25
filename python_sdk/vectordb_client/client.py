"""Synchronous VectorDB client for Python SDK"""

import asyncio
from typing import Dict, List, Optional, Any
import threading
import atexit

from .async_client import AsyncVectorDBClient
from .models import (
    Library,
    LibraryCreate,
    LibraryUpdate,
    LibraryList,
    Document,
    DocumentCreate,
    DocumentUpdate,
    DocumentList,
    Chunk,
    ChunkCreate,
    ChunkUpdate,
    ChunkList,
    SearchRequest,
    SearchResponse,
    MetadataFilter,
    HealthCheck,
    StorageStats,
)


class VectorDBClient:
    """
    Synchronous client for VectorDB API

    This is a wrapper around AsyncVectorDBClient that provides a synchronous
    interface for easier use in non-async contexts.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        api_key: Optional[str] = None,
        timeout: float = 30.0,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        max_connections: int = 100,
        verify_ssl: bool = True,
    ):
        """
        Initialize synchronous VectorDB client

        Args:
            base_url: Base URL of the VectorDB API
            api_key: Optional API key for authentication
            timeout: Request timeout in seconds
            max_retries: Maximum number of retry attempts
            retry_delay: Initial delay between retries (exponential backoff)
            max_connections: Maximum number of connections in pool
            verify_ssl: Whether to verify SSL certificates
        """
        self._async_client = AsyncVectorDBClient(
            base_url=base_url,
            api_key=api_key,
            timeout=timeout,
            max_retries=max_retries,
            retry_delay=retry_delay,
            max_connections=max_connections,
            verify_ssl=verify_ssl,
        )

    # Event loop management for sync client
        self._loop = None
        self._loop_thread = None
        self._setup_event_loop()

    # Register cleanup on exit
        atexit.register(self.close)

    def _setup_event_loop(self):
        """Set up event loop in a separate thread"""

        def run_loop():
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)
            self._loop.run_forever()

        self._loop_thread = threading.Thread(target=run_loop, daemon=True)
        self._loop_thread.start()

    # Wait for loop to be ready
        while self._loop is None:
            threading.Event().wait(0.01)

    def _run_async(self, coro):
        """Run async coroutine in the event loop"""
        if self._loop is None or self._loop.is_closed():
            raise RuntimeError("Event loop is not available")

        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return future.result()

    def close(self):
        """Close the client and cleanup resources"""
        if self._loop and not self._loop.is_closed():
        # Close async client
            self._run_async(self._async_client.close())

        # Stop event loop
            self._loop.call_soon_threadsafe(self._loop.stop)

        # Wait for thread to finish
            if self._loop_thread and self._loop_thread.is_alive():
                self._loop_thread.join(timeout=5.0)

    def __enter__(self):
        """Context manager entry"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.close()

# Health and Admin

    def health_check(self) -> HealthCheck:
        """Get API health status"""
        return self._run_async(self._async_client.health_check())

    def get_storage_stats(self) -> StorageStats:
        """Get storage statistics"""
        return self._run_async(self._async_client.get_storage_stats())

# Library Operations

    def create_library(self, library: LibraryCreate) -> Library:
        """Create a new library"""
        return self._run_async(self._async_client.create_library(library))

    def get_library(self, library_id: str) -> Library:
        """Get library by ID"""
        return self._run_async(self._async_client.get_library(library_id))

    def list_libraries(
        self, skip: int = 0, limit: int = 20, search: Optional[str] = None
    ) -> LibraryList:
        """List libraries with pagination and search"""
        return self._run_async(self._async_client.list_libraries(skip, limit, search))

    def update_library(self, library_id: str, library: LibraryUpdate) -> Library:
        """Update library"""
        return self._run_async(self._async_client.update_library(library_id, library))

    def delete_library(self, library_id: str) -> None:
        """Delete library"""
        return self._run_async(self._async_client.delete_library(library_id))

    def rebuild_library_index(self, library_id: str) -> Dict[str, Any]:
        """Rebuild library index"""
        return self._run_async(self._async_client.rebuild_library_index(library_id))

# Document Operations

    def create_document(self, library_id: str, document: DocumentCreate) -> Document:
        """Create a new document in a library"""
        return self._run_async(self._async_client.create_document(library_id, document))

    def get_document(self, library_id: str, document_id: str) -> Document:
        """Get document by ID"""
        return self._run_async(self._async_client.get_document(library_id, document_id))

    def list_documents(
        self,
        library_id: str,
        skip: int = 0,
        limit: int = 20,
        search: Optional[str] = None,
    ) -> DocumentList:
        """List documents in a library"""
        return self._run_async(
            self._async_client.list_documents(library_id, skip, limit, search)
        )

    def update_document(
        self, library_id: str, document_id: str, document: DocumentUpdate
    ) -> Document:
        """Update document"""
        return self._run_async(
            self._async_client.update_document(library_id, document_id, document)
        )

    def delete_document(self, library_id: str, document_id: str) -> None:
        """Delete document"""
        return self._run_async(
            self._async_client.delete_document(library_id, document_id)
        )

# Chunk Operations

    def create_chunk(self, library_id: str, chunk: ChunkCreate) -> Chunk:
        """Create a new chunk in a library"""
        return self._run_async(self._async_client.create_chunk(library_id, chunk))

    def get_chunk(self, library_id: str, chunk_id: str) -> Chunk:
        """Get chunk by ID"""
        return self._run_async(self._async_client.get_chunk(library_id, chunk_id))

    def list_chunks(
        self,
        library_id: str,
        skip: int = 0,
        limit: int = 20,
        document_id: Optional[str] = None,
    ) -> ChunkList:
        """List chunks in a library"""
        return self._run_async(
            self._async_client.list_chunks(library_id, skip, limit, document_id)
        )

    def update_chunk(self, library_id: str, chunk_id: str, chunk: ChunkUpdate) -> Chunk:
        """Update chunk"""
        return self._run_async(
            self._async_client.update_chunk(library_id, chunk_id, chunk)
        )

    def delete_chunk(self, library_id: str, chunk_id: str) -> None:
        """Delete chunk"""
        return self._run_async(self._async_client.delete_chunk(library_id, chunk_id))

# Search Operations

    def search(
        self,
        library_id: str,
        query: str,
        k: int = 10,
        similarity_threshold: Optional[float] = None,
        metadata_filters: Optional[List[MetadataFilter]] = None,
        filter_mode: str = "and",
        include_metadata: bool = True,
    ) -> SearchResponse:
        """Perform vector similarity search"""
        return self._run_async(
            self._async_client.search(
                library_id=library_id,
                query=query,
                k=k,
                similarity_threshold=similarity_threshold,
                metadata_filters=metadata_filters,
                filter_mode=filter_mode,
                include_metadata=include_metadata,
            )
        )

    def search_simple(
        self,
        library_id: str,
        query: str,
        k: int = 10,
        threshold: Optional[float] = None,
    ) -> SearchResponse:
        """Simple search using GET method"""
        return self._run_async(
            self._async_client.search_simple(library_id, query, k, threshold)
        )

# Batch Operations

    def create_chunks_batch(
        self, library_id: str, chunks: List[ChunkCreate], batch_size: int = 50
    ) -> List[Chunk]:
        """Create multiple chunks in batches"""
        return self._run_async(
            self._async_client.create_chunks_batch(library_id, chunks, batch_size)
        )

    def search_multiple_libraries(
        self, library_ids: List[str], query: str, k: int = 10, **kwargs
    ) -> Dict[str, SearchResponse]:
        """Search across multiple libraries concurrently"""
        return self._run_async(
            self._async_client.search_multiple_libraries(
                library_ids, query, k, **kwargs
            )
        )

# Utility Methods

    def ping(self) -> bool:
        """Check if the API is reachable"""
        return self._run_async(self._async_client.ping())

    def wait_for_ready(
        self, timeout: float = 30.0, check_interval: float = 1.0
    ) -> bool:
        """Wait for the API to be ready"""
        return self._run_async(
            self._async_client.wait_for_ready(timeout, check_interval)
        )

    def create_metadata_filter(
        self, field: str, operator: str, value: Any
    ) -> MetadataFilter:
        """Helper to create metadata filter"""
        return self._async_client.create_metadata_filter(field, operator, value)

    def get_library_statistics(self, library_id: str) -> Dict[str, Any]:
        """Get detailed library statistics"""
        return self._run_async(self._async_client.get_library_statistics(library_id))

# Convenience Methods

    def create_library_simple(
        self,
        name: str,
        description: Optional[str] = None,
        index_type: str = "linear",
        similarity_metric: str = "cosine",
    ) -> Library:
        """Create library with simple parameters"""
        from .models import LibraryCreate, IndexType, SimilarityMetric

        library = LibraryCreate(
            name=name,
            description=description,
            index_type=IndexType(index_type),
            similarity_metric=SimilarityMetric(similarity_metric),
        )
        return self.create_library(library)

    def add_text_chunk(
        self,
        library_id: str,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
        document_id: Optional[str] = None,
    ) -> Chunk:
        """Add a text chunk with simple parameters"""
        chunk = ChunkCreate(text=text, metadata=metadata or {}, document_id=document_id)
        return self.create_chunk(library_id, chunk)

    def search_text(
        self,
        library_id: str,
        query: str,
        limit: int = 10,
        min_similarity: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """Simple text search returning basic results"""
        response = self.search_simple(library_id, query, limit, min_similarity)

        results = []
        for result in response.results:
            results.append(
                {
                    "text": result.chunk.text,
                    "similarity": result.similarity_score,
                    "rank": result.rank,
                    "metadata": (
                        result.chunk.metadata.dict() if result.chunk.metadata else {}
                    ),
                    "chunk_id": result.chunk.id,
                }
            )

        return results

    def filter_by_metadata(
        self,
        library_id: str,
        query: str,
        field: str,
        operator: str,
        value: Any,
        k: int = 10,
    ) -> SearchResponse:
        """Search with a single metadata filter"""
        metadata_filter = self.create_metadata_filter(field, operator, value)
        return self.search(
            library_id=library_id, query=query, k=k, metadata_filters=[metadata_filter]
        )
