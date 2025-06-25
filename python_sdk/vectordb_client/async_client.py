"""Async VectorDB client for Python SDK"""

import asyncio
import json
from typing import Dict, List, Optional, Any, Union
from urllib.parse import urljoin, urlencode
import aiohttp
import time
import logging
from http import HTTPStatus

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
from .exceptions import (
    VectorDBError,
    ConnectionError,
    TimeoutError,
    RateLimitError,
    create_exception_from_response,
)

logger = logging.getLogger(__name__)


class AsyncVectorDBClient:
    """
    Async client for VectorDB API with:
    - Full CRUD operations for libraries, documents, and chunks
    - Advanced search with metadata filtering
    - Automatic retry logic with exponential backoff
    - Connection pooling and timeout management
    - Comprehensive error handling
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
        Initialize async VectorDB client

        Args:
            base_url: Base URL of the VectorDB API
            api_key: Optional API key for authentication
            timeout: Request timeout in seconds
            max_retries: Maximum number of retry attempts
            retry_delay: Initial delay between retries (exponential backoff)
            max_connections: Maximum number of connections in pool
            verify_ssl: Whether to verify SSL certificates
        """
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay

    # Session configuration
        connector = aiohttp.TCPConnector(
            limit=max_connections,
            limit_per_host=max_connections // 4,
            verify_ssl=verify_ssl,
            keepalive_timeout=60,
        )

        timeout_config = aiohttp.ClientTimeout(total=timeout)

    # Default headers
        headers = {
            "Content-Type": "application/json",
            "User-Agent": "VectorDB-Python-SDK/1.0.0",
        }

        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        self._session = aiohttp.ClientSession(
            connector=connector, timeout=timeout_config, headers=headers
        )

        logger.info(f"AsyncVectorDBClient initialized for {base_url}")

    async def __aenter__(self):
        """Async context manager entry"""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.close()

    async def close(self):
        """Close the client session"""
        if self._session and not self._session.closed:
            await self._session.close()

# Core HTTP Methods

    async def _request(
        self,
        method: str,
        path: str,
        json_data: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Make HTTP request with retry logic"""

        url = urljoin(self.base_url, path.lstrip("/"))

    # Clean params (remove None values)
        if params:
            params = {k: v for k, v in params.items() if v is not None}

        request_context = {"method": method, "path": path, "url": url}

        last_exception = None

        for attempt in range(self.max_retries + 1):
            try:
                async with self._session.request(
                    method=method, url=url, json=json_data, params=params, **kwargs
                ) as response:

                    response_text = await response.text()

                # Handle successful responses
                    if HTTPStatus.OK.value <= response.status < HTTPStatus.MULTIPLE_CHOICES.value:
                        if response_text:
                            try:
                                return json.loads(response_text)
                            except json.JSONDecodeError:
                                return {"raw_response": response_text}
                        return {}

                # Handle rate limiting with retry
                    if response.status == HTTPStatus.TOO_MANY_REQUESTS.value and attempt < self.max_retries:
                        retry_after = response.headers.get("Retry-After")
                        delay = (
                            float(retry_after)
                            if retry_after
                            else self.retry_delay * (2**attempt)
                        )
                        logger.warning(f"Rate limited, retrying after {delay}s")
                        await asyncio.sleep(delay)
                        continue

                # Handle server errors with retry
                    if response.status >= HTTPStatus.INTERNAL_SERVER_ERROR.value and attempt < self.max_retries:
                        delay = self.retry_delay * (2**attempt)
                        logger.warning(
                            f"Server error {response.status}, retrying after {delay}s"
                        )
                        await asyncio.sleep(delay)
                        continue

                # Create exception for non-retryable errors
                    raise create_exception_from_response(
                        response.status, response_text, request_context
                    )

            except asyncio.TimeoutError as e:
                last_exception = TimeoutError("Request timed out", self.timeout)
                if attempt < self.max_retries:
                    delay = self.retry_delay * (2**attempt)
                    logger.warning(f"Timeout, retrying after {delay}s")
                    await asyncio.sleep(delay)
                    continue

            except aiohttp.ClientError as e:
                last_exception = ConnectionError(
                    f"Connection error: {str(e)}", self.base_url
                )
                if attempt < self.max_retries:
                    delay = self.retry_delay * (2**attempt)
                    logger.warning(f"Connection error, retrying after {delay}s")
                    await asyncio.sleep(delay)
                    continue

            except Exception as e:
                last_exception = VectorDBError(f"Unexpected error: {str(e)}")
                break  # Don't retry unexpected errors

    # If we get here, all retries failed
        raise last_exception or VectorDBError("Request failed after all retries")

    async def _get(
        self, path: str, params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Make GET request"""
        return await self._request("GET", path, params=params)

    async def _post(
        self, path: str, data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Make POST request"""
        return await self._request("POST", path, json_data=data)

    async def _put(
        self, path: str, data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Make PUT request"""
        return await self._request("PUT", path, json_data=data)

    async def _delete(self, path: str) -> Dict[str, Any]:
        """Make DELETE request"""
        return await self._request("DELETE", path)

# Health and Admin

    async def health_check(self) -> HealthCheck:
        """Get API health status"""
        response = await self._get("/health")
        return HealthCheck(**response)

    async def get_storage_stats(self) -> StorageStats:
        """Get storage statistics"""
        response = await self._get("/admin/stats")
        return StorageStats(**response)

# Library Operations

    async def create_library(self, library: LibraryCreate) -> Library:
        """Create a new library"""
        response = await self._post("/api/v1/libraries/", library.dict())
        return Library(**response)

    async def get_library(self, library_id: str) -> Library:
        """Get library by ID"""
        response = await self._get(f"/api/v1/libraries/{library_id}")
        return Library(**response)

    async def list_libraries(
        self, skip: int = 0, limit: int = 20, search: Optional[str] = None
    ) -> LibraryList:
        """List libraries with pagination and search"""
        params = {"skip": skip, "limit": limit}
        if search:
            params["search"] = search

        response = await self._get("/api/v1/libraries/", params=params)
        return LibraryList(**response)

    async def update_library(self, library_id: str, library: LibraryUpdate) -> Library:
        """Update library"""
        response = await self._put(
            f"/api/v1/libraries/{library_id}", library.dict(exclude_unset=True)
        )
        return Library(**response)

    async def delete_library(self, library_id: str) -> None:
        """Delete library"""
        await self._delete(f"/api/v1/libraries/{library_id}")

    async def rebuild_library_index(self, library_id: str) -> Dict[str, Any]:
        """Rebuild library index"""
        response = await self._post(f"/api/v1/libraries/{library_id}/rebuild")
        return response

# Document Operations

    async def create_document(
        self, library_id: str, document: DocumentCreate
    ) -> Document:
        """Create a new document in a library"""
        response = await self._post(
            f"/api/v1/libraries/{library_id}/documents", document.dict()
        )
        return Document(**response)

    async def get_document(self, library_id: str, document_id: str) -> Document:
        """Get document by ID"""
        response = await self._get(
            f"/api/v1/libraries/{library_id}/documents/{document_id}"
        )
        return Document(**response)

    async def list_documents(
        self,
        library_id: str,
        skip: int = 0,
        limit: int = 20,
        search: Optional[str] = None,
    ) -> DocumentList:
        """List documents in a library"""
        params = {"skip": skip, "limit": limit}
        if search:
            params["search"] = search

        response = await self._get(
            f"/api/v1/libraries/{library_id}/documents", params=params
        )
        return DocumentList(**response)

    async def update_document(
        self, library_id: str, document_id: str, document: DocumentUpdate
    ) -> Document:
        """Update document"""
        response = await self._put(
            f"/api/v1/libraries/{library_id}/documents/{document_id}",
            document.dict(exclude_unset=True),
        )
        return Document(**response)

    async def delete_document(self, library_id: str, document_id: str) -> None:
        """Delete document"""
        await self._delete(f"/api/v1/libraries/{library_id}/documents/{document_id}")

# Chunk Operations

    async def create_chunk(self, library_id: str, chunk: ChunkCreate) -> Chunk:
        """Create a new chunk in a library"""
        response = await self._post(
            f"/api/v1/libraries/{library_id}/chunks", chunk.dict()
        )
        return Chunk(**response)

    async def get_chunk(self, library_id: str, chunk_id: str) -> Chunk:
        """Get chunk by ID"""
        response = await self._get(f"/api/v1/libraries/{library_id}/chunks/{chunk_id}")
        return Chunk(**response)

    async def list_chunks(
        self,
        library_id: str,
        skip: int = 0,
        limit: int = 20,
        document_id: Optional[str] = None,
    ) -> ChunkList:
        """List chunks in a library"""
        params = {"skip": skip, "limit": limit}
        if document_id:
            params["document_id"] = document_id

        response = await self._get(
            f"/api/v1/libraries/{library_id}/chunks", params=params
        )
        return ChunkList(**response)

    async def update_chunk(
        self, library_id: str, chunk_id: str, chunk: ChunkUpdate
    ) -> Chunk:
        """Update chunk"""
        response = await self._put(
            f"/api/v1/libraries/{library_id}/chunks/{chunk_id}",
            chunk.dict(exclude_unset=True),
        )
        return Chunk(**response)

    async def delete_chunk(self, library_id: str, chunk_id: str) -> None:
        """Delete chunk"""
        await self._delete(f"/api/v1/libraries/{library_id}/chunks/{chunk_id}")

# Search Operations

    async def search(
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
        search_request = SearchRequest(
            query=query,
            k=k,
            similarity_threshold=similarity_threshold,
            metadata_filters=metadata_filters,
            filter_mode=filter_mode,
            include_metadata=include_metadata,
        )

        response = await self._post(
            f"/api/v1/libraries/{library_id}/search",
            search_request.dict(exclude_unset=True),
        )
        return SearchResponse(**response)

    async def search_simple(
        self,
        library_id: str,
        query: str,
        k: int = 10,
        threshold: Optional[float] = None,
    ) -> SearchResponse:
        """Simple search using GET method"""
        params = {"q": query, "k": k}
        if threshold:
            params["threshold"] = threshold

        response = await self._get(
            f"/api/v1/libraries/{library_id}/search", params=params
        )
        return SearchResponse(**response)

# Batch Operations

    async def create_chunks_batch(
        self, library_id: str, chunks: List[ChunkCreate], batch_size: int = 50
    ) -> List[Chunk]:
        """Create multiple chunks in batches"""
        created_chunks = []

        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]
            batch_tasks = [self.create_chunk(library_id, chunk) for chunk in batch]

            batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)

            for result in batch_results:
                if isinstance(result, Exception):
                    logger.error(f"Failed to create chunk: {result}")
                else:
                    created_chunks.append(result)

        return created_chunks

    async def search_multiple_libraries(
        self, library_ids: List[str], query: str, k: int = 10, **kwargs
    ) -> Dict[str, SearchResponse]:
        """Search across multiple libraries concurrently"""
        search_tasks = {
            library_id: self.search(library_id, query, k, **kwargs)
            for library_id in library_ids
        }

        results = await asyncio.gather(*search_tasks.values(), return_exceptions=True)

        search_results = {}
        for library_id, result in zip(library_ids, results):
            if isinstance(result, Exception):
                logger.error(f"Search failed for library {library_id}: {result}")
            else:
                search_results[library_id] = result

        return search_results

# Utility Methods

    async def ping(self) -> bool:
        """Check if the API is reachable"""
        try:
            await self.health_check()
            return True
        except Exception:
            return False

    async def wait_for_ready(
        self, timeout: float = 30.0, check_interval: float = 1.0
    ) -> bool:
        """Wait for the API to be ready"""
        start_time = time.time()

        while time.time() - start_time < timeout:
            if await self.ping():
                return True
            await asyncio.sleep(check_interval)

        return False

    def create_metadata_filter(
        self, field: str, operator: str, value: Any
    ) -> MetadataFilter:
        """Helper to create metadata filter"""
        return MetadataFilter(field=field, operator=operator, value=value)

# Advanced Features

    async def stream_search_results(
        self, library_id: str, queries: List[str], k: int = 10, **kwargs
    ):
        """Stream search results for multiple queries"""
        for query in queries:
            try:
                result = await self.search(library_id, query, k, **kwargs)
                yield query, result
            except Exception as e:
                yield query, e

    async def get_library_statistics(self, library_id: str) -> Dict[str, Any]:
        """Get detailed library statistics"""
        try:
        # Get basic library info
            library = await self.get_library(library_id)

        # Get document and chunk counts via list operations
            documents = await self.list_documents(library_id, limit=1)
            chunks = await self.list_chunks(library_id, limit=1)

            return {
                "library_id": library_id,
                "name": library.name,
                "index_type": library.index_type,
                "similarity_metric": library.similarity_metric,
                "document_count": documents.total,
                "chunk_count": chunks.total,
                "created_at": library.created_at,
                "updated_at": library.updated_at,
            }

        except Exception as e:
            logger.error(f"Failed to get library statistics: {e}")
            raise
