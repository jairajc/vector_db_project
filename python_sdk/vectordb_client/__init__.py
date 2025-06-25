"""VectorDB Python SDK - Official client library for Vector Database API"""

__version__ = "1.0.0"
__author__ = "VectorDB Team"
__email__ = "support@vectordb.ai"

from .client import VectorDBClient
from .async_client import AsyncVectorDBClient
from .models import (
    Library,
    LibraryCreate,
    LibraryUpdate,
    Document,
    DocumentCreate,
    DocumentUpdate,
    Chunk,
    ChunkCreate,
    ChunkUpdate,
    SearchResult,
    MetadataFilter,
    SearchRequest,
)
from .exceptions import (
    VectorDBError,
    LibraryNotFoundError,
    DocumentNotFoundError,
    ChunkNotFoundError,
    ValidationError,
    ConnectionError,
)

__all__ = [
    # Main clients
    "VectorDBClient",
    "AsyncVectorDBClient",
    # Data models
    "Library",
    "LibraryCreate",
    "LibraryUpdate",
    "Document",
    "DocumentCreate",
    "DocumentUpdate",
    "Chunk",
    "ChunkCreate",
    "ChunkUpdate",
    "SearchResult",
    "MetadataFilter",
    "SearchRequest",
    # Exceptions
    "VectorDBError",
    "LibraryNotFoundError",
    "DocumentNotFoundError",
    "ChunkNotFoundError",
    "ValidationError",
    "ConnectionError",
]
