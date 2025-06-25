"""Data models for VectorDB SDK"""

from typing import Dict, List, Optional, Any, Union
from datetime import datetime
from enum import Enum
from pydantic import BaseModel, Field, validator


class IndexType(str, Enum):
    """Available index types"""

    LINEAR = "linear"
    KD_TREE = "kdtree"
    LSH = "lsh"


class SimilarityMetric(str, Enum):
    """Available similarity metrics"""

    COSINE = "cosine"
    EUCLIDEAN = "euclidean"
    DOT_PRODUCT = "dot_product"


class LSHConfig(BaseModel):
    """LSH index configuration"""

    num_hash_tables: int = Field(default=10, ge=1, le=50)
    num_hash_functions: int = Field(default=8, ge=1, le=32)
    hash_width: float = Field(default=1.0, ge=0.1, le=10.0)
    random_seed: int = Field(default=42, ge=0)


# Library Models


class LibraryBase(BaseModel):
    """Base library model"""

    name: str = Field(..., min_length=1, max_length=255)
    description: Optional[str] = Field(None, max_length=1000)
    index_type: IndexType = Field(default=IndexType.LINEAR)
    similarity_metric: SimilarityMetric = Field(default=SimilarityMetric.COSINE)
    lsh_config: Optional[LSHConfig] = None

    @validator("lsh_config", always=True)
    def validate_lsh_config(cls, v, values):
        """Validate LSH config is provided when index_type is LSH"""
        index_type = values.get("index_type")
        if index_type == IndexType.LSH and v is None:
            return LSHConfig()  # Use defaults
        elif index_type != IndexType.LSH and v is not None:
            return None  # Remove config for non-LSH indexes
        return v


class LibraryCreate(LibraryBase):
    """Model for creating a library"""

    pass


class LibraryUpdate(BaseModel):
    """Model for updating a library"""

    name: Optional[str] = Field(None, min_length=1, max_length=255)
    description: Optional[str] = Field(None, max_length=1000)
    lsh_config: Optional[LSHConfig] = None


class Library(LibraryBase):
    """Complete library model"""

    id: str
    created_at: datetime
    updated_at: datetime
    chunk_count: int = 0
    document_count: int = 0

    class Config:
        from_attributes = True


# Document Models


class DocumentBase(BaseModel):
    """Base document model"""

    title: str = Field(..., min_length=1, max_length=500)
    content: str = Field(..., min_length=1)
    source: Optional[str] = Field(None, max_length=500)
    author: Optional[str] = Field(None, max_length=255)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class DocumentCreate(DocumentBase):
    """Model for creating a document"""

    pass


class DocumentUpdate(BaseModel):
    """Model for updating a document"""

    title: Optional[str] = Field(None, min_length=1, max_length=500)
    content: Optional[str] = Field(None, min_length=1)
    source: Optional[str] = Field(None, max_length=500)
    author: Optional[str] = Field(None, max_length=255)
    metadata: Optional[Dict[str, Any]] = None


class Document(DocumentBase):
    """Complete document model"""

    id: str
    library_id: str
    created_at: datetime
    updated_at: datetime
    chunk_count: int = 0

    class Config:
        from_attributes = True


# Chunk Models


class ChunkMetadata(BaseModel):
    """Chunk metadata model"""

    created_at: datetime
    updated_at: datetime
    source: Optional[str] = None
    page_number: Optional[int] = None
    section: Optional[str] = None
    custom_fields: Dict[str, Any] = Field(default_factory=dict)


class ChunkBase(BaseModel):
    """Base chunk model"""

    text: str = Field(..., min_length=1, max_length=10000)
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)


class ChunkCreate(ChunkBase):
    """Model for creating a chunk"""

    document_id: Optional[str] = None


class ChunkUpdate(BaseModel):
    """Model for updating a chunk"""

    text: Optional[str] = Field(None, min_length=1, max_length=10000)
    metadata: Optional[Dict[str, Any]] = None


class Chunk(ChunkBase):
    """Complete chunk model"""

    id: str
    document_id: Optional[str] = None
    library_id: str
    embedding: Optional[List[float]] = None
    metadata: ChunkMetadata

    class Config:
        from_attributes = True


# Search Models


class MetadataFilter(BaseModel):
    """Metadata filter for search operations"""

    field: str = Field(..., description="Metadata field to filter on")
    operator: str = Field(
        ...,
        description="Filter operator (eq, ne, gt, gte, lt, lte, in, not_in, contains)",
        regex="^(eq|ne|gt|gte|lt|lte|in|not_in|contains)$",
    )
    value: Union[str, int, float, bool, List[Union[str, int, float, bool]]] = Field(
        ..., description="Filter value"
    )


class SearchRequest(BaseModel):
    """Search request model"""

    query: str = Field(..., min_length=1, max_length=10000)
    k: int = Field(default=10, ge=1, le=100)
    similarity_threshold: Optional[float] = Field(None, ge=0.0, le=1.0)
    include_metadata: bool = Field(default=True)
    metadata_filters: Optional[List[MetadataFilter]] = None
    filter_mode: str = Field(default="and", regex="^(and|or)$")


class SearchResult(BaseModel):
    """Search result model"""

    chunk: Chunk
    similarity_score: float = Field(..., ge=0.0, le=1.0)
    rank: int = Field(..., ge=1)

    class Config:
        from_attributes = True


class SearchResponse(BaseModel):
    """Search response model"""

    query: str
    results: List[SearchResult]
    total_found: int
    search_time_ms: float
    library_id: str


# Response Models


class LibraryList(BaseModel):
    """Library list response"""

    libraries: List[Library]
    total: int
    page: int
    page_size: int
    has_next: bool
    has_previous: bool


class DocumentList(BaseModel):
    """Document list response"""

    documents: List[Document]
    total: int
    page: int
    page_size: int
    has_next: bool
    has_previous: bool


class ChunkList(BaseModel):
    """Chunk list response"""

    chunks: List[Chunk]
    total: int
    page: int
    page_size: int
    has_next: bool
    has_previous: bool


# Admin Models


class HealthCheck(BaseModel):
    """Health check response"""

    status: str
    timestamp: datetime
    version: str
    uptime_seconds: float
    database_status: str
    memory_usage_mb: float
    cpu_usage_percent: float


class StorageStats(BaseModel):
    """Storage statistics"""

    total_libraries: int
    total_documents: int
    total_chunks: int
    total_vectors: int
    storage_size_bytes: int
    index_sizes: Dict[str, int]
