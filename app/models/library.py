"""Library models for vector database"""

from datetime import datetime, timezone
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field, field_validator, model_validator, ConfigDict
from app.core.constants import IndexType, SimilarityMetric


def _utc_now() -> datetime:
    """Return current UTC datetime"""
    return datetime.now(timezone.utc)


class LSHConfig(BaseModel):
    """Configuration for LSH (Locality Sensitive Hashing) index"""
    
    model_config = ConfigDict(
        validate_assignment=True,
        extra='forbid'
    )

    num_hash_tables: int = Field(default=10, ge=1, le=50, description="Number of hash tables (L)")
    hash_length: int = Field(default=8, ge=1, le=32, description="Number of hash functions per table (k)")
    hash_family: str = Field(default="cosine", description="Hash family type (cosine, euclidean, dot_product)")
    hash_width: float = Field(default=1.0, gt=0.0, le=10.0, description="Hash bucket width (for Euclidean distance)")
    random_seed: int = Field(default=42, ge=0, description="Random seed for reproducible hash functions")

    @field_validator("num_hash_tables")
    @classmethod
    def validate_num_hash_tables(cls, v: int) -> int:
        """Validate number of hash tables"""
        if v < 1 or v > 50:
            raise ValueError("Number of hash tables must be between 1 and 50")
        return v

    @field_validator("hash_length")
    @classmethod
    def validate_hash_length(cls, v: int) -> int:
        """Validate number of hash functions per table"""
        if v < 1 or v > 32:
            raise ValueError("Hash length must be between 1 and 32")
        return v

    @field_validator("hash_family")
    @classmethod
    def validate_hash_family(cls, v: str) -> str:
        """Validate hash family type"""
        valid_families = {"cosine", "euclidean", "dot_product"}
        if v.lower() not in valid_families:
            raise ValueError(f"Hash family must be one of: {', '.join(valid_families)}")
        return v.lower()

    def __str__(self) -> str:
        """String representation for debugging"""
        return f"LSHConfig(L={self.num_hash_tables}, k={self.hash_length}, family={self.hash_family})"

    def __repr__(self) -> str:
        """Detailed representation for debugging"""
        return (f"LSHConfig(num_hash_tables={self.num_hash_tables}, "
                f"hash_length={self.hash_length}, hash_family='{self.hash_family}', "
                f"hash_width={self.hash_width}, random_seed={self.random_seed})")


class LibraryCreate(BaseModel):
    """Schema for creating a new library"""
    
    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
        extra='forbid'
    )

    name: str = Field(..., min_length=1, max_length=255, description="Library name")
    description: Optional[str] = Field(None, max_length=1000, description="Library description")
    index_type: IndexType = Field(default=IndexType.LINEAR, description="Vector index algorithm")
    similarity_metric: SimilarityMetric = Field(default=SimilarityMetric.COSINE, description="Similarity calculation method")
    embedding_dimension: Optional[int] = Field(None, ge=1, le=4096, description="Vector embedding dimension")
    lsh_config: Optional[LSHConfig] = Field(None, description="LSH-specific configuration (only used when index_type=LSH)")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str) -> str:
        """Validate library name is not empty or whitespace"""
        if not v or not v.strip():
            raise ValueError("Library name cannot be empty or only whitespace")
        return v.strip()

    @field_validator("embedding_dimension")
    @classmethod
    def validate_embedding_dimension(cls, v: Optional[int]) -> Optional[int]:
        """Validate embedding dimension if provided"""
        if v is not None and (v < 1 or v > 4096):
            raise ValueError("Embedding dimension must be between 1 and 4096")
        return v


class LibraryUpdate(BaseModel):
    """Schema for updating an existing library"""
    
    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
        extra='forbid'
    )

    name: Optional[str] = Field(None, min_length=1, max_length=255, description="Library name")
    description: Optional[str] = Field(None, max_length=1000, description="Library description")
    index_type: Optional[IndexType] = Field(None, description="Vector index algorithm")
    similarity_metric: Optional[SimilarityMetric] = Field(None, description="Similarity calculation method")
    lsh_config: Optional[LSHConfig] = Field(None, description="LSH-specific configuration (only used when index_type=LSH)")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")

    @field_validator("name")
    @classmethod
    def validate_name(cls, v: Optional[str]) -> Optional[str]:
        """Validate library name is not empty or whitespace if provided"""
        if v is not None and (not v or not v.strip()):
            raise ValueError("Library name cannot be empty or only whitespace")
        return v.strip() if v else v


class LibraryStats(BaseModel):
    """Statistics for a library"""
    
    model_config = ConfigDict(
        validate_assignment=True,
        extra='forbid'
    )

    total_chunks: int = Field(default=0, ge=0, description="Total number of chunks")
    total_documents: int = Field(default=0, ge=0, description="Total number of documents")
    index_size_bytes: int = Field(default=0, ge=0, description="Index size in bytes")
    last_indexed_at: Optional[datetime] = Field(None, description="Last indexing timestamp")
    average_chunk_length: float = Field(default=0.0, ge=0.0, description="Average chunk text length")

    def __str__(self) -> str:
        """String representation for debugging"""
        return f"LibraryStats(chunks={self.total_chunks}, docs={self.total_documents})"

    def __repr__(self) -> str:
        """Detailed representation for debugging."""
        return (f"LibraryStats(total_chunks={self.total_chunks}, "
                f"total_documents={self.total_documents}, "
                f"index_size_bytes={self.index_size_bytes})")


class Library(BaseModel):
    """Complete library model"""
    
    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
        extra='forbid'
    )

    id: str = Field(..., description="Unique identifier for the library")
    name: str = Field(..., min_length=1, max_length=255, description="Library name")
    description: Optional[str] = Field(None, max_length=1000, description="Library description")
    index_type: IndexType = Field(..., description="Vector index algorithm")
    similarity_metric: SimilarityMetric = Field(..., description="Similarity calculation method")
    embedding_dimension: Optional[int] = Field(None, ge=1, le=4096, description="Vector embedding dimension")
    lsh_config: Optional[LSHConfig] = Field(None, description="LSH-specific configuration (only used when index_type=LSH)")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    stats: LibraryStats = Field(default_factory=LibraryStats, description="Library statistics")
    created_at: datetime = Field(default_factory=_utc_now, description="Creation timestamp")
    updated_at: datetime = Field(default_factory=_utc_now, description="Last update timestamp")

    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str) -> str:
        """Validate library name is not empty or whitespace"""
        if not v or not v.strip():
            raise ValueError("Library name cannot be empty or only whitespace")
        return v.strip()

    @field_validator("embedding_dimension")
    @classmethod
    def validate_embedding_dimension(cls, v: Optional[int]) -> Optional[int]:
        """Validate embedding dimension if provided"""
        if v is not None and (v < 1 or v > 4096):
            raise ValueError("Embedding dimension must be between 1 and 4096")
        return v

    def __str__(self) -> str:
        """String representation for debugging"""
        return f"Library(id={self.id}, name='{self.name}')"

    def __repr__(self) -> str:
        """Detailed representation for debugging"""
        return (f"Library(id='{self.id}', name='{self.name}', "
                f"index_type={self.index_type}, similarity_metric={self.similarity_metric})")


class LibraryResponse(BaseModel):
    """Response schema for library operations"""
    
    model_config = ConfigDict(
        from_attributes=True,
        validate_assignment=True,
        extra='forbid'
    )

    id: str = Field(..., description="Unique identifier for the library")
    name: str = Field(..., description="Library name")
    description: Optional[str] = Field(None, description="Library description")
    index_type: IndexType = Field(..., description="Vector index algorithm")
    similarity_metric: SimilarityMetric = Field(..., description="Similarity calculation method")
    embedding_dimension: Optional[int] = Field(None, description="Vector embedding dimension")
    lsh_config: Optional[LSHConfig] = Field(None, description="LSH-specific configuration (only used when index_type=LSH)")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    stats: LibraryStats = Field(..., description="Library statistics")
    created_at: datetime = Field(..., description="Creation timestamp")
    updated_at: datetime = Field(..., description="Last update timestamp")


class LibraryList(BaseModel):
    """Response schema for listing libraries"""
    
    model_config = ConfigDict(
        validate_assignment=True,
        extra='forbid'
    )

    libraries: List[LibraryResponse] = Field(..., description="List of libraries")
    total: int = Field(..., ge=0, description="Total number of libraries")
    page: int = Field(..., ge=1, description="Current page number")
    page_size: int = Field(..., ge=1, le=100, description="Number of items per page")
    has_next: bool = Field(default=False, description="Whether there are more pages")
    has_previous: bool = Field(default=False, description="Whether there are previous pages")

    @model_validator(mode='after')
    def validate_pagination_consistency(self) -> 'LibraryList':
        """Validate pagination consistency"""
    # Calculate expected values
        expected_has_next = (self.page - 1) * self.page_size + len(self.libraries) < self.total
        expected_has_previous = self.page > 1
        
    # Validate consistency
        if self.has_next != expected_has_next:
            self.has_next = expected_has_next
        if self.has_previous != expected_has_previous:
            self.has_previous = expected_has_previous
            
        return self

    def __str__(self) -> str:
        """String representation for debugging"""
        return f"LibraryList(total={self.total}, page={self.page}, items={len(self.libraries)})"

    def __repr__(self) -> str:
        """Detailed representation for debugging"""
        return (f"LibraryList(libraries={len(self.libraries)}, total={self.total}, "
                f"page={self.page}, page_size={self.page_size})")
