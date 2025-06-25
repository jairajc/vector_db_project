"""Chunk models for vector database"""

from datetime import datetime, timezone
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, field_validator, model_validator, ConfigDict
from app.core.constants import (
    MIN_TEXT_LENGTH,
    MAX_TEXT_LENGTH,
    MIN_EMBEDDING_DIMENSION,
    MAX_EMBEDDING_DIMENSION,
)


def _utc_now() -> datetime:
    """Return current UTC datetime"""
    return datetime.now(timezone.utc)


class ChunkMetadata(BaseModel):
    """Metadata associated with a text chunk"""
    
    model_config = ConfigDict(
        validate_assignment=True,
        extra='forbid'
    )

    created_at: datetime = Field(default_factory=_utc_now, description="Creation timestamp")
    updated_at: datetime = Field(default_factory=_utc_now, description="Last update timestamp")
    source: Optional[str] = Field(
        None, 
        max_length=1000,
        description="Source of the chunk (e.g., file name, URL)"
    )
    page_number: Optional[int] = Field(
        None, 
        ge=1, 
        description="Page number if from a document"
    )
    section: Optional[str] = Field(
        None, 
        max_length=255,
        description="Section or chapter name"
    )
    custom_fields: Dict[str, Any] = Field(
        default_factory=dict, 
        description="Additional custom metadata"
    )

    @field_validator("source")
    @classmethod
    def validate_source(cls, v: Optional[str]) -> Optional[str]:
        """Validate source if provided"""
        if v is not None and len(v.strip()) == 0:
            return None  # Convert empty strings to None
        return v.strip() if v else v

    @field_validator("section")
    @classmethod
    def validate_section(cls, v: Optional[str]) -> Optional[str]:
        """Validate section if provided"""
        if v is not None and len(v.strip()) == 0:
            return None  
        return v.strip() if v else v

    def __str__(self) -> str:
        """String representation for debugging"""
        return f"ChunkMetadata(source='{self.source}', page={self.page_number})"

    def __repr__(self) -> str:
        """Detailed representation for debugging"""
        return (f"ChunkMetadata(source='{self.source}', page_number={self.page_number}, "
                f"section='{self.section}', custom_fields={len(self.custom_fields)} items)")


class ChunkCreate(BaseModel):
    """Schema for creating a new chunk"""
    
    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
        extra='forbid'
    )

    text: str = Field(
        ..., 
        min_length=MIN_TEXT_LENGTH, 
        max_length=MAX_TEXT_LENGTH,
        description="Text content of the chunk"
    )
    metadata: Optional[ChunkMetadata] = Field(None, description="Chunk metadata")
    document_id: Optional[str] = Field(None, description="ID of the parent document")

    @field_validator("text")
    @classmethod
    def validate_text(cls, v: str) -> str:
        """Validate text is not empty or whitespaces"""
        if not v or not v.strip():
            raise ValueError("Text cannot be empty or only whitespace")
        return v.strip()

    @field_validator("document_id")
    @classmethod
    def validate_document_id(cls, v: Optional[str]) -> Optional[str]:
        """Validate document ID if provided"""
        if v is not None and len(v.strip()) == 0:
            return None  
        return v.strip() if v else v

    def __str__(self) -> str:
        """String representation for debugging"""
        return f"ChunkCreate(text_length={len(self.text)})"

    def __repr__(self) -> str:
        """Detailed representation for debugging"""
        return (f"ChunkCreate(text='{self.text[:50]}...', "
                f"document_id={self.document_id})")


class ChunkUpdate(BaseModel):
    """Schema for updating an existing chunk"""
    
    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
        extra='forbid'
    )

    text: Optional[str] = Field(
        None, 
        min_length=MIN_TEXT_LENGTH, 
        max_length=MAX_TEXT_LENGTH,
        description="Text content of the chunk"
    )
    metadata: Optional[ChunkMetadata] = Field(None, description="Chunk metadata")
    document_id: Optional[str] = Field(None, description="ID of the parent document")

    @field_validator("text")
    @classmethod
    def validate_text(cls, v: Optional[str]) -> Optional[str]:
        """Validate text is not empty or only whitespace if provided"""
        if v is not None and (not v or not v.strip()):
            raise ValueError("Text cannot be empty or only whitespace")
        return v.strip() if v else v

    @field_validator("document_id")
    @classmethod
    def validate_document_id(cls, v: Optional[str]) -> Optional[str]:
        """Validate document ID if provided"""
        if v is not None and len(v.strip()) == 0:
            return None 
        return v.strip() if v else v

    @model_validator(mode='after')
    def validate_at_least_one_field(self) -> 'ChunkUpdate':
        """Ensure at least one field is being updated"""
        fields_to_check = ['text', 'metadata', 'document_id']
        if not any(getattr(self, field) is not None for field in fields_to_check):
            raise ValueError("At least one field must be provided for update")
        return self


class Chunk(BaseModel):
    """Complete chunk model with embedding"""
    
    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
        extra='forbid'
    )

    id: str = Field(..., description="Unique identifier for the chunk")
    text: str = Field(
        ..., 
        min_length=MIN_TEXT_LENGTH, 
        max_length=MAX_TEXT_LENGTH,
        description="Text content of the chunk"
    )
    embedding: List[float] = Field(..., description="Vector embedding of the text")
    metadata: ChunkMetadata = Field(..., description="Chunk metadata")
    document_id: Optional[str] = Field(None, description="ID of the parent document")
    library_id: str = Field(..., description="ID of the library containing this chunk")

    @field_validator("text")
    @classmethod
    def validate_text(cls, v: str) -> str:
        """Validate text is not empty or whitespaces"""
        if not v or not v.strip():
            raise ValueError("Text cannot be empty or only whitespace")
        return v.strip()

    @field_validator("embedding")
    @classmethod
    def validate_embedding(cls, v: List[float]) -> List[float]:
        """Validate embedding vector"""
        if not v:
            raise ValueError("Embedding cannot be empty")

        if len(v) < MIN_EMBEDDING_DIMENSION or len(v) > MAX_EMBEDDING_DIMENSION:
            raise ValueError(
                f"Embedding dimension must be between {MIN_EMBEDDING_DIMENSION} and {MAX_EMBEDDING_DIMENSION}"
            )

    # Check if all values are valid floats
        for i, val in enumerate(v):
            if not isinstance(val, (int, float)):
                raise ValueError(f"Embedding value at index {i} must be a number, got {type(val)}")
            if not (-1e10 <= val <= 1e10):
                raise ValueError(f"Embedding value at index {i} is out of valid range: {val}")

        return v

    @field_validator("document_id")
    @classmethod
    def validate_document_id(cls, v: Optional[str]) -> Optional[str]:
        """Validate document ID if provided"""
        if v is not None and len(v.strip()) == 0:
            return None  # Just adding this to convert empty strings to None
        return v.strip() if v else v

    @field_validator("library_id")
    @classmethod
    def validate_library_id(cls, v: str) -> str:
        """Validate library ID is not empty"""
        if not v or not v.strip():
            raise ValueError("Library ID cannot be empty or only whitespace")
        return v.strip()

    def __str__(self) -> str:
        """String representation for debugging"""
        return f"Chunk(id={self.id}, text_length={len(self.text)})"

    def __repr__(self) -> str:
        """Detailed representation for debugging"""
        return (f"Chunk(id='{self.id}', text='{self.text[:50]}...', "
                f"library_id='{self.library_id}', document_id={self.document_id})")


class ChunkResponse(BaseModel):
    """Response schema for chunk operations"""
    
    model_config = ConfigDict(
        from_attributes=True,
        validate_assignment=True,
        extra='forbid'
    )

    id: str = Field(..., description="Unique identifier for the chunk")
    text: str = Field(..., description="Text content of the chunk")
    embedding: List[float] = Field(..., description="Vector embedding of the text")
    metadata: ChunkMetadata = Field(..., description="Chunk metadata")
    document_id: Optional[str] = Field(None, description="ID of the parent document")
    library_id: str = Field(..., description="ID of the library containing this chunk")


class ChunkList(BaseModel):
    """Response schema for listing chunks"""
    
    model_config = ConfigDict(
        validate_assignment=True,
        extra='forbid'
    )

    chunks: List[ChunkResponse] = Field(..., description="List of chunks")
    total: int = Field(..., ge=0, description="Total number of chunks")
    page: int = Field(..., ge=1, description="Current page number")
    page_size: int = Field(..., ge=1, le=100, description="Number of items per page")
    has_next: bool = Field(default=False, description="Whether there are more pages")
    has_previous: bool = Field(default=False, description="Whether there are previous pages")

    @model_validator(mode='after')
    def validate_pagination_consistency(self) -> 'ChunkList':
        """Validate pagination consistency"""
    # Calculate expected values
        expected_has_next = (self.page - 1) * self.page_size + len(self.chunks) < self.total
        expected_has_previous = self.page > 1
        
    # Validate consistency
        if self.has_next != expected_has_next:
            self.has_next = expected_has_next
        if self.has_previous != expected_has_previous:
            self.has_previous = expected_has_previous
            
        return self

    def __str__(self) -> str:
        """String representation for debugging"""
        return f"ChunkList(total={self.total}, page={self.page}, items={len(self.chunks)})"

    def __repr__(self) -> str:
        """Detailed representation for debugging"""
        return (f"ChunkList(chunks={len(self.chunks)}, total={self.total}, "
                f"page={self.page}, page_size={self.page_size})")


class SearchResult(BaseModel):
    """Result from vector similarity search"""
    
    model_config = ConfigDict(
        validate_assignment=True,
        extra='forbid'
    )

    chunk: ChunkResponse = Field(..., description="Matched chunk")
    similarity_score: float = Field(
        ..., 
        ge=0.0, 
        le=1.0, 
        description="Similarity score (0-1)"
    )
    rank: int = Field(..., ge=1, description="Rank in search results")

    @field_validator("similarity_score")
    @classmethod
    def validate_similarity_score(cls, v: float) -> float:
        """Validate similarity score is within valid range (0-1)"""
        if not (0.0 <= v <= 1.0):
            raise ValueError("Similarity score must be between 0.0 and 1.0")
        return v

    @field_validator("rank")
    @classmethod
    def validate_rank(cls, v: int) -> int:
        """Validate rank is positive"""
        if v < 1:
            raise ValueError("Rank must be a positive integer")
        return v

    def __str__(self) -> str:
        """String representation for debugging"""
        return f"SearchResult(rank={self.rank}, score={self.similarity_score:.3f})"

    def __repr__(self) -> str:
        """Detailed representation for debugging"""
        return (f"SearchResult(chunk_id='{self.chunk.id}', "
                f"rank={self.rank}, similarity_score={self.similarity_score:.3f})")


class SearchResults(BaseModel):
    """Collection of search results"""
    
    model_config = ConfigDict(
        validate_assignment=True,
        extra='forbid'
    )

    results: List[SearchResult] = Field(..., description="List of search results")
    total_results: int = Field(..., ge=0, description="Total number of results found")
    query_time_ms: float = Field(..., ge=0.0, description="Query execution time in milliseconds")
    
    @field_validator("query_time_ms")
    @classmethod
    def validate_query_time(cls, v: float) -> float:
        """Validate query time is non-negative"""
        if v < 0.0:
            raise ValueError("Query time cannot be negative")
        return v

    @model_validator(mode='after')
    def validate_results_consistency(self) -> 'SearchResults':
        """Validate results consistency"""
    # Ensure ranks are sequential starting from 1
        for i, result in enumerate(self.results):
            expected_rank = i + 1
            if result.rank != expected_rank:
                result.rank = expected_rank
        
    # Ensure similarity scores are in descending order
        for i in range(1, len(self.results)):
            if self.results[i].similarity_score > self.results[i-1].similarity_score:
            # Swap if not in descending order
                self.results[i], self.results[i-1] = self.results[i-1], self.results[i]
        
        return self

    def __str__(self) -> str:
        """String representation for debugging"""
        return f"SearchResults(found={len(self.results)}, time={self.query_time_ms:.2f}ms)"

    def __repr__(self) -> str:
        """Detailed representation for debugging"""
        return (f"SearchResults(results={len(self.results)}, "
                f"total_results={self.total_results}, query_time_ms={self.query_time_ms})")
