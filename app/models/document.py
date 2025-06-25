"""Document models for vector database"""

from datetime import datetime, timezone
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field, field_validator, model_validator, ConfigDict


def _utc_now() -> datetime:
    """Return current UTC datetime ( cuz more explicit and testable than lambda)"""
    return datetime.now(timezone.utc)


class DocumentCreate(BaseModel):
    """Schema for creating a new document"""
    
    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
        extra='forbid'
    )

    title: str = Field(
        ..., 
        min_length=1, 
        max_length=500, 
        description="Title of the document"
    )
    content: Optional[str] = Field(None, description="Full content of the document")
    source: Optional[str] = Field(
        None, 
        max_length=1000, 
        description="Source URL or file path"
    )
    author: Optional[str] = Field(
        None, 
        max_length=255, 
        description="Author of the document"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict, 
        description="Additional metadata"
    )

    @field_validator("title")
    @classmethod
    def validate_title(cls, v: str) -> str:
        """Validate document title is not empty or whitespace"""
        if not v or not v.strip():
            raise ValueError("Document title cannot be empty or whitespace")
        return v.strip()

    @field_validator("content")
    @classmethod
    def validate_content(cls, v: Optional[str]) -> Optional[str]:
        """Validate content if provided to avoid empty strings"""
        if v is not None and len(v.strip()) == 0:
            return None  # Convert empty strings to None
        return v.strip() if v else v

    def __str__(self) -> str:
        """String representation for debugging """
        return f"DocumentCreate(title='{self.title}')"

    def __repr__(self) -> str:
        """Detailed representation for debugging to keep it organized"""
        return f"DocumentCreate(title='{self.title}', author='{self.author}')"


class DocumentUpdate(BaseModel):
    """Schema for updating an existing document """
    
    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
        extra='forbid'
    )

    title: Optional[str] = Field(None, min_length=1, max_length=500, description="Document title")
    content: Optional[str] = Field(None, description="Document content")
    source: Optional[str] = Field(None, max_length=1000, description="Source URL or file path")
    author: Optional[str] = Field(None, max_length=255, description="Document author")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")

    @field_validator("title")
    @classmethod
    def validate_title(cls, v: Optional[str]) -> Optional[str]:
        """Validate document title is not empty or whitespace if provided"""
        if v is not None and (not v or not v.strip()):
            raise ValueError("Document title cannot be empty or only whitespace")
        return v.strip() if v else v

    @field_validator("content")
    @classmethod
    def validate_content(cls, v: Optional[str]) -> Optional[str]:
        """Validate content if provided to avoid empty strings"""
        if v is not None and len(v.strip()) == 0:
            return None  
        return v.strip() if v else v

    @model_validator(mode='after')
    def validate_at_least_one_field(self) -> 'DocumentUpdate':
        """To ensure at least one field is being updated"""
        fields_to_check = ['title', 'content', 'source', 'author', 'metadata']
        if not any(getattr(self, field) is not None for field in fields_to_check):
            raise ValueError("At least one field must be provided for update")
        return self


class Document(BaseModel):
    """Complete document model"""
    
    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
        extra='forbid'
    )

    id: str = Field(..., description="Unique identifier for the document")
    title: str = Field(..., min_length=1, max_length=500, description="Document title")
    content: Optional[str] = Field(None, description="Document content")
    source: Optional[str] = Field(None, max_length=1000, description="Source URL or file path")
    author: Optional[str] = Field(None, max_length=255, description="Document author")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    library_id: str = Field(
        ..., 
        description="ID of the library containing this document"
    )
    chunk_count: int = Field(
        default=0, 
        ge=0, 
        description="Number of chunks in this document"
    )
    created_at: datetime = Field(default_factory=_utc_now, description="Creation timestamp")
    updated_at: datetime = Field(default_factory=_utc_now, description="Last update timestamp")

    @field_validator("title")
    @classmethod
    def validate_title(cls, v: str) -> str:
        """Validate document title is not empty or whitespace"""
        if not v or not v.strip():
            raise ValueError("Document title cannot be empty or only whitespace")
        return v.strip()

    @field_validator("content")
    @classmethod
    def validate_content(cls, v: Optional[str]) -> Optional[str]:
        """Validate content if provided"""
        if v is not None and len(v.strip()) == 0:
            return None  
        return v.strip() if v else v

    @field_validator("chunk_count")
    @classmethod
    def validate_chunk_count(cls, v: int) -> int:
        """Validate chunk count is non-negative"""
        if v < 0:
            raise ValueError("Chunk count cannot be negative")
        return v

    def __str__(self) -> str:
        """String representation for debugging"""
        return f"Document(id={self.id}, title='{self.title}')"

    def __repr__(self) -> str:
        """Detailed representation for debugging"""
        return (f"Document(id='{self.id}', title='{self.title}', "
                f"library_id='{self.library_id}', chunks={self.chunk_count})")


class DocumentResponse(BaseModel):
    """Response schema for document operations"""
    
    model_config = ConfigDict(
        from_attributes=True,
        validate_assignment=True,
        extra='forbid'
    )

    id: str = Field(..., description="Unique identifier for the document")
    title: str = Field(..., description="Document title")
    content: Optional[str] = Field(None, description="Document content")
    source: Optional[str] = Field(None, description="Source URL or file path")
    author: Optional[str] = Field(None, description="Document author")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    library_id: str = Field(..., description="Library ID")
    chunk_count: int = Field(default=0, description="Number of chunks")
    created_at: datetime = Field(..., description="Creation timestamp")
    updated_at: datetime = Field(..., description="Last update timestamp")


class DocumentList(BaseModel):
    """Response schema for listing documents """
    
    model_config = ConfigDict(
        validate_assignment=True,
        extra='forbid'
    )

    documents: List[DocumentResponse] = Field(..., description="List of documents")
    total: int = Field(..., ge=0, description="Total number of documents")
    page: int = Field(..., ge=1, description="Current page number")
    page_size: int = Field(..., ge=1, le=100, description="Number of items per page")
    has_next: bool = Field(default=False, description="Whether there are more pages")
    has_previous: bool = Field(default=False, description="Whether there are previous pages")

    @model_validator(mode='after')
    def validate_pagination_consistency(self) -> 'DocumentList':
        """Validate pagination consistency"""
    # Calculate expected values
        expected_has_next = (self.page - 1) * self.page_size + len(self.documents) < self.total
        expected_has_previous = self.page > 1
        
    # Validate consistency
        if self.has_next != expected_has_next:
            self.has_next = expected_has_next
        if self.has_previous != expected_has_previous:
            self.has_previous = expected_has_previous
            
        return self

    def __str__(self) -> str:
        """String representation for debugging"""
        return f"DocumentList(total={self.total}, page={self.page}, items={len(self.documents)})"

    def __repr__(self) -> str:
        """Simillarly to __str__ but more detailed for debugging"""
        return (f"DocumentList(documents={len(self.documents)}, total={self.total}, "
                f"page={self.page}, page_size={self.page_size})")
