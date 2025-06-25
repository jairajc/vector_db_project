"""API endpoints for vector similarity search"""

from typing import List, Optional, Dict, Any, Union
from fastapi import APIRouter, HTTPException, Query, Depends, status
from pydantic import BaseModel, Field
from app.models.chunk import SearchResult
from app.services.index_service import IndexService
from app.core.constants import DEFAULT_SEARCH_K, MAX_SEARCH_K, SimilarityMetric
from app.core.exceptions import LibraryNotFound

router = APIRouter()


class MetadataFilter(BaseModel):
    """Metadata filter for search operations"""

    field: str = Field(..., description="Metadata field to filter on")
    operator: str = Field(
        ...,
        description="Filter operator (eq, ne, gt, gte, lt, lte, in, not_in, contains)",
    )
    value: Union[str, int, float, bool, List[Union[str, int, float, bool]]] = Field(
        ..., description="Filter value"
    )


class SearchRequest(BaseModel):
    """Request model for vector search"""

    query: str = Field(
        ..., min_length=1, max_length=10000, description="Search query text"
    )
    k: int = Field(
        default=DEFAULT_SEARCH_K,
        ge=1,
        le=MAX_SEARCH_K,
        description="Number of results to return",
    )
    similarity_threshold: Optional[float] = Field(
        None, ge=0.0, le=1.0, description="Minimum similarity threshold"
    )
    include_metadata: bool = Field(
        default=True, description="Include chunk metadata in results"
    )
    metadata_filters: Optional[List[MetadataFilter]] = Field(
        None, description="Metadata filters to apply"
    )
    filter_mode: str = Field(
        default="and", description="Filter combination mode: 'and' or 'or'"
    )


class SearchResponse(BaseModel):
    """Response model for vector search"""

    query: str
    results: List[SearchResult]
    total_found: int
    search_time_ms: float
    library_id: str


# Dependency to get index services
def get_index_service() -> IndexService:
    """Get index service instance from container"""
    from app.core.container import container

    return container.get_index_service()


@router.post("/{library_id}/search", response_model=SearchResponse)
async def search_vectors(
    library_id: str,
    search_request: SearchRequest,
    service: IndexService = Depends(get_index_service),
):
    """Perform vector similarity search in a library"""
    try:
        import time

        start_time = time.time()

        results = await service.search(
            library_id=library_id,
            query_text=search_request.query,
            k=search_request.k,
            similarity_threshold=search_request.similarity_threshold,
            metadata_filters=search_request.metadata_filters,
            filter_mode=search_request.filter_mode,
        )

        search_time_ms = (time.time() - start_time) * 1000

        return SearchResponse(
            query=search_request.query,
            results=results,
            total_found=len(results),
            search_time_ms=search_time_ms,
            library_id=library_id,
        )
    except LibraryNotFound:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Library {library_id} not found",
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Search failed: {str(e)}",
        )


@router.get("/{library_id}/search")
async def search_vectors_get(
    library_id: str,
    q: str = Query(
        ..., min_length=1, max_length=10000, description="Search query text"
    ),
    k: int = Query(
        default=DEFAULT_SEARCH_K, ge=1, le=MAX_SEARCH_K, description="Number of results"
    ),
    threshold: Optional[float] = Query(
        None, ge=0.0, le=1.0, description="Similarity threshold"
    ),
    service: IndexService = Depends(get_index_service),
):
    """Perform vector similarity search using GET method"""
    search_request = SearchRequest(query=q, k=k, similarity_threshold=threshold)

    return await search_vectors(library_id, search_request, service)
