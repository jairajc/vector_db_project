"""API endpoints for chunk management within libraries"""

from fastapi import APIRouter, HTTPException, Query, Depends, status
from app.models.chunk import ChunkCreate, ChunkUpdate, ChunkResponse
from app.services.chunk_service import ChunkService
from app.core.constants import DEFAULT_PAGE_SIZE, MAX_PAGE_SIZE
from app.core.exceptions import LibraryNotFound, ChunkNotFound
from app.core.container import container

router = APIRouter()


def get_chunk_service() -> ChunkService:
    """Get chunk service instance from container"""
    return container.get_chunk_service()


@router.post(
    "/{library_id}/chunks",
    response_model=ChunkResponse,
    status_code=status.HTTP_201_CREATED,
)
async def create_chunk(
    library_id: str,
    chunk_create: ChunkCreate,
    service: ChunkService = Depends(get_chunk_service),
):
    """Create a new chunk in a library"""
    try:
        chunk = await service.create_chunk(library_id, chunk_create)
        return ChunkResponse(**chunk.model_dump())
    except LibraryNotFound:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Library {library_id} not found",
        )
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))


@router.get("/{library_id}/chunks")
async def list_chunks(
    library_id: str,
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(
        DEFAULT_PAGE_SIZE, ge=1, le=MAX_PAGE_SIZE, description="Page size"
    ),
    service: ChunkService = Depends(get_chunk_service),
):
    """List all chunks in a library with pagination"""
    try:
        skip = (page - 1) * page_size

        chunks = await service.list_chunks(library_id, skip=skip, limit=page_size)
        total = await service.count_chunks(library_id)

        return {
            "chunks": [ChunkResponse(**chunk.model_dump()) for chunk in chunks],
            "total": total,
            "page": page,
            "page_size": page_size,
            "has_next": (skip + page_size) < total,
            "has_previous": page > 1,
        }
    except LibraryNotFound:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Library {library_id} not found",
        )


@router.get("/{library_id}/chunks/{chunk_id}", response_model=ChunkResponse)
async def get_chunk(
    library_id: str, chunk_id: str, service: ChunkService = Depends(get_chunk_service)
):
    """Get a chunk by ID"""
    try:
        chunk = await service.get_chunk(library_id, chunk_id)
        return ChunkResponse(**chunk.model_dump())
    except LibraryNotFound:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Library {library_id} not found",
        )
    except ChunkNotFound:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=f"Chunk {chunk_id} not found"
        )


@router.put("/{library_id}/chunks/{chunk_id}", response_model=ChunkResponse)
async def update_chunk(
    library_id: str,
    chunk_id: str,
    chunk_update: ChunkUpdate,
    service: ChunkService = Depends(get_chunk_service),
):
    """Update a chunk"""
    try:
        chunk = await service.update_chunk(library_id, chunk_id, chunk_update)
        return ChunkResponse(**chunk.model_dump())
    except LibraryNotFound:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Library {library_id} not found",
        )
    except ChunkNotFound:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=f"Chunk {chunk_id} not found"
        )
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))


@router.delete(
    "/{library_id}/chunks/{chunk_id}", status_code=status.HTTP_204_NO_CONTENT
)
async def delete_chunk(
    library_id: str, chunk_id: str, service: ChunkService = Depends(get_chunk_service)
):
    """Delete a chunk"""
    try:
        await service.delete_chunk(library_id, chunk_id)
    except LibraryNotFound:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Library {library_id} not found",
        )
    except ChunkNotFound:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=f"Chunk {chunk_id} not found"
        )
