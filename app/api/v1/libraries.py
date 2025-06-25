"""API endpoints for library management"""

from typing import List
from fastapi import APIRouter, HTTPException, Query, Depends, status
from app.models.library import (
    LibraryCreate,
    LibraryUpdate,
    LibraryResponse,
    LibraryList,
)
from app.services.library_service import LibraryService
from app.core.constants import DEFAULT_PAGE_SIZE, MAX_PAGE_SIZE
from app.core.exceptions import LibraryNotFound
from app.core.container import container

router = APIRouter()


# Dependency to get library services
def get_library_service() -> LibraryService:
    """Get library service instance from container"""
    return container.get_library_service()


@router.post("/", response_model=LibraryResponse, status_code=status.HTTP_201_CREATED)
async def create_library(
    library_create: LibraryCreate,
    service: LibraryService = Depends(get_library_service),
):
    """Create a new library"""
    try:
        library = await service.create_library(library_create)
        return LibraryResponse(**library.model_dump())
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))


@router.get("/", response_model=LibraryList)
async def list_libraries(
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(
        DEFAULT_PAGE_SIZE, ge=1, le=MAX_PAGE_SIZE, description="Page size"
    ),
    service: LibraryService = Depends(get_library_service),
):
    """List all libraries with pagination"""
    skip = (page - 1) * page_size

    libraries = await service.list_libraries(skip=skip, limit=page_size)
    total = await service.count_libraries()

    return LibraryList(
        libraries=[LibraryResponse(**lib.model_dump()) for lib in libraries],
        total=total,
        page=page,
        page_size=page_size,
        has_next=(skip + page_size) < total,
        has_previous=page > 1,
    )


@router.get("/{library_id}", response_model=LibraryResponse)
async def get_library(
    library_id: str, service: LibraryService = Depends(get_library_service)
):
    """Get a library by ID"""
    try:
        library = await service.get_library(library_id)
        return LibraryResponse(**library.model_dump())
    except LibraryNotFound:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Library {library_id} not found",
        )


@router.put("/{library_id}", response_model=LibraryResponse)
async def update_library(
    library_id: str,
    library_update: LibraryUpdate,
    service: LibraryService = Depends(get_library_service),
):
    """Update the library"""
    try:
        library = await service.update_library(library_id, library_update)
        return LibraryResponse(**library.model_dump())
    except LibraryNotFound:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Library {library_id} not found",
        )
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))


@router.delete("/{library_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_library(
    library_id: str, service: LibraryService = Depends(get_library_service)
):
    """Delete the library and all its chunks"""
    try:
        await service.delete_library(library_id)
    except LibraryNotFound:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Library {library_id} not found",
        )


@router.post("/{library_id}/rebuild-index", status_code=status.HTTP_202_ACCEPTED)
async def rebuild_index(
    library_id: str, service: LibraryService = Depends(get_library_service)
):
    """Rebuild the vector index for a library"""
    try:
        await service.rebuild_index(library_id)
        return {"message": f"Index rebuild started for library {library_id}"}
    except LibraryNotFound:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Library {library_id} not found",
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to rebuild index: {str(e)}",
        )
