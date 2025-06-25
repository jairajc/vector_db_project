"""Document API endpoints"""

from typing import List
from fastapi import APIRouter, HTTPException, Query, status, Depends
from app.models.document import (
    DocumentCreate,
    DocumentUpdate,
    DocumentResponse,
    DocumentList,
)
from app.services.document_service import DocumentService
from app.core.exceptions import DocumentNotFound, LibraryNotFound
from app.core.container import container

router = APIRouter()


def get_document_service() -> DocumentService:
    """Get document service instance from container"""
    return container.get_document_service()


@router.post(
    "/{library_id}/documents",
    response_model=DocumentResponse,
    status_code=status.HTTP_201_CREATED,
)
async def create_document(
    library_id: str,
    document: DocumentCreate,
    service: DocumentService = Depends(get_document_service),
):
    """Create a new document in a library"""
    try:
        created_document = await service.create_document(library_id, document)
        return DocumentResponse(
            id=created_document.id,
            title=created_document.title,
            content=created_document.content,
            source=created_document.source,
            author=created_document.author,
            metadata=created_document.metadata,
            library_id=created_document.library_id,
            chunk_count=created_document.chunk_count,
            created_at=created_document.created_at,
            updated_at=created_document.updated_at,
        )
    except LibraryNotFound as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=e.message)


@router.get("/{library_id}/documents", response_model=DocumentList)
async def list_documents(
    library_id: str,
    skip: int = Query(0, ge=0, description="Number of documents to skip"),
    limit: int = Query(20, ge=1, le=100, description="Number of documents to return"),
    service: DocumentService = Depends(get_document_service),
):
    """List documents in a library"""
    try:
        documents = await service.list_documents(library_id, skip=skip, limit=limit)
        total = await service.count_documents(library_id)

        document_responses = [
            DocumentResponse(
                id=doc.id,
                title=doc.title,
                content=doc.content,
                source=doc.source,
                author=doc.author,
                metadata=doc.metadata,
                library_id=doc.library_id,
                chunk_count=doc.chunk_count,
                created_at=doc.created_at,
                updated_at=doc.updated_at,
            )
            for doc in documents
        ]

        return DocumentList(
            documents=document_responses,
            total=total,
            page=(skip // limit) + 1,
            page_size=limit,
            has_next=(skip + limit) < total,
            has_previous=skip > 0,
        )
    except LibraryNotFound as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=e.message)


@router.get("/{library_id}/documents/{document_id}", response_model=DocumentResponse)
async def get_document(
    library_id: str,
    document_id: str,
    service: DocumentService = Depends(get_document_service),
):
    """Get a document by ID"""
    try:
        document = await service.get_document(document_id)

# Verify document belongs to the specified library
        if document.library_id != library_id:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Document {document_id} not found in library {library_id}",
            )

        return DocumentResponse(
            id=document.id,
            title=document.title,
            content=document.content,
            source=document.source,
            author=document.author,
            metadata=document.metadata,
            library_id=document.library_id,
            chunk_count=document.chunk_count,
            created_at=document.created_at,
            updated_at=document.updated_at,
        )
    except DocumentNotFound as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=e.message)


@router.put("/{library_id}/documents/{document_id}", response_model=DocumentResponse)
async def update_document(
    library_id: str,
    document_id: str,
    document_update: DocumentUpdate,
    service: DocumentService = Depends(get_document_service),
):
    """Update a document"""
    try:
# First verify the document exists and belongs to the library
        existing_document = await service.get_document(document_id)
        if existing_document.library_id != library_id:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Document {document_id} not found in library {library_id}",
            )

        updated_document = await service.update_document(document_id, document_update)

        return DocumentResponse(
            id=updated_document.id,
            title=updated_document.title,
            content=updated_document.content,
            source=updated_document.source,
            author=updated_document.author,
            metadata=updated_document.metadata,
            library_id=updated_document.library_id,
            chunk_count=updated_document.chunk_count,
            created_at=updated_document.created_at,
            updated_at=updated_document.updated_at,
        )
    except DocumentNotFound as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=e.message)


@router.delete(
    "/{library_id}/documents/{document_id}", status_code=status.HTTP_204_NO_CONTENT
)
async def delete_document(
    library_id: str,
    document_id: str,
    service: DocumentService = Depends(get_document_service),
):
    """Delete a document and all its chunks"""
    try:
        existing_document = await service.get_document(document_id)
        if existing_document.library_id != library_id:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Document {document_id} not found in library {library_id}",
            )

        await service.delete_document(document_id)
    except DocumentNotFound as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=e.message)


@router.get("/{library_id}/documents/{document_id}/chunks")
async def get_document_chunks(
    library_id: str,
    document_id: str,
    skip: int = Query(0, ge=0, description="Number of chunks to skip"),
    limit: int = Query(20, ge=1, le=100, description="Number of chunks to return"),
    service: DocumentService = Depends(get_document_service),
):
    """Get all chunks belonging to a document"""
    try:
        existing_document = await service.get_document(document_id)
        if existing_document.library_id != library_id:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Document {document_id} not found in library {library_id}",
            )

        chunks = await service.get_document_chunks(document_id, skip=skip, limit=limit)
        total = len(chunks)  # For now, simple count

        return {
            "chunks": chunks,
            "total": total,
            "page": (skip // limit) + 1,
            "page_size": limit,
            "has_next": (skip + limit) < total,
            "has_previous": skip > 0,
        }
    except DocumentNotFound as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=e.message)
