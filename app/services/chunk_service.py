"""Chunk service for business logic operations"""

from typing import List
from app.models.chunk import Chunk, ChunkCreate, ChunkUpdate, ChunkMetadata
from app.repositories.factory import repository_factory
from app.core.exceptions import LibraryNotFound, ChunkNotFound
from app.utils.embeddings import embedding_service
from app.services.index_service import IndexService
from datetime import datetime


class ChunkService:
    def __init__(self, chunk_repo, library_repo, index_service):
        # Use dependency injection - no fallback to factory
        self.chunk_repo = chunk_repo
        self.library_repo = library_repo
        self.index_service = index_service

    async def _verify_library_exists(self, library_id: str) -> None:
        """Verify library exists"""
        library = await self.library_repo.get_by_id(library_id)
        if not library:
            raise LibraryNotFound(library_id)

    async def _verify_chunk_exists_in_library(
        self, library_id: str, chunk_id: str
    ) -> Chunk:
        """Verify chunk exists in library"""
        chunk = await self.chunk_repo.get_by_id(chunk_id)
        if not chunk:
            raise ChunkNotFound(chunk_id, library_id)

        if chunk.library_id != library_id:
            raise ChunkNotFound(chunk_id, library_id)

        return chunk

    async def create_chunk(self, library_id: str, chunk_create: ChunkCreate) -> Chunk:
        """Create a new chunk in a library"""
        await self._verify_library_exists(library_id)

        # Generate embedding for the text
        embedding = await embedding_service.generate_embedding(chunk_create.text)

        # Create metadata if not provided
        metadata = chunk_create.metadata or ChunkMetadata()

        # Create chunk
        chunk = Chunk(
            id="",  # Will be set by repository
            text=chunk_create.text,
            embedding=embedding,
            metadata=metadata,
            document_id=chunk_create.document_id,
            library_id=library_id,
        )

        # Save chunk to repository
        saved_chunk = await self.chunk_repo.create(chunk)

        # Add to vector index
        index_metadata = {
            "text": saved_chunk.text,
            "document_id": saved_chunk.document_id,
            "created_at": (
                saved_chunk.metadata.created_at.isoformat()
                if saved_chunk.metadata.created_at
                else None
            ),
            "updated_at": (
                saved_chunk.metadata.updated_at.isoformat()
                if saved_chunk.metadata.updated_at
                else None
            ),
            "source": saved_chunk.metadata.source,
            "page_number": saved_chunk.metadata.page_number,
            "section": saved_chunk.metadata.section,
            "custom_fields": saved_chunk.metadata.custom_fields or {},
        }

        await self.index_service.add_vector(
            library_id=library_id,
            vector_id=saved_chunk.id,
            vector=embedding,
            metadata=index_metadata,
        )

        return saved_chunk

    async def get_chunk(self, library_id: str, chunk_id: str) -> Chunk:
        """Get a chunk by ID"""
        await self._verify_library_exists(library_id)
        return await self._verify_chunk_exists_in_library(library_id, chunk_id)

    async def list_chunks(
        self, library_id: str, skip: int = 0, limit: int = 100
    ) -> List[Chunk]:
        """List chunks in a library with pagination"""
        await self._verify_library_exists(library_id)

        return await self.chunk_repo.get_by_library_id(
            library_id, skip=skip, limit=limit
        )

    async def count_chunks(self, library_id: str) -> int:
        """Count chunks in a library"""
        await self._verify_library_exists(library_id)

        return await self.chunk_repo.count_by_library_id(library_id)

    async def update_chunk(
        self, library_id: str, chunk_id: str, chunk_update: ChunkUpdate
    ) -> Chunk:
        """Update a chunk"""
        await self._verify_library_exists(library_id)
        existing_chunk = await self._verify_chunk_exists_in_library(
            library_id, chunk_id
        )

        # Prepare updates
        updates = {}
        regenerate_embedding = False

        if chunk_update.text is not None:
            updates["text"] = chunk_update.text
            regenerate_embedding = True

        if chunk_update.metadata is not None:
            updates["metadata"] = chunk_update.metadata

        if chunk_update.document_id is not None:
            updates["document_id"] = chunk_update.document_id

        # Generate new embedding if text changed
        if regenerate_embedding and chunk_update.text:
            new_embedding = await embedding_service.generate_embedding(
                chunk_update.text
            )
            updates["embedding"] = new_embedding

        # Update chunk in repository
        updated_chunk = await self.chunk_repo.update(chunk_id, updates)
        if not updated_chunk:
            raise ChunkNotFound(chunk_id, library_id)

        # Update vector index if embedding changed
        if regenerate_embedding:
            index_metadata = {
                "text": updated_chunk.text,
                "document_id": updated_chunk.document_id,
                "created_at": (
                    updated_chunk.metadata.created_at.isoformat()
                    if updated_chunk.metadata.created_at
                    else None
                ),
                "updated_at": (
                    updated_chunk.metadata.updated_at.isoformat()
                    if updated_chunk.metadata.updated_at
                    else None
                ),
                "source": updated_chunk.metadata.source,
                "page_number": updated_chunk.metadata.page_number,
                "section": updated_chunk.metadata.section,
                "custom_fields": updated_chunk.metadata.custom_fields or {},
            }

            await self.index_service.update_vector(
                library_id=library_id,
                vector_id=chunk_id,
                vector=new_embedding,
                metadata=index_metadata,
            )

        return updated_chunk

    async def delete_chunk(self, library_id: str, chunk_id: str) -> bool:
        """Delete a chunk"""
        await self._verify_library_exists(library_id)
        await self._verify_chunk_exists_in_library(library_id, chunk_id)

        # Remove from vector index
        await self.index_service.remove_vector(library_id, chunk_id)

        # Delete from repository
        return await self.chunk_repo.delete(chunk_id)
