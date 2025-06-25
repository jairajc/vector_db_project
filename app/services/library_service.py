"""Library services for business logic operations"""

from typing import List
from app.models.library import Library, LibraryCreate, LibraryUpdate
from app.repositories.factory import repository_factory
from app.core.exceptions import LibraryNotFound
from app.core.constants import IndexType
from app.services.index_service import IndexService


class LibraryService:
    """Services for library management operations"""

    def __init__(self, library_repo, chunk_repo, document_repo, index_service):
    # Use dependency injection - no fallback to factory
        self.library_repo = library_repo
        self.chunk_repo = chunk_repo
        self.document_repo = document_repo
        self.index_service = index_service

    async def _get_library_or_raise(self, library_id: str) -> Library:
        """Get library or raise exception"""
        library = await self.library_repo.get_by_id(library_id)
        if not library:
            raise LibraryNotFound(library_id)
        return library

    async def _update_library_stats(self, library: Library) -> None:
        """Update library statistics"""
        library.stats.total_chunks = await self.chunk_repo.count_by_library_id(
            library.id
        )
        library.stats.total_documents = await self.document_repo.count_by_library_id(
            library.id
        )

    async def create_library(self, library_create: LibraryCreate) -> Library:
        """Create a new library"""
        library = await self.library_repo.create_from_request(library_create)

    # Initialize empty index for the library with LSH configuration if provided
        index_kwargs = {}
        if library.index_type == IndexType.LSH and library.lsh_config:
            index_kwargs["lsh_config"] = library.lsh_config

        await self.index_service.create_index(
            library.id, library.index_type, library.similarity_metric, **index_kwargs
        )

        return library

    async def get_library(self, library_id: str) -> Library:
        """Get a library by ID"""
        library = await self._get_library_or_raise(library_id)

    # Update stats
        await self._update_library_stats(library)
        return library

    async def list_libraries(self, skip: int = 0, limit: int = 100) -> List[Library]:
        """List all libraries with pagination"""
        libraries = await self.library_repo.list_all(skip=skip, limit=limit)

    # Update stats for each library
        for library in libraries:
            await self._update_library_stats(library)

        return libraries

    async def count_libraries(self) -> int:
        """Count total number of libraries"""
        return await self.library_repo.count()

    async def update_library(
        self, library_id: str, library_update: LibraryUpdate
    ) -> Library:
        """Update a library"""
        library = await self.library_repo.update_from_request(
            library_id, library_update
        )
        if not library:
            raise LibraryNotFound(library_id)

    # Check if index rebuild is needed
        needs_index_rebuild = (
            library_update.index_type is not None
            or library_update.similarity_metric is not None
            or library_update.lsh_config is not None
        )

        if needs_index_rebuild:
        # Prepare index kwargs with LSH configuration if needed
            index_kwargs = {}
            if library.index_type == IndexType.LSH and library.lsh_config:
                index_kwargs["lsh_config"] = library.lsh_config

            await self.index_service.rebuild_index(
                library_id,
                library.index_type,
                library.similarity_metric,
                **index_kwargs
            )

        return library

    async def delete_library(self, library_id: str) -> bool:
        """Delete a library and all its data"""
        await self._get_library_or_raise(library_id)

    # Delete all chunks and documents in the library
        await self.chunk_repo.delete_by_library_id(library_id)
        await self.document_repo.delete_by_library_id(library_id)

    # Delete the index
        await self.index_service.delete_index(library_id)

    # Delete the library
        return await self.library_repo.delete(library_id)

    async def rebuild_index(self, library_id: str):
        """Rebuild the vector index for a library"""
        library = await self._get_library_or_raise(library_id)

    # Get all chunks in the library
        chunks = await self.chunk_repo.get_by_library_id(
            library_id, skip=0, limit=10000
        )

    # Rebuild the index with LSH configuration if needed
        index_kwargs = {}
        if library.index_type == IndexType.LSH and library.lsh_config:
            index_kwargs["lsh_config"] = library.lsh_config

        await self.index_service.rebuild_index(
            library_id, library.index_type, library.similarity_metric, **index_kwargs
        )
    # Re add all chunks to the index
        for chunk in chunks:
            index_metadata = {
                "text": chunk.text,
                "document_id": chunk.document_id,
                "created_at": (
                    chunk.metadata.created_at.isoformat()
                    if chunk.metadata and chunk.metadata.created_at
                    else None
                ),
                "updated_at": (
                    chunk.metadata.updated_at.isoformat()
                    if chunk.metadata and chunk.metadata.updated_at
                    else None
                ),
                "source": chunk.metadata.source if chunk.metadata else None,
                "page_number": chunk.metadata.page_number if chunk.metadata else None,
                "section": chunk.metadata.section if chunk.metadata else None,
                "custom_fields": chunk.metadata.custom_fields if chunk.metadata else {},
            }

            await self.index_service.add_vector(
                library_id,
                chunk.id,
                chunk.embedding,
                index_metadata,
            )
