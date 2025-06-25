"""Document service for business logic operations"""

from typing import List, Optional
from app.models.document import Document, DocumentCreate, DocumentUpdate
from app.models.chunk import Chunk
from app.repositories.factory import repository_factory
from app.core.exceptions import DocumentNotFound, LibraryNotFound


class DocumentService:
    def __init__(self, document_repo, chunk_repo, library_repo):
    # Use dependency injection - no fallback to factory
        self.document_repo = document_repo
        self.chunk_repo = chunk_repo
        self.library_repo = library_repo

    async def _get_document_or_raise(self, document_id: str) -> Document:
        """Get document or raise exception"""
        document = await self.document_repo.get_by_id(document_id)
        if not document:
            raise DocumentNotFound(document_id)
        return document

    async def _update_document_chunk_count(self, document: Document) -> None:
        """Update document chunk count"""
        document.chunk_count = await self.chunk_repo.count_by_document_id(document.id)

    async def create_document(
        self, library_id: str, document_create: DocumentCreate
    ) -> Document:
        """Create a new document in a library"""
    # Verify library exists
        library = await self.library_repo.get_by_id(library_id)
        if not library:
            raise LibraryNotFound(library_id)

    # Create document
        document = await self.document_repo.create_from_request(
            document_create, library_id=library_id
        )

        return document

    async def get_document(self, document_id: str) -> Document:
        """Get a document by ID"""
        document = await self._get_document_or_raise(document_id)

    # Update chunk count
        await self._update_document_chunk_count(document)
        return document

    async def list_documents(
        self, library_id: str, skip: int = 0, limit: int = 100
    ) -> List[Document]:
        """List all documents in a library with pagination"""
    # Verify library exists
        library = await self.library_repo.get_by_id(library_id)
        if not library:
            raise LibraryNotFound(library_id)

        documents = await self.document_repo.get_by_library_id(
            library_id, skip=skip, limit=limit
        )

    # Update chunk counts for each document
        for document in documents:
            await self._update_document_chunk_count(document)

        return documents

    async def count_documents(self, library_id: str) -> int:
        """Count total number of documents in a library"""
        return await self.document_repo.count_by_library_id(library_id)

    async def update_document(
        self, document_id: str, document_update: DocumentUpdate
    ) -> Document:
        """Update a document"""
        document = await self.document_repo.update_from_request(
            document_id, document_update
        )
        if not document:
            raise DocumentNotFound(document_id)

        return document

    async def delete_document(self, document_id: str) -> bool:
        """Delete a document and all its chunks"""
        await self._get_document_or_raise(document_id)

    # Delete all chunks in the document
        await self.chunk_repo.delete_by_document_id(document_id)

    # Delete the document
        return await self.document_repo.delete(document_id)

    async def get_document_chunks(
        self, document_id: str, skip: int = 0, limit: int = 100
    ) -> List[Chunk]:
        """Get all chunks belonging to a document"""
    # Verify document exists
        await self.get_document(document_id)

        return await self.chunk_repo.get_by_document_id(
            document_id, skip=skip, limit=limit
        )
