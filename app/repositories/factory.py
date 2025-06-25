"""Repository factory for creating appropriate repository instances"""

from typing import Union
from app.core.config import settings
from app.repositories.memory_repository import (
    LibraryRepository as MemoryLibraryRepository,
    ChunkRepository as MemoryChunkRepository,
    DocumentRepository as MemoryDocumentRepository,
)
from app.repositories.file_repository import (
    FileLibraryRepository,
    FileChunkRepository,
    FileDocumentRepository,
)


class RepositoryFactory:
    def __init__(self):
        self._library_repo = None
        self._chunk_repo = None
        self._document_repo = None

    def create_library_repository(
        self,
    ) -> Union[MemoryLibraryRepository, FileLibraryRepository]:
        """Create a library repository based on persistence configuration"""
        if self._library_repo is None:
            if settings.persistence_type == "file":
                self._library_repo = FileLibraryRepository(settings.data_directory)
            else:
                self._library_repo = MemoryLibraryRepository()
        return self._library_repo

    def create_chunk_repository(
        self,
    ) -> Union[MemoryChunkRepository, FileChunkRepository]:
        """Create a chunk repository based on persistence configuration"""
        if self._chunk_repo is None:
            if settings.persistence_type == "file":
                self._chunk_repo = FileChunkRepository(settings.data_directory)
            else:
                self._chunk_repo = MemoryChunkRepository()
        return self._chunk_repo

    def create_document_repository(
        self,
    ) -> Union[MemoryDocumentRepository, FileDocumentRepository]:
        """Create a document repository based on persistence configuration"""
        if self._document_repo is None:
            if settings.persistence_type == "file":
                self._document_repo = FileDocumentRepository(settings.data_directory)
            else:
                self._document_repo = MemoryDocumentRepository()
        return self._document_repo


# Global instance
repository_factory = RepositoryFactory()
