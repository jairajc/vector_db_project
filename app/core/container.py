"""Dependency injection container for managing service and repository instances"""

from typing import Dict, Any, Optional
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


class Container:
    """Dependency injection container with singleton pattern"""

    def __init__(self):
        self._instances: Dict[str, Any] = {}
        self._initialized = False

    def _initialize_repositories(self):
        """Initialize repository instances based on configuration"""
    # Early return if already initialized
        if self._initialized:
            return

        if settings.persistence_type == "file":
            self._instances["library_repository"] = FileLibraryRepository(
                settings.data_directory
            )
            self._instances["chunk_repository"] = FileChunkRepository(
                settings.data_directory
            )
            self._instances["document_repository"] = FileDocumentRepository(
                settings.data_directory
            )
        else:
            self._instances["library_repository"] = MemoryLibraryRepository()
            self._instances["chunk_repository"] = MemoryChunkRepository()
            self._instances["document_repository"] = MemoryDocumentRepository()

        self._initialized = True

    def get_library_repository(self):
        """Get the singleton library repository instance"""
        self._initialize_repositories()
        return self._instances["library_repository"]

    def get_chunk_repository(self):
        """Get the singleton chunk repository instance"""
        self._initialize_repositories()
        return self._instances["chunk_repository"]

    def get_document_repository(self):
        """Get the singleton document repository instance"""
        self._initialize_repositories()
        return self._instances["document_repository"]

    def get_library_service(self):
        """Get the singleton library service instance"""
    # Early return if already exists
        if "library_service" in self._instances:
            return self._instances["library_service"]

        from app.services.library_service import LibraryService

        self._instances["library_service"] = LibraryService(
            library_repo=self.get_library_repository(),
            chunk_repo=self.get_chunk_repository(),
            document_repo=self.get_document_repository(),
            index_service=self.get_index_service(),
        )
        return self._instances["library_service"]

    def get_chunk_service(self):
        """Get the singleton chunk service instance"""
        if "chunk_service" in self._instances:
            return self._instances["chunk_service"]

        from app.services.chunk_service import ChunkService

        self._instances["chunk_service"] = ChunkService(
            chunk_repo=self.get_chunk_repository(),
            library_repo=self.get_library_repository(),
            index_service=self.get_index_service(),
        )
        return self._instances["chunk_service"]

    def get_document_service(self):
        """Get the singleton document service instance"""
        if "document_service" in self._instances:
            return self._instances["document_service"]

        from app.services.document_service import DocumentService

        self._instances["document_service"] = DocumentService(
            document_repo=self.get_document_repository(),
            chunk_repo=self.get_chunk_repository(),
            library_repo=self.get_library_repository(),
        )
        return self._instances["document_service"]

    def get_index_service(self):
        """Get the singleton index service instance"""
        if "index_service" in self._instances:
            return self._instances["index_service"]

        from app.services.index_service import IndexService

        self._instances["index_service"] = IndexService()
        return self._instances["index_service"]


# Global container instance 
container = Container()
