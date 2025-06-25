"""In-memory repository implementations with async concurrency control"""

import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any, Type, TypeVar
from app.repositories.base_repository import BaseRepository
from app.utils.concurrency import lock_manager

T = TypeVar("T")
from app.models.library import Library, LibraryCreate, LibraryUpdate
from app.models.chunk import Chunk, ChunkCreate, ChunkUpdate
from app.models.document import Document, DocumentCreate, DocumentUpdate


class MemoryRepository(BaseRepository[T]):
    def __init__(self, entity_type: Type[T]):
        self._data: Dict[str, T] = {}
        self._entity_type = entity_type

    def _get_resource_id(self, operation: str, entity_id: Optional[str] = None) -> str:
        """Generate resource ID for locking"""
        base_id = f"{self._entity_type.__name__}"
        if entity_id:
            return f"{base_id}:{entity_id}:{operation}"
        return f"{base_id}:{operation}"

    async def create(self, entity: T) -> T:
        """Create a new entity with write lock"""
        entity_id = str(uuid.uuid4())
        resource_id = self._get_resource_id("create", entity_id)

        async with await lock_manager.write_lock(resource_id):
            # Set the ID on the entity
            if hasattr(entity, "id"):
                setattr(entity, "id", entity_id)

            # Set timestamps if applicable
            now = datetime.utcnow()
            if hasattr(entity, "created_at"):
                setattr(entity, "created_at", now)
            if hasattr(entity, "updated_at"):
                setattr(entity, "updated_at", now)

            self._data[entity_id] = entity
            return entity

    async def get_by_id(self, entity_id: str) -> Optional[T]:
        """Get an entity by its ID with read lock"""
        resource_id = self._get_resource_id("read", entity_id)

        async with await lock_manager.read_lock(resource_id):
            return self._data.get(entity_id)

    async def update(self, entity_id: str, updates: Dict[str, Any]) -> Optional[T]:
        """Update an entity with write lock"""
        resource_id = self._get_resource_id("update", entity_id)

        async with await lock_manager.write_lock(resource_id):
            entity = self._data.get(entity_id)
            if not entity:
                return None

            # Update fields
            for field, value in updates.items():
                if hasattr(entity, field):
                    setattr(entity, field, value)

            # Update timestamp
            if hasattr(entity, "updated_at"):
                setattr(entity, "updated_at", datetime.utcnow())

            self._data[entity_id] = entity
            return entity

    async def delete(self, entity_id: str) -> bool:
        """Delete an entity by its ID with write lock"""
        resource_id = self._get_resource_id("delete", entity_id)

        async with await lock_manager.write_lock(resource_id):
            if entity_id in self._data:
                del self._data[entity_id]
                return True
            return False

    async def list_all(self, skip: int = 0, limit: int = 100) -> List[T]:
        """List all entities with read lock"""
        resource_id = self._get_resource_id("list")

        async with await lock_manager.read_lock(resource_id):
            entities = list(self._data.values())
            return entities[skip : skip + limit]

    async def count(self) -> int:
        """Count total number of entities with read lock"""
        resource_id = self._get_resource_id("count")

        async with await lock_manager.read_lock(resource_id):
            return len(self._data)

    async def exists(self, entity_id: str) -> bool:
        """Check if an entity exists with read lock"""
        resource_id = self._get_resource_id("exists", entity_id)

        async with await lock_manager.read_lock(resource_id):
            return entity_id in self._data


class LibraryRepository(MemoryRepository[Library]):
    """Repository for library entities with concurrency control"""

    def __init__(self):
        super().__init__(Library)

    async def create_from_request(self, library_create: LibraryCreate) -> Library:
        """Create a library from a create request"""
        library = Library(
            id="",  # this will be set by create method
            name=library_create.name,
            description=library_create.description,
            index_type=library_create.index_type,
            similarity_metric=library_create.similarity_metric,
            embedding_dimension=library_create.embedding_dimension,
            lsh_config=library_create.lsh_config,
            metadata=library_create.metadata,
        )
        return await self.create(library)

    async def update_from_request(
        self, library_id: str, library_update: LibraryUpdate
    ) -> Optional[Library]:
        """Update a library from an update request"""
        updates = {}
        if library_update.name is not None:
            updates["name"] = library_update.name
        if library_update.description is not None:
            updates["description"] = library_update.description
        if library_update.index_type is not None:
            updates["index_type"] = library_update.index_type
        if library_update.similarity_metric is not None:
            updates["similarity_metric"] = library_update.similarity_metric
        if library_update.lsh_config is not None:
            updates["lsh_config"] = library_update.lsh_config
        if library_update.metadata is not None:
            updates["metadata"] = library_update.metadata

        return await self.update(library_id, updates)


class ChunkRepository(MemoryRepository[Chunk]):
    """Repository for chunk entities with concurrency control"""

    def __init__(self):
        super().__init__(Chunk)

    async def get_by_library_id(
        self, library_id: str, skip: int = 0, limit: int = 100
    ) -> List[Chunk]:
        """Get chunks by library ID with read lock"""
        resource_id = self._get_resource_id("get_by_library", library_id)

        async with await lock_manager.read_lock(resource_id):
            chunks = [
                chunk for chunk in self._data.values() if chunk.library_id == library_id
            ]
            return chunks[skip : skip + limit]

    async def get_by_document_id(
        self, document_id: str, skip: int = 0, limit: int = 100
    ) -> List[Chunk]:
        """Get chunks by document ID with read lock"""
        resource_id = self._get_resource_id("get_by_document", document_id)

        async with await lock_manager.read_lock(resource_id):
            chunks = [
                chunk
                for chunk in self._data.values()
                if chunk.document_id == document_id
            ]
            return chunks[skip : skip + limit]

    async def count_by_library_id(self, library_id: str) -> int:
        """Count chunks by library ID with read lock"""
        resource_id = self._get_resource_id("count_by_library", library_id)

        async with await lock_manager.read_lock(resource_id):
            return len(
                [
                    chunk
                    for chunk in self._data.values()
                    if chunk.library_id == library_id
                ]
            )

    async def count_by_document_id(self, document_id: str) -> int:
        """Count chunks by document ID with read lock"""
        resource_id = self._get_resource_id("count_by_document", document_id)

        async with await lock_manager.read_lock(resource_id):
            return len(
                [
                    chunk
                    for chunk in self._data.values()
                    if chunk.document_id == document_id
                ]
            )

    async def delete_by_library_id(self, library_id: str) -> int:
        """Delete all chunks in a library with write lock"""
        resource_id = self._get_resource_id("delete_by_library", library_id)

        async with await lock_manager.write_lock(resource_id):
            chunks_to_delete = [
                chunk_id
                for chunk_id, chunk in self._data.items()
                if chunk.library_id == library_id
            ]

            for chunk_id in chunks_to_delete:
                del self._data[chunk_id]

            return len(chunks_to_delete)

    async def delete_by_document_id(self, document_id: str) -> int:
        """Delete all chunks in a document with write lock"""
        resource_id = self._get_resource_id("delete_by_document", document_id)

        async with await lock_manager.write_lock(resource_id):
            chunks_to_delete = [
                chunk_id
                for chunk_id, chunk in self._data.items()
                if chunk.document_id == document_id
            ]

            for chunk_id in chunks_to_delete:
                del self._data[chunk_id]

            return len(chunks_to_delete)


class DocumentRepository(MemoryRepository[Document]):
    """Repository for document entities with concurrency control"""

    def __init__(self):
        super().__init__(Document)

    async def create_from_request(
        self, document_create: DocumentCreate, library_id: str
    ) -> Document:
        """Create a document from a create request"""
        document = Document(
            id="",  # this will be set by create method
            title=document_create.title,
            content=document_create.content,
            source=document_create.source,
            author=document_create.author,
            metadata=document_create.metadata,
            library_id=library_id,
        )
        return await self.create(document)

    async def update_from_request(
        self, document_id: str, document_update: DocumentUpdate
    ) -> Optional[Document]:
        """Update a document from an update request"""
        updates = {}
        if document_update.title is not None:
            updates["title"] = document_update.title
        if document_update.content is not None:
            updates["content"] = document_update.content
        if document_update.source is not None:
            updates["source"] = document_update.source
        if document_update.author is not None:
            updates["author"] = document_update.author
        if document_update.metadata is not None:
            updates["metadata"] = document_update.metadata

        return await self.update(document_id, updates)

    async def get_by_library_id(
        self, library_id: str, skip: int = 0, limit: int = 100
    ) -> List[Document]:
        """Get documents by library ID with read lock"""
        resource_id = self._get_resource_id("get_by_library", library_id)

        async with await lock_manager.read_lock(resource_id):
            documents = [
                doc for doc in self._data.values() if doc.library_id == library_id
            ]
            return documents[skip : skip + limit]

    async def count_by_library_id(self, library_id: str) -> int:
        """Count documents by library ID with read lock"""
        resource_id = self._get_resource_id("count_by_library", library_id)

        async with await lock_manager.read_lock(resource_id):
            return len(
                [doc for doc in self._data.values() if doc.library_id == library_id]
            )

    async def delete_by_library_id(self, library_id: str) -> int:
        """Delete all documents in a library with write lock"""
        resource_id = self._get_resource_id("delete_by_library", library_id)

        async with await lock_manager.write_lock(resource_id):
            docs_to_delete = [
                doc_id
                for doc_id, doc in self._data.items()
                if doc.library_id == library_id
            ]

            for doc_id in docs_to_delete:
                del self._data[doc_id]

            return len(docs_to_delete)
