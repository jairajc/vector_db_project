"""File-based repository implementations for disk persistence"""

import json
import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Type, TypeVar
from app.repositories.base_repository import BaseRepository
from app.core.config import settings

T = TypeVar("T")
from app.models.library import Library, LibraryCreate, LibraryUpdate
from app.models.chunk import Chunk, ChunkCreate, ChunkUpdate
from app.models.document import Document, DocumentCreate, DocumentUpdate


class FileRepository(BaseRepository[T]):
    def __init__(self, entity_type: Type[T], data_dir: str = "data"):
        self._entity_type = entity_type
        self._data_dir = Path(data_dir)
        self._data_dir.mkdir(exist_ok=True)

    # Create subdirectory for this entity type
        self._entity_dir = self._data_dir / entity_type.__name__.lower()
        self._entity_dir.mkdir(exist_ok=True)

    # Index file for quick lookups
        self._index_file = self._entity_dir / "index.json"
        self._index: Dict[str, str] = self._load_index()

    def _load_index(self) -> Dict[str, str]:
        """Load the index file"""
        if self._index_file.exists():
            try:
                with open(self._index_file, "r") as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                return {}
        return {}

    def _save_index(self):
        """Save the index file"""
        try:
            with open(self._index_file, "w") as f:
                json.dump(self._index, f, indent=2)
        except IOError as e:
            raise RuntimeError(f"Failed to save index: {e}")

    def _get_entity_file(self, entity_id: str) -> Path:
        """Get the file path for an entity"""
        return self._entity_dir / f"{entity_id}.json"

    def _serialize_entity(self, entity: T) -> Dict[str, Any]:
        """Serialize entity to dictionary"""
        if hasattr(entity, "dict"):
            data = entity.dict()
        elif hasattr(entity, "__dict__"):
            data = entity.__dict__.copy()
        else:
            raise ValueError(f"Cannot serialize entity of type {type(entity)}")

    # Recursively convert datetime objects to ISO strings
        return self._convert_datetimes_to_strings(data)

    def _convert_datetimes_to_strings(self, data: Any) -> Any:
        if isinstance(data, datetime):
            return data.isoformat()
        elif isinstance(data, dict):
            return {
                key: self._convert_datetimes_to_strings(value)
                for key, value in data.items()
            }
        elif isinstance(data, list):
            return [self._convert_datetimes_to_strings(item) for item in data]
        else:
            return data

    def _deserialize_entity(self, data: Dict[str, Any]) -> T:
        """Deserialize dictionary to entity"""
        converted_data = self._convert_strings_to_datetimes(data)

        return self._entity_type(**converted_data)

    def _convert_strings_to_datetimes(self, data: Any) -> Any:
        if isinstance(data, str) and self._looks_like_datetime(data):
            try:
                return datetime.fromisoformat(data)
            except ValueError:
                return data
        elif isinstance(data, dict):
            result = {}
            for key, value in data.items():
                if key in ["created_at", "updated_at"] and isinstance(value, str):
                    try:
                        result[key] = datetime.fromisoformat(value)
                    except ValueError:
                        result[key] = value
                else:
                    result[key] = self._convert_strings_to_datetimes(value)
            return result
        elif isinstance(data, list):
            return [self._convert_strings_to_datetimes(item) for item in data]
        else:
            return data

    def _looks_like_datetime(self, value: str) -> bool:
        if not isinstance(value, str) or len(value) < 10:
            return False
    # Check for ISO datetime format
        return "T" in value or ("-" in value and ":" in value)

    async def create(self, entity: T) -> T:
        """Create a new entity"""
        entity_id = str(uuid.uuid4())

    # Set the ID on the entity
        if hasattr(entity, "id"):
            setattr(entity, "id", entity_id)

    # Set timestamps if applicable
        now = datetime.utcnow()
        if hasattr(entity, "created_at"):
            setattr(entity, "created_at", now)
        if hasattr(entity, "updated_at"):
            setattr(entity, "updated_at", now)

    # Save to file
        entity_file = self._get_entity_file(entity_id)
        entity_data = self._serialize_entity(entity)

        try:
            with open(entity_file, "w") as f:
                json.dump(entity_data, f, indent=2)

        # Update index
            self._index[entity_id] = entity_file.name
            self._save_index()

            return entity
        except IOError as e:
            raise RuntimeError(f"Failed to create entity: {e}")

    async def get_by_id(self, entity_id: str) -> Optional[T]:
        """Get an entity by its ID"""
        if entity_id not in self._index:
            return None

        entity_file = self._get_entity_file(entity_id)
        if not entity_file.exists():
        # Remove from index if file doesn't exist
            del self._index[entity_id]
            self._save_index()
            return None

        try:
            with open(entity_file, "r") as f:
                entity_data = json.load(f)
            return self._deserialize_entity(entity_data)
        except (json.JSONDecodeError, IOError):
            return None

    async def update(self, entity_id: str, updates: Dict[str, Any]) -> Optional[T]:
        """Update an entity with the given updates"""
        entity = await self.get_by_id(entity_id)
        if not entity:
            return None

    # Update fields
        for field, value in updates.items():
            if hasattr(entity, field):
                setattr(entity, field, value)

    # Update timestamp
        if hasattr(entity, "updated_at"):
            setattr(entity, "updated_at", datetime.utcnow())

    # Save to file
        entity_file = self._get_entity_file(entity_id)
        entity_data = self._serialize_entity(entity)

        try:
            with open(entity_file, "w") as f:
                json.dump(entity_data, f, indent=2)
            return entity
        except IOError as e:
            raise RuntimeError(f"Failed to update entity: {e}")

    async def delete(self, entity_id: str) -> bool:
        """Delete an entity by its ID"""
        if entity_id not in self._index:
            return False

        entity_file = self._get_entity_file(entity_id)

        try:
            if entity_file.exists():
                entity_file.unlink()

        # Remove from index
            del self._index[entity_id]
            self._save_index()
            return True
        except IOError:
            return False

    async def list_all(self, skip: int = 0, limit: int = 100) -> List[T]:
        """List all entities with pagination"""
        entities = []
        entity_ids = list(self._index.keys())[skip : skip + limit]

        for entity_id in entity_ids:
            entity = await self.get_by_id(entity_id)
            if entity:
                entities.append(entity)

        return entities

    async def count(self) -> int:
        """Count total number of entities"""
        return len(self._index)

    async def exists(self, entity_id: str) -> bool:
        """Check if an entity exists by its ID"""
        return entity_id in self._index and self._get_entity_file(entity_id).exists()


class FileLibraryRepository(FileRepository[Library]):
    """File based repository for library entities"""

    def __init__(self, data_dir: str = "data"):
        super().__init__(Library, data_dir)

    async def create_from_request(self, library_create: LibraryCreate) -> Library:
        """Create a library from a create request"""
        library = Library(
            id="",  # Will be set by create method
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


class FileChunkRepository(FileRepository[Chunk]):
    """File-based repository for chunk entities"""

    def __init__(self, data_dir: str = "data"):
        super().__init__(Chunk, data_dir)
    # Secondary indexes for efficient queries
        self._library_index_file = self._entity_dir / "library_index.json"
        self._document_index_file = self._entity_dir / "document_index.json"
        self._library_index: Dict[str, List[str]] = self._load_secondary_index(
            self._library_index_file
        )
        self._document_index: Dict[str, List[str]] = self._load_secondary_index(
            self._document_index_file
        )

    def _load_secondary_index(self, index_file: Path) -> Dict[str, List[str]]:
        """Load a secondary index file"""
        if index_file.exists():
            try:
                with open(index_file, "r") as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                return {}
        return {}

    def _save_secondary_indexes(self):
        """Save secondary index files"""
        try:
            with open(self._library_index_file, "w") as f:
                json.dump(self._library_index, f, indent=2)
            with open(self._document_index_file, "w") as f:
                json.dump(self._document_index, f, indent=2)
        except IOError as e:
            raise RuntimeError(f"Failed to save secondary indexes: {e}")

    async def create(self, entity: Chunk) -> Chunk:
        """Create a new chunk with secondary index updates"""
        result = await super().create(entity)

    # Update secondary indexes
        library_id = entity.library_id
        document_id = entity.document_id
        chunk_id = entity.id

        if library_id not in self._library_index:
            self._library_index[library_id] = []
        self._library_index[library_id].append(chunk_id)

        if document_id and document_id not in self._document_index:
            self._document_index[document_id] = []
        if document_id:
            self._document_index[document_id].append(chunk_id)

        self._save_secondary_indexes()
        return result

    async def delete(self, entity_id: str) -> bool:
        """Delete a chunk with secondary index updates"""
        chunk = await self.get_by_id(entity_id)
        if not chunk:
            return False

        result = await super().delete(entity_id)

        if result:
        # Update secondary indexes
            library_id = chunk.library_id
            document_id = chunk.document_id

            if library_id in self._library_index:
                self._library_index[library_id] = [
                    cid for cid in self._library_index[library_id] if cid != entity_id
                ]
                if not self._library_index[library_id]:
                    del self._library_index[library_id]

            if document_id and document_id in self._document_index:
                self._document_index[document_id] = [
                    cid for cid in self._document_index[document_id] if cid != entity_id
                ]
                if not self._document_index[document_id]:
                    del self._document_index[document_id]

            self._save_secondary_indexes()

        return result

    async def get_by_library_id(
        self, library_id: str, skip: int = 0, limit: int = 100
    ) -> List[Chunk]:
        """Get chunks by library ID using secondary index"""
        chunk_ids = self._library_index.get(library_id, [])[skip : skip + limit]
        chunks = []

        for chunk_id in chunk_ids:
            chunk = await self.get_by_id(chunk_id)
            if chunk:
                chunks.append(chunk)

        return chunks

    async def get_by_document_id(
        self, document_id: str, skip: int = 0, limit: int = 100
    ) -> List[Chunk]:
        """Get chunks by document ID using secondary index"""
        chunk_ids = self._document_index.get(document_id, [])[skip : skip + limit]
        chunks = []

        for chunk_id in chunk_ids:
            chunk = await self.get_by_id(chunk_id)
            if chunk:
                chunks.append(chunk)

        return chunks

    async def count_by_library_id(self, library_id: str) -> int:
        """Count chunks by library ID using secondary index"""
        return len(self._library_index.get(library_id, []))

    async def count_by_document_id(self, document_id: str) -> int:
        """Count chunks by document ID using secondary index"""
        return len(self._document_index.get(document_id, []))

    async def delete_by_library_id(self, library_id: str) -> int:
        """Delete all chunks in a library"""
        chunk_ids = self._library_index.get(library_id, []).copy()
        deleted_count = 0

        for chunk_id in chunk_ids:
            if await self.delete(chunk_id):
                deleted_count += 1

        return deleted_count

    async def delete_by_document_id(self, document_id: str) -> int:
        """Delete all chunks in a document"""
        chunk_ids = self._document_index.get(document_id, []).copy()
        deleted_count = 0

        for chunk_id in chunk_ids:
            if await self.delete(chunk_id):
                deleted_count += 1

        return deleted_count


class FileDocumentRepository(FileRepository[Document]):
    """File-based repository for document entities"""

    def __init__(self, data_dir: str = "data"):
        super().__init__(Document, data_dir)
        self._library_index_file = self._entity_dir / "library_index.json"
        self._library_index: Dict[str, List[str]] = self._load_secondary_index(
            self._library_index_file
        )

    def _load_secondary_index(self, index_file: Path) -> Dict[str, List[str]]:
        """Load a secondary index file"""
        if index_file.exists():
            try:
                with open(index_file, "r") as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                return {}
        return {}

    def _save_secondary_index(self):
        """Save secondary index file"""
        try:
            with open(self._library_index_file, "w") as f:
                json.dump(self._library_index, f, indent=2)
        except IOError as e:
            raise RuntimeError(f"Failed to save secondary index: {e}")

    async def create(self, entity: Document) -> Document:
        """Create a new document with secondary index updates"""
        result = await super().create(entity)

    # Update secondary index
        library_id = entity.library_id
        document_id = entity.id

        if library_id not in self._library_index:
            self._library_index[library_id] = []
        self._library_index[library_id].append(document_id)

        self._save_secondary_index()
        return result

    async def delete(self, entity_id: str) -> bool:
        """Delete a document with secondary index updates"""
        document = await self.get_by_id(entity_id)
        if not document:
            return False

        result = await super().delete(entity_id)

        if result:
    
            library_id = document.library_id

            if library_id in self._library_index:
                self._library_index[library_id] = [
                    did for did in self._library_index[library_id] if did != entity_id
                ]
                if not self._library_index[library_id]:
                    del self._library_index[library_id]

            self._save_secondary_index()

        return result

    async def create_from_request(
        self, document_create: DocumentCreate, library_id: str
    ) -> Document:
        """Create a document from a create request"""
        document = Document(
            id="",
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
        """Get documents by library ID using secondary index"""
        document_ids = self._library_index.get(library_id, [])[skip : skip + limit]
        documents = []

        for document_id in document_ids:
            document = await self.get_by_id(document_id)
            if document:
                documents.append(document)

        return documents

    async def count_by_library_id(self, library_id: str) -> int:
        """Count documents by library ID using secondary index"""
        return len(self._library_index.get(library_id, []))

    async def delete_by_library_id(self, library_id: str) -> int:
        """Delete all documents in a library"""
        document_ids = self._library_index.get(library_id, []).copy()
        deleted_count = 0

        for document_id in document_ids:
            if await self.delete(document_id):
                deleted_count += 1

        return deleted_count
