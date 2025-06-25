"""Index service for vector indexing and search operations"""

from typing import List, Optional, Dict, Any, Union
import numpy as np
from app.models.chunk import SearchResult, ChunkResponse, ChunkMetadata
from app.core.constants import IndexType, SimilarityMetric
from app.core.exceptions import LibraryNotFound, IndexingError
from app.indexing.linear_index import LinearIndex
from app.indexing.kdtree_index import KDTreeIndex
from app.indexing.lsh_index import LSHIndex, LSHParams
from app.utils.embeddings import embedding_service
from app.utils.metadata_filters import MetadataFilterEngine, MetadataFilter


class IndexService:
    def __init__(self):
        self._indexes: Dict[str, Any] = {}  # library_id -> index instance
        self._metadata_filter_engine = MetadataFilterEngine()

    async def create_index(
        self,
        library_id: str,
        index_type: IndexType,
        similarity_metric: SimilarityMetric,
        **kwargs,
    ):
        """Create a new vector index for a library"""
        try:
            if index_type == IndexType.LINEAR:
                index = LinearIndex(similarity_metric)
            elif index_type == IndexType.KD_TREE:
                index = KDTreeIndex(similarity_metric)
            elif index_type == IndexType.LSH:
            # Extract LSH-specific parameters from kwargs
                lsh_params = None
                if "lsh_config" in kwargs and kwargs["lsh_config"] is not None:
                    config = kwargs["lsh_config"]
                # Use the new conversion method that handles both old and new field names
                    lsh_params = LSHParams.from_config(config)
                index = LSHIndex(similarity_metric, lsh_params)
            else:
            # Default to linear index for unsupported types
                index = LinearIndex(similarity_metric)

            self._indexes[library_id] = index
        except Exception as e:
            raise IndexingError(f"Failed to create index: {str(e)}", "create_index")

    async def delete_index(self, library_id: str):
        """Delete a vector index for a library"""
        if library_id in self._indexes:
            del self._indexes[library_id]

    async def rebuild_index(
        self,
        library_id: str,
        index_type: IndexType,
        similarity_metric: SimilarityMetric,
        **kwargs,
    ):
        """Rebuild a vector index for a library"""
        await self.delete_index(library_id)
        await self.create_index(library_id, index_type, similarity_metric, **kwargs)

    async def add_vector(
        self,
        library_id: str,
        vector_id: str,
        vector: List[float],
        metadata: Dict[str, Any],
    ):
        """Add a vector to the index"""
        if library_id not in self._indexes:
            raise LibraryNotFound(library_id)

        try:
            index = self._indexes[library_id]
            await index.add_vector(vector_id, vector, metadata)
        except Exception as e:
            raise IndexingError(f"Failed to add vector: {str(e)}", "add_vector")

    async def update_vector(
        self,
        library_id: str,
        vector_id: str,
        vector: List[float],
        metadata: Dict[str, Any],
    ):
        """Update a vector in the index"""
        if library_id not in self._indexes:
            raise LibraryNotFound(library_id)

        try:
            index = self._indexes[library_id]
            await index.update_vector(vector_id, vector, metadata)
        except Exception as e:
            raise IndexingError(f"Failed to update vector: {str(e)}", "update_vector")

    async def remove_vector(self, library_id: str, vector_id: str):
        """Remove a vector from the index"""
        if library_id not in self._indexes:
            raise LibraryNotFound(library_id)

        try:
            index = self._indexes[library_id]
            await index.remove_vector(vector_id)
        except Exception as e:
            raise IndexingError(f"Failed to remove vector: {str(e)}", "remove_vector")

    async def search(
        self,
        library_id: str,
        query_text: str,
        k: int = 10,
        similarity_threshold: Optional[float] = None,
        metadata_filters: Optional[List[Any]] = None,
        filter_mode: str = "and",
    ) -> List[SearchResult]:
        """Search for similar vectors in a library"""
        try:
        # Check if index exists
            if library_id not in self._indexes:
                raise LibraryNotFound(library_id)

        # Generate query embedding
            query_embedding = await embedding_service.generate_query_embedding(
                query_text
            )

        # Get index and search
            index = self._indexes[library_id]
            raw_results = await index.search(query_embedding, k)

        # Return empty if no results
            if not raw_results:
                return []

            results = []

            for i, (vector_id, similarity_score) in enumerate(raw_results):
            # Skip if similarity threshold not met
                if (
                    similarity_threshold is not None
                    and similarity_score < similarity_threshold
                ):
                    continue

            # Get vector metadata
                if hasattr(index, "_metadata") and vector_id in index._metadata:
                    metadata = index._metadata[vector_id]
                else:
                    continue  # Skip if no metadata available

            # Skip if metadata filters don't match
                if not self._metadata_filters_match(
                    metadata, metadata_filters, filter_mode
                ):
                    continue

            # Get embedding vector for response
                vector_embedding = []
                if hasattr(index, "_vectors") and vector_id in index._vectors:
                    vector_embedding = index._vectors[vector_id].tolist()

            # Create chunk metadata with proper datetime handling
                from datetime import datetime, timezone

            # Handle datetime fields with proper defaults
                created_at = metadata.get("created_at")
                if created_at is None:
                    created_at = datetime.now(timezone.utc)
                elif isinstance(created_at, str):
                # Parse ISO format datetime string
                    try:
                        created_at = datetime.fromisoformat(
                            created_at.replace("Z", "+00:00")
                        )
                    except ValueError:
                        created_at = datetime.now(timezone.utc)

                updated_at = metadata.get("updated_at")
                if updated_at is None:
                    updated_at = datetime.now(timezone.utc)
                elif isinstance(updated_at, str):
             
                    try:
                        updated_at = datetime.fromisoformat(
                            updated_at.replace("Z", "+00:00")
                        )
                    except ValueError:
                        updated_at = datetime.now(timezone.utc)

                chunk_metadata = ChunkMetadata(
                    created_at=created_at,
                    updated_at=updated_at,
                    source=metadata.get("source"),
                    page_number=metadata.get("page_number"),
                    section=metadata.get("section"),
                    custom_fields=metadata.get("custom_fields", {}),
                )

            # Create chunk response
                chunk_response = ChunkResponse(
                    id=vector_id,
                    text=metadata.get("text", ""),
                    embedding=vector_embedding,
                    metadata=chunk_metadata,
                    document_id=metadata.get("document_id"),
                    library_id=library_id,
                )

                search_result = SearchResult(
                    chunk=chunk_response,
                    similarity_score=float(similarity_score),
                    rank=i + 1,
                )

                results.append(search_result)

            return results

        except Exception as e:
            raise IndexingError(f"Search failed: {str(e)}", "search")

    def _metadata_filters_match(
        self, metadata: Dict[str, Any], filters: Optional[List[Any]], filter_mode: str
    ) -> bool:
        """Check if metadata matches the provided filters using advanced metadata filter engine"""
    # Return if no filters
        if not filters:
            return True

    # Convert filter objects to MetadataFilter instances
        converted_filters = []
        for filter_obj in filters:
        # Handle both dict and object formats
            if hasattr(filter_obj, "field"):
                field = filter_obj.field
                operator = filter_obj.operator
                value = filter_obj.value
            else:
                field = filter_obj.get("field")
                operator = filter_obj.get("operator")
                value = filter_obj.get("value")

        # Skip if invalid filter
            if not field or not operator:
                continue

        # Create MetadataFilter instance
            metadata_filter = MetadataFilter(
                field=field, operator=operator, value=value
            )
            converted_filters.append(metadata_filter)

    # Use advanced metadata filter engine
        return self._metadata_filter_engine.apply_filters(
            metadata=metadata, filters=converted_filters, mode=filter_mode
        )
