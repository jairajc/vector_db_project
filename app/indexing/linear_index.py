from typing import List, Tuple, Dict, Any, Optional
import numpy as np
from app.indexing.base_index import BaseVectorIndex
from app.core.constants import SimilarityMetric
from app.utils.concurrency import lock_manager


class LinearIndex(BaseVectorIndex):

    def __init__(self, similarity_metric: SimilarityMetric):
        super().__init__(similarity_metric)
        self._vectors: Dict[str, np.ndarray] = {}
        self._metadata: Dict[str, Dict[str, Any]] = {}
        self._index_id = f"linear_index_{id(self)}"

    def _get_resource_id(self, operation: str, vector_id: Optional[str] = None) -> str:
        """Generate resource ID for locking"""
        if vector_id:
            return f"{self._index_id}:{vector_id}:{operation}"
        return f"{self._index_id}:{operation}"

    async def add_vector(
        self,
        vector_id: str,
        vector: List[float],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Add a vector to the index with write lock"""
        resource_id = self._get_resource_id("add", vector_id)

        async with await lock_manager.write_lock(resource_id):
            vector_array = np.array(vector, dtype=np.float32)

            # Normalize vector if using cosine similarity
            if self.similarity_metric == SimilarityMetric.COSINE:
                norm = np.linalg.norm(vector_array)
                if norm > 0:
                    vector_array = vector_array / norm

            self._vectors[vector_id] = vector_array.copy()
            self._metadata[vector_id] = (metadata or {}).copy()

    async def remove_vector(self, vector_id: str) -> bool:
        """Remove a vector from the index with write lock"""
        resource_id = self._get_resource_id("remove", vector_id)

        async with await lock_manager.write_lock(resource_id):
            if vector_id in self._vectors:
                del self._vectors[vector_id]
                del self._metadata[vector_id]
                return True
            return False

    async def update_vector(
        self,
        vector_id: str,
        vector: List[float],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Update a vector in the index with write lock"""
        resource_id = self._get_resource_id("update", vector_id)

        async with await lock_manager.write_lock(resource_id):
            if vector_id in self._vectors:
                vector_array = np.array(vector, dtype=np.float32)

                if self.similarity_metric == SimilarityMetric.COSINE:
                    norm = np.linalg.norm(vector_array)
                    if norm > 0:
                        vector_array = vector_array / norm

                self._vectors[vector_id] = vector_array.copy()
                self._metadata[vector_id] = (metadata or {}).copy()
                return True
            return False

    async def search(
        self, query_vector: List[float], k: int = 10, **kwargs
    ) -> List[Tuple[str, float]]:
        """Search for k nearest neighbors using linear search with read lock"""
        resource_id = self._get_resource_id("search")

        async with await lock_manager.read_lock(resource_id):
            if not self._vectors:
                return []

            query_array = np.array(query_vector, dtype=np.float32)

            if self.similarity_metric == SimilarityMetric.COSINE:
                norm = np.linalg.norm(query_array)
                if norm > 0:
                    query_array = query_array / norm

        # Calculate similarities for all vectors
            similarities = []
            for vector_id, vector in self._vectors.items():
                similarity = self._calculate_similarity(query_array, vector)
                similarities.append((vector_id, similarity))

        # Sort by similarity (descending order)
            similarities.sort(key=lambda x: x[1], reverse=True)

        # Return top k results
            return similarities[:k]

    async def get_vector_count(self) -> int:
        """Get the number of vectors in the index with read lock"""
        resource_id = self._get_resource_id("count")

        async with await lock_manager.read_lock(resource_id):
            return len(self._vectors)

    async def clear(self) -> None:
        """Clear all vectors from the index with write lock"""
        resource_id = self._get_resource_id("clear")

        async with await lock_manager.write_lock(resource_id):
            self._vectors.clear()
            self._metadata.clear()

    def get_vector_ids(self) -> List[str]:
        """Get all vector IDs in the index"""
        return list(self._vectors.keys())

    def add_vector_sync(
        self, vector_id: str, vector: np.ndarray, metadata: Dict[str, Any]
    ):
        """Add a vector to the index (synchronous version with basic safety)"""

        if self.similarity_metric == SimilarityMetric.COSINE:
            norm = np.linalg.norm(vector)
            if norm > 0:
                vector = vector / norm

        self._vectors[vector_id] = vector.copy()
        self._metadata[vector_id] = metadata.copy()

    def search_knn(
        self, query_vector: np.ndarray, k: int
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        """Search for k nearest neighbors using linear search (legacy method)"""
        if not self._vectors:
            return []

        if self.similarity_metric == SimilarityMetric.COSINE:
            norm = np.linalg.norm(query_vector)
            if norm > 0:
                query_vector = query_vector / norm

    # Calculate similarities for all vectors
        similarities = []
        for vector_id, vector in self._vectors.items():
            similarity = self._calculate_similarity(query_vector, vector)
            similarities.append((vector_id, similarity, self._metadata[vector_id]))

        similarities.sort(key=lambda x: x[1], reverse=True)

    # Return top k results
        return similarities[:k]
