"""Base vector index interface"""

from abc import ABC, abstractmethod
from typing import List, Tuple, Dict, Any, Optional
import numpy as np
from app.core.constants import SimilarityMetric


class BaseVectorIndex(ABC):
    def __init__(self, similarity_metric: SimilarityMetric):
        self.similarity_metric = similarity_metric

    async def add_vector(
        self,
        vector_id: str,
        vector: List[float],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Add a vector to the index (async interface)"""
        vector_array = np.array(vector, dtype=np.float32)
        self.add_vector_sync(vector_id, vector_array, metadata or {})

    async def remove_vector(self, vector_id: str) -> bool:
        """Remove a vector from the index (async interface)"""

        self.remove_vector_sync(vector_id)
        return True

    async def update_vector(
        self,
        vector_id: str,
        vector: List[float],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Update a vector in the index (async interface)"""
    # Default implementation calls sync version for backward compatibility
        vector_array = np.array(vector, dtype=np.float32)
        self.update_vector_sync(vector_id, vector_array, metadata or {})
        return True

    async def search(
        self, query_vector: List[float], k: int = 10, **kwargs
    ) -> List[Tuple[str, float]]:
        """Search for k nearest neighbors (async interface)"""

        query_array = np.array(query_vector, dtype=np.float32)
        results = self.search_knn(query_array, k)
    # Convert from (id, score, metadata) to (id, score)
        return [(result[0], result[1]) for result in results]

    async def get_vector_count(self) -> int:
        """Get the number of vectors in the index (async interface)"""
    # Default implementation
        return 0

    async def clear(self) -> None:
        """Clear all vectors from the index (async interface)."""
    # Default implementation - subclasses should override
        pass

# Legacy sync interface (for backward compatibility)
    def add_vector_sync(
        self, vector_id: str, vector: np.ndarray, metadata: Dict[str, Any]
    ):
        """Add a vector to the index (sync interface)"""
        pass

    @abstractmethod
    def search_knn(
        self, query_vector: np.ndarray, k: int
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        """Search for k nearest neighbors. Returns list of (vector_id, similarity_score, metadata)"""
        pass

    def remove_vector_sync(self, vector_id: str):
        """Remove a vector from the index (sync interface)"""
        pass

    def update_vector_sync(
        self, vector_id: str, vector: np.ndarray, metadata: Dict[str, Any]
    ):
        """Update a vector in the index (sync interface)"""
        pass

    def _calculate_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate similarity between two vectors based on the configured metric"""
        if self.similarity_metric == SimilarityMetric.COSINE:
            return self._cosine_similarity(vec1, vec2)
        elif self.similarity_metric == SimilarityMetric.EUCLIDEAN:
            return self._euclidean_similarity(vec1, vec2)
        elif self.similarity_metric == SimilarityMetric.DOT_PRODUCT:
            return self._dot_product_similarity(vec1, vec2)
        else:
            raise ValueError(f"Unsupported similarity metric: {self.similarity_metric}")

    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors"""
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return dot_product / (norm1 * norm2)

    def _euclidean_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate Euclidean similarity (1 / (1 + distance)) between two vectors"""
        distance = np.linalg.norm(vec1 - vec2)
        return 1.0 / (1.0 + distance)

    def _dot_product_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate dot product similarity between two vectors"""
        return np.dot(vec1, vec2)
