"""LSH (Locality Sensitive Hashing) based vector index implementation with async concurrency control"""

import numpy as np
import math
from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass
from app.indexing.base_index import BaseVectorIndex
from app.core.constants import SimilarityMetric
from app.utils.concurrency import lock_manager


@dataclass
class LSHParams:
    """LSH configuration parameters"""

    num_hash_tables: int = 10  # Number of hash tables (L)
    num_hash_functions: int = 8  # Number of hash functions per table (k)
    hash_width: float = 1.0  # Hash bucket width (for Euclidean)
    random_seed: int = 42  # For reproducible hash functions

    @classmethod
    def from_config(cls, config):
        """Create LSHParams from LSHConfig model"""
        if hasattr(config, "hash_length"):
            # Convert from new API model format
            return cls(
                num_hash_tables=config.num_hash_tables,
                num_hash_functions=config.hash_length,
                hash_width=config.hash_width,
                random_seed=config.random_seed,
            )
        else:
            # Direct instantiation or old format
            return cls(
                num_hash_tables=getattr(config, "num_hash_tables", 10),
                num_hash_functions=getattr(config, "num_hash_functions", 8),
                hash_width=getattr(config, "hash_width", 1.0),
                random_seed=getattr(config, "random_seed", 42),
            )


class LSHIndex(BaseVectorIndex):
    """
    LSH (Locality Sensitive Hashing) based vector index for approximate nearest neighbor search

    Time Complexity:
    - Build: O(n * L * k) where n = vectors, L = tables, k = hash functions per table
    - Search: O(L * k + c) where c = number of candidates returned
    - Insert: O(L * k)
    - Delete: O(L * k)

    Space Complexity: O(n * L) where n = vectors, L = hash tables

    LSH provides sub-linear search time with probabilistic guarantees for similarity search.
    Different hash families are used for different similarity metrics:
    - Cosine: Random hyperplane hashing
    - Euclidean: p-stable distribution hashing (Gaussian)
    - Dot Product: Asymmetric LSH transformation + cosine hashing
    """

    def __init__(
        self,
        similarity_metric: SimilarityMetric,
        lsh_params: Optional[LSHParams] = None,
    ):
        super().__init__(similarity_metric)
        self.lsh_params = lsh_params or LSHParams()
        self.dimension: Optional[int] = None

        # Storage
        self._vectors: Dict[str, np.ndarray] = {}
        self._metadata: Dict[str, Dict[str, Any]] = {}

        # LSH hash tables: List[Dict[hash_value, Set[vector_id]]]
        self._hash_tables: List[Dict[str, Set[str]]] = []
        self._hash_functions: List[List[np.ndarray]] = []  # Random vectors for hashing

        # For p-stable hashing (Euclidean)
        self._hash_offsets: List[List[float]] = []  # Random offsets

        # Concurrency
        self._index_id = f"lsh_index_{id(self)}"

        # Initialize with seed for reproducibility
        np.random.seed(self.lsh_params.random_seed)
        self._initialize_hash_tables()

    def _get_resource_id(self, operation: str, vector_id: Optional[str] = None) -> str:
        """Generate resource ID for locking"""
        if vector_id:
            return f"{self._index_id}:{vector_id}:{operation}"
        return f"{self._index_id}:{operation}"

    def _initialize_hash_tables(self):
        """Initialize hash tables and hash functions"""
        for _ in range(self.lsh_params.num_hash_tables):
            self._hash_tables.append({})

    def _generate_hash_functions(self, dimension: int):
        """Generate hash functions for the given dimension"""
        # Return if already generated
        if self._hash_functions and len(self._hash_functions[0][0]) == dimension:
            return

        self._hash_functions.clear()
        self._hash_offsets.clear()

        for _ in range(self.lsh_params.num_hash_tables):
            table_hash_functions = []
            table_hash_offsets = []

            for _ in range(self.lsh_params.num_hash_functions):
                if self.similarity_metric == SimilarityMetric.COSINE:
                    # Random hyperplane hashing for cosine similarity
                    random_vector = np.random.normal(0, 1, dimension)
                    random_vector = random_vector / np.linalg.norm(random_vector)
                    table_hash_functions.append(random_vector)

                elif self.similarity_metric == SimilarityMetric.EUCLIDEAN:
                    # p-stable distribution hashing (Gaussian for L2)
                    random_vector = np.random.normal(0, 1, dimension)
                    random_offset = np.random.uniform(0, self.lsh_params.hash_width)
                    table_hash_functions.append(random_vector)
                    table_hash_offsets.append(random_offset)

                elif self.similarity_metric == SimilarityMetric.DOT_PRODUCT:
                    # Transform to cosine similarity space and use hyperplane hashing
                    random_vector = np.random.normal(
                        0, 1, dimension + 1
                    )  # +1 for normalization
                    random_vector = random_vector / np.linalg.norm(random_vector)
                    table_hash_functions.append(random_vector)

            self._hash_functions.append(table_hash_functions)
            if table_hash_offsets:
                self._hash_offsets.append(table_hash_offsets)

    def _transform_vector_for_dot_product(self, vector: np.ndarray) -> np.ndarray:
        """Transform vector for dot product LSH (asymmetric transformation)"""
        # Add dimension for dot product -> cosine transformation
        norm = np.linalg.norm(vector)
        if norm == 0:
            return np.append(vector, 0)

        # Normalize and add extra dimension
        normalized = vector / norm
        extra_dim = math.sqrt(1 - min(1.0, norm**2))
        return np.append(normalized, extra_dim)

    def _compute_hash_signature(self, vector: np.ndarray, table_idx: int) -> str:
        """Compute hash signature for a vector in a specific table"""
        hash_values = []

        for func_idx in range(self.lsh_params.num_hash_functions):
            hash_func = self._hash_functions[table_idx][func_idx]

            if self.similarity_metric == SimilarityMetric.COSINE:
                # Sign of dot product with random hyperplane
                hash_val = 1 if np.dot(vector, hash_func) >= 0 else 0
                hash_values.append(str(hash_val))

            elif self.similarity_metric == SimilarityMetric.EUCLIDEAN:
                # p-stable hashing
                offset = self._hash_offsets[table_idx][func_idx]
                hash_val = int(
                    (np.dot(vector, hash_func) + offset) / self.lsh_params.hash_width
                )
                hash_values.append(str(hash_val))

            elif self.similarity_metric == SimilarityMetric.DOT_PRODUCT:
                # Use transformed vector with cosine hashing
                transformed_vector = self._transform_vector_for_dot_product(vector)
                hash_val = 1 if np.dot(transformed_vector, hash_func) >= 0 else 0
                hash_values.append(str(hash_val))

        return "|".join(hash_values)

    async def add_vector(
        self,
        vector_id: str,
        vector: List[float],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Add a vector to the LSH index with write lock"""
        resource_id = self._get_resource_id("add", vector_id)

        async with await lock_manager.write_lock(resource_id):
            vector_array = np.array(vector, dtype=np.float32)

            # Initialize dimension and hash functions on first vector
            if self.dimension is None:
                self.dimension = len(vector_array)
                self._generate_hash_functions(self.dimension)
            elif len(vector_array) != self.dimension:
                raise ValueError(
                    f"Vector dimension {len(vector_array)} doesn't match index dimension {self.dimension}"
                )

            # Normalize vector if using cosine similarity
            if self.similarity_metric == SimilarityMetric.COSINE:
                norm = np.linalg.norm(vector_array)
                if norm > 0:
                    vector_array = vector_array / norm

            # Store vector
            self._vectors[vector_id] = vector_array
            self._metadata[vector_id] = metadata or {}

            # Add to all hash tables
            for table_idx in range(self.lsh_params.num_hash_tables):
                hash_signature = self._compute_hash_signature(vector_array, table_idx)

                if hash_signature not in self._hash_tables[table_idx]:
                    self._hash_tables[table_idx][hash_signature] = set()

                self._hash_tables[table_idx][hash_signature].add(vector_id)

    async def remove_vector(self, vector_id: str) -> bool:
        """Remove a vector from the LSH index with write lock"""
        resource_id = self._get_resource_id("remove", vector_id)

        async with await lock_manager.write_lock(resource_id):
            # Early return if vector doesn't exist
            if vector_id not in self._vectors:
                return False

            vector = self._vectors[vector_id]

            # Remove from all hash tables
            for table_idx in range(self.lsh_params.num_hash_tables):
                hash_signature = self._compute_hash_signature(vector, table_idx)

                if hash_signature in self._hash_tables[table_idx]:
                    self._hash_tables[table_idx][hash_signature].discard(vector_id)

                    # Remove empty buckets
                    if not self._hash_tables[table_idx][hash_signature]:
                        del self._hash_tables[table_idx][hash_signature]

            # Remove from storage
            del self._vectors[vector_id]
            del self._metadata[vector_id]

            return True

    async def update_vector(
        self,
        vector_id: str,
        vector: List[float],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Update a vector in the LSH index with write lock"""
        resource_id = self._get_resource_id("update", vector_id)

        async with await lock_manager.write_lock(resource_id):
            # Early return if vector doesn't exist
            if vector_id not in self._vectors:
                return False

            # Remove old vector
            await self.remove_vector(vector_id)

            # Add updated vector
            await self.add_vector(vector_id, vector, metadata)

            return True

    async def search(
        self, query_vector: List[float], k: int = 10, **kwargs
    ) -> List[Tuple[str, float]]:
        """Search for k nearest neighbors using LSH with read lock"""
        resource_id = self._get_resource_id("search")

        async with await lock_manager.read_lock(resource_id):
            # Return if no vectors or uninitialized
            if not self._vectors or self.dimension is None:
                return []

            query_array = np.array(query_vector, dtype=np.float32)

            if len(query_array) != self.dimension:
                raise ValueError(
                    f"Query vector dimension {len(query_array)} doesn't match index dimension {self.dimension}"
                )

            # Normalize query vector if using cosine similarity
            if self.similarity_metric == SimilarityMetric.COSINE:
                norm = np.linalg.norm(query_array)
                if norm > 0:
                    query_array = query_array / norm

            # Collect candidate vectors from all hash tables
            candidates = set()

            for table_idx in range(self.lsh_params.num_hash_tables):
                hash_signature = self._compute_hash_signature(query_array, table_idx)

                if hash_signature in self._hash_tables[table_idx]:
                    candidates.update(self._hash_tables[table_idx][hash_signature])

            # Return if no candidates
            if not candidates:
                return []

            # Calculate exact similarities for candidates
            similarities = []
            for vector_id in candidates:
                if vector_id in self._vectors:
                    vector = self._vectors[vector_id]
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

            # Clear all hash tables
            for table in self._hash_tables:
                table.clear()

    def get_vector_ids(self) -> List[str]:
        """Get all vector IDs in the index"""
        return list(self._vectors.keys())

    def get_lsh_stats(self) -> Dict[str, Any]:
        """Get LSH-specific statistics"""
        total_buckets = sum(len(table) for table in self._hash_tables)
        non_empty_buckets = sum(
            1 for table in self._hash_tables for bucket in table.values() if bucket
        )

        bucket_sizes = [
            len(bucket) for table in self._hash_tables for bucket in table.values()
        ]
        avg_bucket_size = sum(bucket_sizes) / len(bucket_sizes) if bucket_sizes else 0
        max_bucket_size = max(bucket_sizes) if bucket_sizes else 0

        return {
            "num_hash_tables": self.lsh_params.num_hash_tables,
            "num_hash_functions": self.lsh_params.num_hash_functions,
            "hash_width": self.lsh_params.hash_width,
            "total_buckets": total_buckets,
            "non_empty_buckets": non_empty_buckets,
            "average_bucket_size": avg_bucket_size,
            "max_bucket_size": max_bucket_size,
            "total_vectors": len(self._vectors),
            "dimension": self.dimension,
        }

    # Legacy compatibility methods
    def search_knn(
        self, query_vector: np.ndarray, k: int
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        """Legacy sync search method for backward compatibility"""
        # Return if no vectors
        if not self._vectors or self.dimension is None:
            return []

        if len(query_vector) != self.dimension:
            raise ValueError(
                f"Query vector dimension {len(query_vector)} doesn't match index dimension {self.dimension}"
            )

        # Normalize query vector if using cosine similarity
        if self.similarity_metric == SimilarityMetric.COSINE:
            norm = np.linalg.norm(query_vector)
            if norm > 0:
                query_vector = query_vector / norm

        # Collect candidates from all hash tables
        candidates = set()

        for table_idx in range(self.lsh_params.num_hash_tables):
            hash_signature = self._compute_hash_signature(query_vector, table_idx)

            if hash_signature in self._hash_tables[table_idx]:
                candidates.update(self._hash_tables[table_idx][hash_signature])

        # Calculate exact similarities for candidates
        similarities = []
        for vector_id in candidates:
            if vector_id in self._vectors:
                vector = self._vectors[vector_id]
                similarity = self._calculate_similarity(query_vector, vector)
                metadata = self._metadata.get(vector_id, {})
                similarities.append((vector_id, similarity, metadata))

        # Sort by similarity and return top k
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:k]

    def add_vector_sync(
        self, vector_id: str, vector: np.ndarray, metadata: Dict[str, Any]
    ):
        """Add a vector to the index (synchronous version for compatibility)"""
        # Initialize dimension and hash functions on first vector
        if self.dimension is None:
            self.dimension = len(vector)
            self._generate_hash_functions(self.dimension)

        # Normalize vector if using cosine similarity
        if self.similarity_metric == SimilarityMetric.COSINE:
            norm = np.linalg.norm(vector)
            if norm > 0:
                vector = vector / norm

        # Store vector and metadata
        self._vectors[vector_id] = vector.copy()
        self._metadata[vector_id] = metadata.copy()

        # Add to all hash tables
        for table_idx in range(self.lsh_params.num_hash_tables):
            hash_signature = self._compute_hash_signature(vector, table_idx)

            if hash_signature not in self._hash_tables[table_idx]:
                self._hash_tables[table_idx][hash_signature] = set()

            self._hash_tables[table_idx][hash_signature].add(vector_id)
