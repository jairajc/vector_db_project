"""KD-Tree based vector index implementation with async concurrency control"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from app.indexing.base_index import BaseVectorIndex
from app.core.constants import SimilarityMetric
from app.utils.concurrency import lock_manager


@dataclass
class KDNode:

    vector_id: str
    vector: np.ndarray
    metadata: Dict[str, Any]
    left: Optional["KDNode"] = None
    right: Optional["KDNode"] = None
    axis: int = 0


class KDTreeIndex(BaseVectorIndex):

    def __init__(self, similarity_metric: SimilarityMetric = SimilarityMetric.COSINE):
        super().__init__(similarity_metric)
        self.root: Optional[KDNode] = None
        self.dimension: Optional[int] = None
        self._vectors: Dict[str, np.ndarray] = {}
        self._metadata: Dict[str, Dict[str, Any]] = {}
        self._index_id = f"kdtree_index_{id(self)}"

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
        """Add a vector to the KD-Tree index with write lock"""
        resource_id = self._get_resource_id("add", vector_id)

        async with await lock_manager.write_lock(resource_id):
            vector_array = np.array(vector, dtype=np.float32)

        # Initialize dimension on first vector
            if self.dimension is None:
                self.dimension = len(vector_array)
            elif len(vector_array) != self.dimension:
                raise ValueError(
                    f"Vector dimension {len(vector_array)} doesn't match index dimension {self.dimension}"
                )

        # Normalize vector if using cosine similarity
            if self.similarity_metric == SimilarityMetric.COSINE:
                vector_array = self._normalize_vector(vector_array)

        # Store vector
            self._vectors[vector_id] = vector_array
            self._metadata[vector_id] = metadata or {}

        # Create new node
            new_node = KDNode(
                vector_id=vector_id, vector=vector_array, metadata=metadata or {}
            )

        # Insert into tree
            self.root = self._insert_node(self.root, new_node, 0)

    async def remove_vector(self, vector_id: str) -> bool:
        """Remove a vector from the KD-Tree index with write lock."""
        resource_id = self._get_resource_id("remove", vector_id)

        async with await lock_manager.write_lock(resource_id):
            if vector_id not in self._vectors:
                return False

        # Remove from storage
            del self._vectors[vector_id]
            del self._metadata[vector_id]

        # Rebuild tree (simple approach for now- could be optimized)
            await self._rebuild_tree_internal()

            return True

    async def update_vector(
        self,
        vector_id: str,
        vector: List[float],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Update a vector in the KD-Tree index with write lock"""
        resource_id = self._get_resource_id("update", vector_id)

        async with await lock_manager.write_lock(resource_id):
            if vector_id not in self._vectors:
                return False

            vector_array = np.array(vector, dtype=np.float32)

            if len(vector_array) != self.dimension:
                raise ValueError(
                    f"Vector dimension {len(vector_array)} doesn't match index dimension {self.dimension}"
                )

            if self.similarity_metric == SimilarityMetric.COSINE:
                vector_array = self._normalize_vector(vector_array)

            self._vectors[vector_id] = vector_array
            self._metadata[vector_id] = metadata or {}

            await self._rebuild_tree_internal()

            return True

    async def search(
        self, query_vector: List[float], k: int = 10, **kwargs
    ) -> List[Tuple[str, float]]:
        """Search for k nearest neighbors using KD-Tree with read lock"""
        resource_id = self._get_resource_id("search")

        async with await lock_manager.read_lock(resource_id):
            if self.root is None or not self._vectors:
                return []

            query_array = np.array(query_vector, dtype=np.float32)

            if len(query_array) != self.dimension:
                raise ValueError(
                    f"Query vector dimension {len(query_array)} doesn't match index dimension {self.dimension}"
                )

        # Normalize query vector if using cosine similarity
            if self.similarity_metric == SimilarityMetric.COSINE:
                query_array = self._normalize_vector(query_array)

        # Perform KD-Tree search
            best_matches = []
            self._search_kdtree(self.root, query_array, k, best_matches, 0)

        # Sort by similarity score
            best_matches.sort(key=lambda x: x[1], reverse=True)

            return best_matches[:k]

    def search_knn(
        self, query_vector: np.ndarray, k: int
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        """Legacy sync search method for backward compatibility"""
        if self.root is None or not self._vectors:
            return []

        if len(query_vector) != self.dimension:
            raise ValueError(
                f"Query vector dimension {len(query_vector)} doesn't match index dimension {self.dimension}"
            )

    # Same normalizing again
        if self.similarity_metric == SimilarityMetric.COSINE:
            query_vector = self._normalize_vector(query_vector)

    # Perform KD-Tree search
        best_matches = []
        self._search_kdtree(self.root, query_vector, k, best_matches, 0)

    # Sort by similarity score and add metadata
        best_matches.sort(key=lambda x: x[1], reverse=True)

        results = []
        for vector_id, similarity in best_matches[:k]:
            metadata = self._metadata.get(vector_id, {})
            results.append((vector_id, similarity, metadata))

        return results

    async def get_vector_count(self) -> int:
        """Get the number of vectors in the index with read lock"""
        resource_id = self._get_resource_id("count")

        async with await lock_manager.read_lock(resource_id):
            return len(self._vectors)

    async def clear(self) -> None:
        """Clear all vectors from the index with write lock"""
        resource_id = self._get_resource_id("clear")

        async with await lock_manager.write_lock(resource_id):
            self.root = None
            self.dimension = None
            self._vectors.clear()
            self._metadata.clear()

    def _insert_node(
        self, node: Optional[KDNode], new_node: KDNode, depth: int
    ) -> KDNode:
        """Insert a node into the KD-Tree (internal method, assumes lock is held)"""
        if node is None:
            new_node.axis = depth % (self.dimension or 1)
            return new_node

        axis = depth % (self.dimension or 1)

        if new_node.vector[axis] < node.vector[axis]:
            node.left = self._insert_node(node.left, new_node, depth + 1)
        else:
            node.right = self._insert_node(node.right, new_node, depth + 1)

        return node

    def _search_kdtree(
        self,
        node: Optional[KDNode],
        query: np.ndarray,
        k: int,
        best_matches: List[Tuple[str, float]],
        depth: int,
    ) -> None:
        """Recursively search the KD-Tree for nearest neighbors"""
        if node is None:
            return

    # Calculate similarity to current node
        similarity = self._calculate_similarity(query, node.vector)

    # Update best matches
        if len(best_matches) < k:
            best_matches.append((node.vector_id, similarity))
        else:
        # Find the worst match
            worst_idx = min(range(len(best_matches)), key=lambda i: best_matches[i][1])
            if similarity > best_matches[worst_idx][1]:
                best_matches[worst_idx] = (node.vector_id, similarity)

    # Determine which subtree to search first
        axis = depth % (self.dimension or 1)

        if query[axis] < node.vector[axis]:
        # Search left subtree first
            self._search_kdtree(node.left, query, k, best_matches, depth + 1)

        # Check if we need to search right subtree
            if len(best_matches) < k or self._should_search_other_side(
                query, node, best_matches, axis
            ):
                self._search_kdtree(node.right, query, k, best_matches, depth + 1)
        else:
        # Search right subtree first
            self._search_kdtree(node.right, query, k, best_matches, depth + 1)

        # Check if we need to search left subtree
            if len(best_matches) < k or self._should_search_other_side(
                query, node, best_matches, axis
            ):
                self._search_kdtree(node.left, query, k, best_matches, depth + 1)

    def _should_search_other_side(
        self,
        query: np.ndarray,
        node: KDNode,
        best_matches: List[Tuple[str, float]],
        axis: int,
    ) -> bool:
        """Determine if we should search the other side of the split (internal method)"""
        if len(best_matches) < 1:
            return True

    # Find current worst match
        worst_similarity = min(match[1] for match in best_matches)

    # Calculate distance to the splitting plane
        plane_distance = abs(query[axis] - node.vector[axis])

    # For simplicity, convert similarity threshold to distance threshold
    # This is a heuristic and could be improved based on the similarity metric
        if self.similarity_metric == SimilarityMetric.COSINE:
        # For cosine similarity, use a conservative threshold
            distance_threshold = 2.0 * (1.0 - worst_similarity)
        else:
        # For other metrics, use plane distance directly
            distance_threshold = plane_distance

        return plane_distance <= distance_threshold

    async def _rebuild_tree_internal(self) -> None:
        """Rebuild the tree from stored vectors."""
        if not self._vectors:
            self.root = None
            return

    # Create nodes for all vectors
        nodes = []
        for vector_id, vector in self._vectors.items():
            node = KDNode(
                vector_id=vector_id, vector=vector, metadata=self._metadata[vector_id]
            )
            nodes.append(node)

    # Build balanced tree
        self.root = self._build_balanced_tree(nodes, 0)

    async def _rebuild_tree(self) -> None:
        """Public method to rebuild tree"""
        await self._rebuild_tree_internal()

    def _build_balanced_tree(self, nodes: List[KDNode], depth: int) -> Optional[KDNode]:
        """Build a balanced KD-Tree from a list of nodes"""
        if not nodes:
            return None

    # Select axis based on depth
        axis = depth % (self.dimension or 1)

    # Sort nodes by the current axis
        nodes.sort(key=lambda x: x.vector[axis])

    # Select median as root
        median_idx = len(nodes) // 2
        median_node = nodes[median_idx]
        median_node.axis = axis

    # Recursively build subtrees
        median_node.left = self._build_balanced_tree(nodes[:median_idx], depth + 1)
        median_node.right = self._build_balanced_tree(
            nodes[median_idx + 1 :], depth + 1
        )

        return median_node

    def _normalize_vector(self, vector: np.ndarray) -> np.ndarray:
        """Normalize a vector to unit length"""
        norm = np.linalg.norm(vector)
        if norm > 0:
            return vector / norm
        return vector
