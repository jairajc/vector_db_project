"""Embedding utilities for generating vector embeddings"""

import asyncio
import numpy as np
from typing import List, Optional
import cohere
from app.core.config import settings
from app.core.exceptions import EmbeddingError


class CohereEmbeddingGenerator:
    """Cohere API embedding generator"""

    def __init__(self, api_key: str, model: str = "embed-english-v3.0"):
        # API key is required
        if not api_key:
            raise EmbeddingError("Cohere API key is required")

        self.api_key = api_key
        self.model = model
        self.client = cohere.Client(api_key)

        # Model dimensions mapping
        self.model_dimensions = {
            "embed-english-v3.0": 1024,
            "embed-english-light-v3.0": 384,
            "embed-multilingual-v3.0": 1024,
            "embed-multilingual-light-v3.0": 384,
            "embed-english-v2.0": 4096,
            "embed-english-light-v2.0": 1024,
            "embed-multilingual-v2.0": 768,
        }

        self.dimension = self.model_dimensions.get(model, 1024)

    def _validate_text_input(self, text: str, context: str = "text") -> None:
        """Validate text input"""
        if not text or not text.strip():
            raise EmbeddingError(f"Cannot generate embedding for empty {context}")

    def _normalize_embedding(self, embedding: List[float]) -> List[float]:
        """Normalize embedding vector"""
        embedding_array = np.array(embedding)
        norm = np.linalg.norm(embedding_array)

        if norm > 0:
            embedding_array = embedding_array / norm

        return embedding_array.tolist()

    async def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding using Cohere API"""
        # Validate input
        self._validate_text_input(text, "text")

        try:
            # Run the synchronous Cohere API call in a thread pool
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: self.client.embed(
                    texts=[text.strip()], model=self.model, input_type="search_document"
                ),
            )

            # Validate response
            if not response.embeddings or len(response.embeddings) == 0:
                raise EmbeddingError(
                    f"No embedding returned from Cohere API for text: {text[:50]}..."
                )

            embedding = response.embeddings[0]

            # Normalize and return
            return self._normalize_embedding(embedding)

        except cohere.CohereError as e:
            raise EmbeddingError(f"Cohere API error: {str(e)}", text)
        except Exception as e:
            raise EmbeddingError(f"Failed to generate embedding: {str(e)}", text)

    async def generate_batch_embeddings(
        self, texts: List[str], batch_size: int = 50
    ) -> List[List[float]]:
        """Generate embeddings for multiple texts in batches"""
        # Validate input
        if not texts:
            return []

        all_embeddings = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]

            # Filter out empty texts
            valid_texts = []
            valid_indices = []

            for idx, text in enumerate(batch):
                if text and text.strip():
                    valid_texts.append(text.strip())
                    valid_indices.append(i + idx)

            # Skip if no valid texts in batch
            if not valid_texts:
                all_embeddings.extend([[]] * len(batch))
                continue

            try:
                loop = asyncio.get_event_loop()
                response = await loop.run_in_executor(
                    None,
                    lambda: self.client.embed(
                        texts=valid_texts,
                        model=self.model,
                        input_type="search_document",
                    ),
                )

                # Validate batch response
                if not response.embeddings or len(response.embeddings) != len(
                    valid_texts
                ):
                    raise EmbeddingError(
                        f"Batch embedding failed: unexpected response length"
                    )

                # Normalize all embeddings in batch
                batch_embeddings = []
                valid_idx = 0

                for idx in range(len(batch)):
                    if idx in [vi - i for vi in valid_indices]:
                        embedding = response.embeddings[valid_idx]
                        batch_embeddings.append(self._normalize_embedding(embedding))
                        valid_idx += 1
                    else:
                        batch_embeddings.append([])  # Empty embedding for invalid text

                all_embeddings.extend(batch_embeddings)

            except cohere.CohereError as e:
                raise EmbeddingError(f"Cohere API batch error: {str(e)}")
            except Exception as e:
                raise EmbeddingError(f"Failed to generate batch embeddings: {str(e)}")

        return all_embeddings

    async def generate_query_embedding(self, query: str) -> List[float]:
        """Generate embedding for search queries"""
        # Validate input
        self._validate_text_input(query, "query")

        try:
            # Run the synchronous Cohere API call in a thread pool
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: self.client.embed(
                    texts=[query.strip()], model=self.model, input_type="search_query"
                ),
            )

            # Validate response
            if not response.embeddings or len(response.embeddings) == 0:
                raise EmbeddingError(
                    f"No embedding returned from Cohere API for query: {query[:50]}..."
                )

            embedding = response.embeddings[0]

            # Normalize and return
            return self._normalize_embedding(embedding)

        except cohere.CohereError as e:
            raise EmbeddingError(f"Cohere API error: {str(e)}", query)
        except Exception as e:
            raise EmbeddingError(f"Failed to generate query embedding: {str(e)}", query)


# Initialize the embedding service
embedding_service = CohereEmbeddingGenerator(settings.cohere_api_key)
