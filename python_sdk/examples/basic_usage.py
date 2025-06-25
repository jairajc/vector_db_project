"""
Basic usage examples for VectorDB Python SDK

This script demonstrates the core functionality of the VectorDB client including:
- Creating libraries with different index types
- Adding documents and chunks
- Performing vector similarity search
- Using metadata filtering
- Batch operations
"""

import asyncio
from datetime import datetime, timedelta
from vectordb_client import VectorDBClient, AsyncVectorDBClient
from vectordb_client.models import (
    LibraryCreate,
    DocumentCreate,
    ChunkCreate,
    IndexType,
    SimilarityMetric,
    LSHConfig,
    MetadataFilter,
)


def basic_sync_example():
    """Basic synchronous client usage"""
    print("=== Basic Sync Example ===")

    # Initialize client
    with VectorDBClient("http://localhost:8000") as client:

        # Check if API is available
        if not client.ping():
            print("VectorDB API is not available. Please start the server first.")
            return

        print("✓ Connected to VectorDB API")

        # Create a library with Linear index (good for small datasets)
        library = client.create_library_simple(
            name="Basic Example Library",
            description="Demonstrating basic VectorDB operations",
            index_type="linear",
            similarity_metric="cosine",
        )
        print(f"✓ Created library: {library.name} (ID: {library.id})")

        # Add some sample text chunks
        sample_texts = [
            {
                "text": "Python is a high-level programming language known for its simplicity and readability.",
                "metadata": {
                    "category": "programming",
                    "language": "python",
                    "difficulty": "beginner",
                },
            },
            {
                "text": "Machine learning algorithms can find patterns in large datasets automatically.",
                "metadata": {
                    "category": "ai",
                    "topic": "machine_learning",
                    "difficulty": "intermediate",
                },
            },
            {
                "text": "Vector databases are optimized for storing and searching high-dimensional vectors.",
                "metadata": {
                    "category": "database",
                    "topic": "vectors",
                    "difficulty": "advanced",
                },
            },
            {
                "text": "FastAPI is a modern web framework for building APIs with Python.",
                "metadata": {
                    "category": "programming",
                    "language": "python",
                    "difficulty": "intermediate",
                },
            },
            {
                "text": "Neural networks are computational models inspired by biological neural networks.",
                "metadata": {
                    "category": "ai",
                    "topic": "neural_networks",
                    "difficulty": "advanced",
                },
            },
        ]

        # Add chunks to the library
        chunk_ids = []
        for i, sample in enumerate(sample_texts):
            chunk = client.add_text_chunk(
                library_id=library.id, text=sample["text"], metadata=sample["metadata"]
            )
            chunk_ids.append(chunk.id)
            print(f"✓ Added chunk {i+1}: {sample['text'][:50]}...")

        # Perform basic search
        print("\n--- Basic Search ---")
        results = client.search_text(
            library_id=library.id, query="programming language", limit=3
        )

        for i, result in enumerate(results):
            print(f"{i+1}. Similarity: {result['similarity']:.3f}")
            print(f"   Text: {result['text'][:80]}...")
            print(f"   Category: {result['metadata'].get('category', 'N/A')}")
            print()

        # Search with metadata filtering
        print("--- Search with Metadata Filter ---")
        filtered_results = client.filter_by_metadata(
            library_id=library.id,
            query="algorithms and models",
            field="category",
            operator="eq",
            value="ai",
            k=5,
        )

        print(f"Found {len(filtered_results.results)} AI-related results:")
        for result in filtered_results.results:
            print(
                f"- {result.chunk.text[:60]}... (Score: {result.similarity_score:.3f})"
            )

        # Get library statistics
        stats = client.get_library_statistics(library.id)
        print(f"\n--- Library Statistics ---")
        print(f"Total chunks: {stats['chunk_count']}")
        print(f"Index type: {stats['index_type']}")
        print(f"Similarity metric: {stats['similarity_metric']}")

        # Cleanup
        client.delete_library(library.id)
        print(f"✓ Cleaned up library: {library.id}")


async def advanced_async_example():
    """Advanced asynchronous client usage with complex filtering"""
    print("\n=== Advanced Async Example ===")

    async with AsyncVectorDBClient("http://localhost:8000") as client:

        # Wait for API to be ready
        if not await client.wait_for_ready(timeout=10.0):
            print("VectorDB API is not ready. Please check the server.")
            return

        print("✓ Connected to VectorDB API (async)")

        # Create a library with LSH index
        library_data = LibraryCreate(
            name="Advanced Example Library",
            description="Demonstrating advanced VectorDB features",
            index_type=IndexType.LSH,
            similarity_metric=SimilarityMetric.COSINE,
            lsh_config=LSHConfig(
                num_hash_tables=12, num_hash_functions=8, hash_width=1.0, random_seed=42
            ),
        )

        library = await client.create_library(library_data)
        print(f"✓ Created LSH library: {library.name} (ID: {library.id})")

        # Create sample data with timestamps
        base_time = datetime.now()
        sample_data = []

        for i in range(20):
            created_at = base_time - timedelta(days=i)
            sample_data.append(
                ChunkCreate(
                    text=f"Research paper {i+1} about artificial intelligence and machine learning applications in modern technology.",
                    metadata={
                        "paper_id": f"paper_{i+1:03d}",
                        "category": "research" if i % 3 == 0 else "technology",
                        "author": f"Author {(i % 5) + 1}",
                        "created_at": created_at.isoformat(),
                        "keywords": (
                            ["ai", "ml", "tech"]
                            if i % 2 == 0
                            else ["research", "analysis"]
                        ),
                        "citations": (i + 1) * 10,
                        "difficulty_level": ["beginner", "intermediate", "advanced"][
                            i % 3
                        ],
                    },
                )
            )

        # Batch insert chunks
        print("Adding chunks in batches...")
        created_chunks = await client.create_chunks_batch(
            library_id=library.id, chunks=sample_data, batch_size=5
        )
        print(f"✓ Added {len(created_chunks)} chunks in batches")

        # Complex metadata filtering examples
        print("\n--- Complex Metadata Filtering ---")

        # 1. Date range filter - papers from last 10 days
        date_filter = MetadataFilter(
            field="created_at",
            operator="date_range",
            value={
                "start": (base_time - timedelta(days=10)).isoformat(),
                "end": base_time.isoformat(),
            },
        )

        # 2. Category filter - only research papers
        category_filter = MetadataFilter(
            field="category", operator="eq", value="research"
        )

        # 3. Citations filter - highly cited papers
        citations_filter = MetadataFilter(field="citations", operator="gte", value=100)

        # Search with combined filters
        response = await client.search(
            library_id=library.id,
            query="artificial intelligence research",
            k=10,
            metadata_filters=[date_filter, category_filter, citations_filter],
            filter_mode="and",
        )

        print(f"Found {len(response.results)} papers matching all criteria:")
        for result in response.results:
            metadata = result.chunk.metadata
            print(f"- Paper: {metadata.get('paper_id')}")
            print(f"  Author: {metadata.get('author')}")
            print(f"  Citations: {metadata.get('citations')}")
            print(f"  Similarity: {result.similarity_score:.3f}")
            print()

        # 4. Array contains filter - papers with specific keywords
        keyword_filter = MetadataFilter(
            field="keywords", operator="array_contains", value="ai"
        )

        response = await client.search(
            library_id=library.id,
            query="machine learning applications",
            k=5,
            metadata_filters=[keyword_filter],
            similarity_threshold=0.1,
        )

        print(f"Papers containing 'ai' keyword: {len(response.results)}")

        # 5. Text search in metadata
        author_filter = MetadataFilter(
            field="author", operator="contains", value="Author 1", case_sensitive=False
        )

        response = await client.search(
            library_id=library.id,
            query="technology applications",
            k=10,
            metadata_filters=[author_filter],
        )

        print(f"Papers by 'Author 1': {len(response.results)}")

        # Multi-library search
        print("\n--- Multi-Library Search ---")

        # Create another library for comparison
        library2_data = LibraryCreate(
            name="Comparison Library",
            description="Second library for multi-search demo",
            index_type=IndexType.LINEAR,
            similarity_metric=SimilarityMetric.COSINE,
        )

        library2 = await client.create_library(library2_data)

        # Add a few chunks to the second library
        await client.create_chunk(
            library2.id,
            ChunkCreate(
                text="Deep learning neural networks for computer vision applications",
                metadata={"source": "vision_research", "category": "computer_vision"},
            ),
        )

        await client.create_chunk(
            library2.id,
            ChunkCreate(
                text="Natural language processing with transformer architectures",
                metadata={"source": "nlp_research", "category": "nlp"},
            ),
        )

        # Search across multiple libraries
        multi_results = await client.search_multiple_libraries(
            library_ids=[library.id, library2.id],
            query="deep learning neural networks",
            k=5,
        )

        print("Multi-library search results:")
        for lib_id, search_response in multi_results.items():
            lib_name = library.name if lib_id == library.id else library2.name
            print(f"Library: {lib_name}")
            print(f"Results: {len(search_response.results)}")
            if search_response.results:
                best_result = search_response.results[0]
                print(
                    f"Best match: {best_result.chunk.text[:60]}... (Score: {best_result.similarity_score:.3f})"
                )
            print()

        # Stream search results
        print("--- Streaming Search Results ---")
        queries = [
            "artificial intelligence",
            "machine learning algorithms",
            "neural network architecture",
            "data science applications",
        ]

        async for query, result in client.stream_search_results(
            library.id, queries, k=2
        ):
            if isinstance(result, Exception):
                print(f"Query '{query}' failed: {result}")
            else:
                print(f"Query: '{query}' -> {len(result.results)} results")

        # Cleanup
        await client.delete_library(library.id)
        await client.delete_library(library2.id)
        print("✓ Cleaned up all libraries")


def lsh_performance_example():
    """Demonstrate LSH performance characteristics"""
    print("\n=== LSH Performance Example ===")

    with VectorDBClient("http://localhost:8000") as client:
        if not client.ping():
            print("VectorDB API is not available.")
            return

        # Create LSH library with custom parameters
        lsh_library = client.create_library(
            LibraryCreate(
                name="LSH Performance Test",
                description="Testing LSH with custom parameters",
                index_type=IndexType.LSH,
                similarity_metric=SimilarityMetric.COSINE,
                lsh_config=LSHConfig(
                    num_hash_tables=15,  # More tables = better recall
                    num_hash_functions=10,  # More functions = better precision
                    hash_width=0.8,  # Smaller width = higher precision
                    random_seed=123,
                ),
            )
        )

        print(f"✓ Created LSH library with custom parameters")
        print(f"  Hash tables: 15")
        print(f"  Hash functions: 10")
        print(f"  Hash width: 0.8")

        # Add more substantial content for LSH testing
        documents = [
            "Artificial intelligence and machine learning are revolutionizing technology",
            "Deep neural networks process complex patterns in data",
            "Natural language processing enables computers to understand human text",
            "Computer vision algorithms analyze and interpret visual information",
            "Reinforcement learning trains agents through reward-based feedback",
            "Supervised learning uses labeled data to train predictive models",
            "Unsupervised learning discovers hidden patterns in unlabeled data",
            "Feature engineering transforms raw data into meaningful representations",
            "Cross-validation ensures model performance generalizes to new data",
            "Hyperparameter tuning optimizes model configuration for best results",
        ]

        # Add documents
        for i, doc in enumerate(documents):
            client.add_text_chunk(
                library_id=lsh_library.id,
                text=doc,
                metadata={"doc_id": i, "length": len(doc), "topic": "machine_learning"},
            )

        print(f"✓ Added {len(documents)} chunks to LSH index")

        # Test search performance
        import time

        search_queries = [
            "machine learning algorithms",
            "neural network training",
            "data pattern recognition",
            "artificial intelligence applications",
        ]

        print("\n--- LSH Search Performance ---")
        for query in search_queries:
            start_time = time.time()
            results = client.search_text(
                library_id=lsh_library.id, query=query, limit=5
            )
            search_time = (time.time() - start_time) * 1000  # Convert to ms

            print(f"Query: '{query}'")
            print(f"  Search time: {search_time:.2f}ms")
            print(f"  Results: {len(results)}")
            if results:
                print(f"  Best score: {results[0]['similarity']:.3f}")
            print()

        # Cleanup
        client.delete_library(lsh_library.id)
        print("✓ Cleaned up LSH library")


def error_handling_example():
    """Demonstrate error handling and edge cases"""
    print("\n=== Error Handling Example ===")

    client = VectorDBClient("http://localhost:8000")

    try:
        # Try to get a non-existent library
        client.get_library("non-existent-id")
    except Exception as e:
        print(f"✓ Handled library not found: {type(e).__name__}")

    try:
        # Create library with invalid parameters
        invalid_library = LibraryCreate(
            name="",  # Empty name should fail validation
            index_type="invalid_type",  # Invalid index type
            similarity_metric="cosine",
        )
        client.create_library(invalid_library)
    except Exception as e:
        print(f"✓ Handled validation error: {type(e).__name__}")

    # Test connection timeout handling
    try:
        slow_client = VectorDBClient(
            "http://localhost:8000", timeout=0.001  # Very short timeout
        )
        slow_client.ping()
    except Exception as e:
        print(f"✓ Handled timeout: {type(e).__name__}")

    client.close()


def main():
    """Run all examples"""
    print("VectorDB Python SDK Examples")
    print("=" * 40)

    # Run examples
    basic_sync_example()
    asyncio.run(advanced_async_example())
    lsh_performance_example()
    error_handling_example()

    print("\n" + "=" * 40)
    print("All examples completed successfully!")


if __name__ == "__main__":
    main()
