# VectorDB Python SDK

Official Python client library for VectorDB API.

## Installation

```bash
pip install vectordb-client
```

## Quick Start

```python
from vectordb_client import VectorDBClient

client = VectorDBClient("http://localhost:8000")
health = client.health_check()
print(f"API Status: {health.status}")
```

## Features

- Synchronous and asynchronous clients
- Full CRUD operations for libraries, documents, and chunks
- Advanced metadata filtering
- Batch operations
- Error handling and retry logic
- Type safety with Pydantic models

For detailed documentation, see the main project README. 