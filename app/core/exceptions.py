"""Custom exception classes for Vector Database API: Providing specific error types
(LibraryNotFound, ChunkNotFound) for better error handling"""

from typing import Any, Dict, Optional


class VectorDBException(Exception):
    """Base exception for all vector database errors"""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        self.message = message
        self.details = details or {}
        super().__init__(self.message)


class LibraryNotFound(VectorDBException):
    """Raised when a library is not found"""

    def __init__(self, library_id: str):
        super().__init__(
            f"Library with ID '{library_id}' not found", {"library_id": library_id}
        )


class DocumentNotFound(VectorDBException):
    """Raised when a document is not found"""

    def __init__(self, document_id: str, library_id: Optional[str] = None):
        message = f"Document with ID '{document_id}' not found"
        if library_id:
            message += f" in library '{library_id}'"

        details = {"document_id": document_id}
        if library_id:
            details["library_id"] = library_id

        super().__init__(message, details)


class ChunkNotFound(VectorDBException):
    """Raised when a chunk is not found"""

    def __init__(self, chunk_id: str, library_id: Optional[str] = None):
        message = f"Chunk with ID '{chunk_id}' not found"
        if library_id:
            message += f" in library '{library_id}'"

        details = {"chunk_id": chunk_id}
        if library_id:
            details["library_id"] = library_id

        super().__init__(message, details)


class IndexingError(VectorDBException):
    """Raised when there's an error during indexing operations"""

    def __init__(
        self, message: str, operation: str, details: Optional[Dict[str, Any]] = None
    ):
        full_message = f"Indexing error during {operation}: {message}"
        error_details = {"operation": operation}
        if details:
            error_details.update(details)
        super().__init__(full_message, error_details)


class EmbeddingError(VectorDBException):
    """Raised when there's an error generating embeddings"""

    def __init__(self, message: str, text: Optional[str] = None):
        details = {}
        if text:
            details["text_length"] = len(text)
            details["text_preview"] = text[:100] + "..." if len(text) > 100 else text
        super().__init__(f"Embedding generation failed: {message}", details)


class ConcurrencyError(VectorDBException):
    """Raised when there's a concurrency-related error"""

    def __init__(self, message: str, operation: str, timeout: Optional[int] = None):
        details = {"operation": operation}
        if timeout:
            details["timeout_seconds"] = str(timeout)
        super().__init__(f"Concurrency error in {operation}: {message}", details)


class ValidationError(VectorDBException):
    """Raised when input validation fails"""

    def __init__(self, field: str, value: Any, reason: str):
        super().__init__(
            f"Validation error for field '{field}': {reason}",
            {"field": field, "value": str(value), "reason": reason},
        )


class PersistenceError(VectorDBException):
    """Raised when there's an error with disk persistence operations"""

    def __init__(self, message: str, operation: Optional[str] = None):
        details = {}
        if operation:
            details["operation"] = operation
        super().__init__(f"Persistence error: {message}", details)
