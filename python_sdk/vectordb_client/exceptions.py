"""Custom exceptions for VectorDB SDK"""

from typing import Dict, Any, Optional, List
import json
from http import HTTPStatus


class VectorDBError(Exception):
    """Base exception for all VectorDB SDK errors"""

    def __init__(
        self,
        message: str,
        details: Optional[Dict[str, Any]] = None,
        status_code: Optional[int] = None,
        response_text: Optional[str] = None,
    ):
        self.message = message
        self.details = details or {}
        self.status_code = status_code
        self.response_text = response_text
        super().__init__(self.message)

    def __str__(self) -> str:
        error_parts = [self.message]
        if self.status_code:
            error_parts.append(f"Status: {self.status_code}")
        if self.details:
            error_parts.append(f"Details: {json.dumps(self.details, indent=2)}")
        return " | ".join(error_parts)


class ConnectionError(VectorDBError):
    """Raised when there's a connection issue with the API"""

    def __init__(
        self,
        message: str = "Failed to connect to VectorDB API",
        base_url: Optional[str] = None,
        timeout: Optional[float] = None,
    ):
        details = {}
        if base_url:
            details["base_url"] = base_url
        if timeout:
            details["timeout"] = timeout
        super().__init__(message, details)


class AuthenticationError(VectorDBError):
    """Raised when authentication fails"""

    def __init__(self, message: str = "Authentication failed"):
        super().__init__(message, status_code=HTTPStatus.UNAUTHORIZED.value)


class AuthorizationError(VectorDBError):
    """Raised when authorization fails"""

    def __init__(self, message: str = "Access denied"):
        super().__init__(message, status_code=HTTPStatus.FORBIDDEN.value)


class LibraryNotFoundError(VectorDBError):
    """Raised when a library is not found"""

    def __init__(self, library_id: str):
        super().__init__(
            f"Library with ID '{library_id}' not found",
            details={"library_id": library_id},
            status_code=HTTPStatus.NOT_FOUND.value,
        )


class DocumentNotFoundError(VectorDBError):
    """Raised when a document is not found"""

    def __init__(self, document_id: str, library_id: Optional[str] = None):
        message = f"Document with ID '{document_id}' not found"
        details = {"document_id": document_id}

        if library_id:
            message += f" in library '{library_id}'"
            details["library_id"] = library_id

        super().__init__(message, details, status_code=HTTPStatus.NOT_FOUND.value)


class ChunkNotFoundError(VectorDBError):
    """Raised when a chunk is not found"""

    def __init__(self, chunk_id: str, library_id: Optional[str] = None):
        message = f"Chunk with ID '{chunk_id}' not found"
        details = {"chunk_id": chunk_id}

        if library_id:
            message += f" in library '{library_id}'"
            details["library_id"] = library_id

        super().__init__(message, details, status_code=HTTPStatus.NOT_FOUND.value)


class ValidationError(VectorDBError):
    """Raised when input validation fails"""

    def __init__(
        self,
        message: str,
        field: Optional[str] = None,
        value: Optional[Any] = None,
        errors: Optional[List[Dict]] = None,
    ):
        details = {}
        if field:
            details["field"] = field
        if value is not None:
            details["value"] = str(value)
        if errors:
            details["validation_errors"] = errors

        super().__init__(message, details, status_code=HTTPStatus.UNPROCESSABLE_ENTITY.value)


class RateLimitError(VectorDBError):
    """Raised when rate limit is exceeded"""

    def __init__(
        self, message: str = "Rate limit exceeded", retry_after: Optional[int] = None
    ):
        details = {}
        if retry_after:
            details["retry_after_seconds"] = retry_after

        super().__init__(message, details, status_code=HTTPStatus.TOO_MANY_REQUESTS.value)


class ServerError(VectorDBError):
    """Raised when there's a server-side error"""

    def __init__(
        self,
        message: str = "Internal server error",
        status_code: int = HTTPStatus.INTERNAL_SERVER_ERROR.value,
        response_text: Optional[str] = None,
    ):
        super().__init__(message, status_code=status_code, response_text=response_text)


class TimeoutError(VectorDBError):
    """Raised when a request times out"""

    def __init__(
        self, message: str = "Request timed out", timeout: Optional[float] = None
    ):
        details = {}
        if timeout:
            details["timeout_seconds"] = timeout
        super().__init__(message, details)


class IndexingError(VectorDBError):
    """Raised when there's an error with indexing operations"""

    def __init__(
        self,
        message: str,
        operation: Optional[str] = None,
        index_type: Optional[str] = None,
    ):
        details = {}
        if operation:
            details["operation"] = operation
        if index_type:
            details["index_type"] = index_type
        super().__init__(message, details)


class EmbeddingError(VectorDBError):
    """Raised when there's an error generating embeddings"""

    def __init__(self, message: str, text_length: Optional[int] = None):
        details = {}
        if text_length:
            details["text_length"] = text_length
        super().__init__(message, details)


# HTTP status code to exception mapping
STATUS_CODE_EXCEPTIONS = {
    HTTPStatus.BAD_REQUEST.value: ValidationError,
    HTTPStatus.UNAUTHORIZED.value: AuthenticationError,
    HTTPStatus.FORBIDDEN.value: AuthorizationError,
    HTTPStatus.NOT_FOUND.value: VectorDBError,  # Will be specialized based on context
    HTTPStatus.UNPROCESSABLE_ENTITY.value: ValidationError,
    HTTPStatus.TOO_MANY_REQUESTS.value: RateLimitError,
    HTTPStatus.INTERNAL_SERVER_ERROR.value: ServerError,
    HTTPStatus.BAD_GATEWAY.value: ServerError,
    HTTPStatus.SERVICE_UNAVAILABLE.value: ServerError,
    HTTPStatus.GATEWAY_TIMEOUT.value: TimeoutError,
}


def create_exception_from_response(
    status_code: int,
    response_text: str,
    request_context: Optional[Dict[str, Any]] = None,
) -> VectorDBError:
    """Create appropriate exception from HTTP response"""

    try:
        error_data = json.loads(response_text)
        message = error_data.get("detail", f"HTTP {status_code} error")

        # Extract additional error details
        details = {}
        if isinstance(error_data, dict):
            details.update(error_data)
            details.pop(
                "detail", None
            )  # Remove detail from details to avoid duplication

    except (json.JSONDecodeError, ValueError):
        message = f"HTTP {status_code} error"
        details = {"raw_response": response_text}

# Add request context
    if request_context:
        details.update(request_context)

# Handle specific NOT_FOUND cases
    if status_code == HTTPStatus.NOT_FOUND.value:
        if request_context:
            path = request_context.get("path", "")
            if "/libraries/" in path:
                if "/documents/" in path:
                    if "/chunks/" in path:
                    # Extract chunk and library IDs from path
                        parts = path.split("/")
                        try:
                            lib_idx = parts.index("libraries") + 1
                            chunk_idx = parts.index("chunks") + 1
                            library_id = (
                                parts[lib_idx] if lib_idx < len(parts) else None
                            )
                            chunk_id = (
                                parts[chunk_idx] if chunk_idx < len(parts) else None
                            )
                            return ChunkNotFoundError(chunk_id or "unknown", library_id)
                        except (ValueError, IndexError):
                            pass
                    else:
                    # Document not found
                        parts = path.split("/")
                        try:
                            lib_idx = parts.index("libraries") + 1
                            doc_idx = parts.index("documents") + 1
                            library_id = (
                                parts[lib_idx] if lib_idx < len(parts) else None
                            )
                            document_id = (
                                parts[doc_idx] if doc_idx < len(parts) else None
                            )
                            return DocumentNotFoundError(
                                document_id or "unknown", library_id
                            )
                        except (ValueError, IndexError):
                            pass
                else:
                # Library not found
                    parts = path.split("/")
                    try:
                        lib_idx = parts.index("libraries") + 1
                        library_id = parts[lib_idx] if lib_idx < len(parts) else None
                        return LibraryNotFoundError(library_id or "unknown")
                    except (ValueError, IndexError):
                        pass

# Use status code mapping
    exception_class = STATUS_CODE_EXCEPTIONS.get(status_code, VectorDBError)
    return exception_class(message, details, status_code, response_text)
