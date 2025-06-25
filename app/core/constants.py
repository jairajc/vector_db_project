"""Constants used throughout the Vector Database API : This defines limits, defaults, and enums (IndexType, SimilarityMetric)
to ensure consistency"""

from enum import Enum


class IndexType(str, Enum):
    """Available vector index types"""

    LINEAR = "linear"
    KD_TREE = "kd_tree"
    LSH = "lsh"


class SimilarityMetric(str, Enum):
    """Available similarity metrics for vector search"""

    COSINE = "cosine"
    EUCLIDEAN = "euclidean"
    DOT_PRODUCT = "dot_product"


# API Configuration
API_V1_PREFIX = "/api/v1"
DOCS_URL = "/docs"
REDOC_URL = "/redoc"

# Pagination
DEFAULT_PAGE_SIZE = 20
MAX_PAGE_SIZE = 100

# Vector Operations
MIN_EMBEDDING_DIMENSION = 1
MAX_EMBEDDING_DIMENSION = 4096
DEFAULT_SEARCH_K = 10
MAX_SEARCH_K = 100

# Concurrency
DEFAULT_LOCK_TIMEOUT = 30
MAX_LOCK_TIMEOUT = 300

# Text Processing
MAX_TEXT_LENGTH = 10000
MIN_TEXT_LENGTH = 1
