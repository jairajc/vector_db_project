"""Configuration management for Vector Database API : Centralizing all
the environment variables and settings for different deployment environments"""

import os


class Settings:
    """Application settings loaded from environment variables"""

    def __init__(self):
        # Application Configuration
        self.app_name = os.getenv("APP_NAME", "Vector Database API")
        self.app_version = os.getenv("APP_VERSION", "1.0.0")
        self.debug = os.getenv("DEBUG", "false").lower() in ("true", "1", "yes")
        self.host = os.getenv("HOST", "0.0.0.0")
        self.port = int(os.getenv("PORT", "8000"))

        # Cohere API Configuration
        self.cohere_api_key = os.getenv("COHERE_API_KEY")

        # Index Configuration
        self.default_index_type = os.getenv("DEFAULT_INDEX_TYPE", "linear")
        self.max_chunks_per_library = int(os.getenv("MAX_CHUNKS_PER_LIBRARY", "10000"))
        self.embedding_dimension = int(os.getenv("EMBEDDING_DIMENSION", "1024"))

        # Concurrency Configuration
        self.max_concurrent_requests = int(os.getenv("MAX_CONCURRENT_REQUESTS", "100"))
        self.lock_timeout_seconds = int(os.getenv("LOCK_TIMEOUT_SECONDS", "30"))

        # Logging Configuration
        self.log_level = os.getenv("LOG_LEVEL", "INFO")

        # Persistence Configuration
        self.persistence_type = os.getenv(
            "PERSISTENCE_TYPE", "memory"
        )  # "memory" or "file"
        self.data_directory = os.getenv("DATA_DIRECTORY", "data")


# Global settings instance
settings = Settings()
