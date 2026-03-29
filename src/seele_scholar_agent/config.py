from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    model_config = {"env_file": Path(__file__).parent / ".env", "extra": "ignore"}

    OPENAI_API_KEY: str = ""
    OPENAI_MODEL: str = "gpt-4o"
    OPENAI_BASE_URL: str = "https://api.openai.com/v1"
    OPENAI_TEMPERATURE: float = 0.7
    OPENAI_MAX_TOKENS: int = 4096

    SEMANTIC_SCHOLAR_API_KEY: str = ""

    QDRANT_URL: str = "http://localhost:6333"
    QDRANT_API_KEY: str = ""
    QDRANT_COLLECTION: str = "user_documents"

    EMBEDDING_MODEL: str = "text-embedding-3-small"

    MAX_REVISIONS: int = 3
    DEFAULT_TOP_K: int = 10


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
