from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    model_config = {"env_file": Path(__file__).parent / ".env", "extra": "ignore"}

    SEMANTIC_SCHOLAR_API_KEY: str = ""

    MAX_REVISIONS: int = 3

    HTTP_TIMEOUT: float = 30.0
    ARXIV_RATE_LIMIT_DELAY: float = 3.0
    LLM_MAX_RETRIES: int = 2
    LLM_RETRY_BASE_DELAY: float = 1.0
    API_MAX_RETRIES: int = 3
    API_RETRY_BASE_DELAY: float = 3.0
    RETRIEVER_MAX_RETRY_AFTER_SECONDS: float = 10.0

    PREVIOUS_SECTION_MAX_CHARS: int = 500
    SECTION_SUMMARY_MAX_CHARS: int = 600
    PAPER_SUMMARY_ABSTRACT_CHARS: int = 300
    PAPER_STATE_ABSTRACT_CHARS: int = 100

    RETRIEVER_TOP_K: int = 10

    STRICT_MIN_SUPPORT_SCORE: float = 0.35
    STRICT_MIN_EVIDENCE_RELEVANCE: float = 0.2
    CLAIM_TEXT_MATCH_THRESHOLD: float = 0.65
    CITATION_BINDER_SUPPORTED_THRESHOLD: float = 0.35
    CITATION_BINDER_WEAK_THRESHOLD: float = 0.15


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
