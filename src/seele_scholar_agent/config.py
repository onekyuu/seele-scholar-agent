from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    model_config = {"env_file": Path(__file__).parent / ".env", "extra": "ignore"}

    SEMANTIC_SCHOLAR_API_KEY: str = ""

    MAX_REVISIONS: int = 3


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
