"""Shared utilities for LangGraph nodes."""

import asyncio
import re
from typing import Any

from langchain_core.runnables import Runnable

from ..logging import get_logger

logger = get_logger(__name__)

HTTP_TIMEOUT = 30.0
ARXIV_RATE_LIMIT_DELAY = 3.0
LLM_MAX_RETRIES = 2
LLM_RETRY_BASE_DELAY = 1.0
API_MAX_RETRIES = 3
API_RETRY_BASE_DELAY = 3.0

PREVIOUS_SECTION_MAX_CHARS = 500

CITATION_PATTERN = re.compile(r"\[(\d+)\]")


async def invoke_with_retry(
    chain: Runnable,
    input_data: dict[str, Any],
    *,
    max_retries: int = LLM_MAX_RETRIES,
    base_delay: float = LLM_RETRY_BASE_DELAY,
) -> Any:
    """Invoke a LangChain Runnable with exponential-backoff retry.

    Retries on any exception (JSON parse failures, transient API errors, etc.).
    Raises the last exception if all attempts (1 + max_retries) fail.
    """
    last_error: Exception | None = None
    for attempt in range(1 + max_retries):
        try:
            return await chain.ainvoke(input_data)
        except Exception as exc:
            last_error = exc
            if attempt < max_retries:
                delay = base_delay * (2**attempt)
                logger.warning(
                    "chain invocation failed, retrying",
                    attempt=attempt + 1,
                    max_attempts=1 + max_retries,
                    delay=delay,
                    error=str(exc),
                )
                await asyncio.sleep(delay)
    raise last_error  # type: ignore[misc]
