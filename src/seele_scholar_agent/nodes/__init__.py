import asyncio
import re
from collections.abc import AsyncIterator
from typing import Any, Literal, TypedDict

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
# Writer 节点为每章生成摘要的最大字符数（约 150 tokens，用于后续章节的上下文）
SECTION_SUMMARY_MAX_CHARS = 600
# Researcher 在 paper_summaries 中每篇文献 abstract 部分的最大字符数
PAPER_SUMMARY_ABSTRACT_CHARS = 300

CITATION_PATTERN = re.compile(r"\[(\d+)\]")


class NodeStreamEvent(TypedDict, total=False):
    type: Literal["token", "progress", "result"]
    token: str
    progress: str
    result: dict[str, Any]


async def _stream_llm_text(
    chain: Runnable[dict[str, Any], Any],
    input_data: dict[str, Any],
    *,
    max_retries: int = LLM_MAX_RETRIES,
    base_delay: float = LLM_RETRY_BASE_DELAY,
) -> AsyncIterator[NodeStreamEvent]:
    last_error: Exception | None = None
    for attempt in range(1 + max_retries):
        try:
            async for chunk in chain.astream(input_data):
                text: str = chunk.content if hasattr(chunk, "content") else str(chunk)
                if isinstance(text, list):
                    text = "".join(str(c) for c in text)
                if text:
                    yield NodeStreamEvent(type="token", token=text)
            return
        except Exception as exc:
            last_error = exc
            if attempt < max_retries:
                delay = base_delay * (2**attempt)
                logger.warning(
                    "stream chain invocation failed, retrying",
                    attempt=attempt + 1,
                    max_attempts=1 + max_retries,
                    delay=delay,
                    error=str(exc),
                )
                await asyncio.sleep(delay)
    raise last_error  # type: ignore[misc]


async def invoke_with_retry(
    chain: Runnable[dict[str, Any], Any],
    input_data: dict[str, Any],
    *,
    max_retries: int = LLM_MAX_RETRIES,
    base_delay: float = LLM_RETRY_BASE_DELAY,
) -> Any:
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
