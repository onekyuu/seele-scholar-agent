from collections.abc import Awaitable, Callable

from pydantic import BaseModel

from .state import DocumentChunk


class PromptsConfig(BaseModel):
    language_names: dict[str, str]
    language_titles: dict[str, str]
    language_abstract: dict[str, str]
    language_keywords: dict[str, str]

    planner_system_prompt: str
    planner_user_prompt: str
    writer_system_prompt: str
    writer_user_prompt: str
    reviewer_system_prompt: str
    reviewer_user_prompt: str
    topic_proposer_system_prompt: str
    topic_proposer_user_prompt: str


RAGRetrieverFunc = Callable[[str], Awaitable[list[DocumentChunk]]]
