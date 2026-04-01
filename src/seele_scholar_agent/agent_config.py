from collections.abc import Awaitable, Callable

from pydantic import BaseModel

from .state import DocumentChunk


class PromptsConfig(BaseModel):
    planner_system_prompt: str
    planner_user_prompt: str
    writer_system_prompt: str
    writer_user_prompt: str
    reviewer_system_prompt: str
    reviewer_user_prompt: str
    topic_proposer_system_prompt: str
    topic_proposer_user_prompt: str
    finalizer_system_prompt: str
    finalizer_user_prompt: str
    consistency_check_system_prompt: str
    consistency_check_user_prompt: str
    citation_alignment_system_prompt: str
    citation_alignment_user_prompt: str


RAGRetrieverFunc = Callable[[str], Awaitable[list[DocumentChunk]]]
