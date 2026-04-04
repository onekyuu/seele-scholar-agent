from collections.abc import Awaitable, Callable

from pydantic import BaseModel

from .state import DocumentChunk, PaperMetadata


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
    topic_translation_system_prompt: str
    topic_translation_user_prompt: str

    # ── ConsistencyCheckerNode Map-Reduce 子检查的可选 prompts ──────────────
    # 若留空字符串，节点将使用内置默认模板，无需主项目更新即可获得优化效果
    terminology_check_system_prompt: str = ""
    terminology_check_user_prompt: str = ""
    logic_check_system_prompt: str = ""
    logic_check_user_prompt: str = ""
    reference_consistency_system_prompt: str = ""
    reference_consistency_user_prompt: str = ""


# Inject document chunks into WriterNode for writing context (figures, data)
RAGRetrieverFunc = Callable[[str], Awaitable[list[DocumentChunk]]]

# Inject additional paper metadata sources into ResearcherNode (e.g. PubMed, IEEE, user library)
PaperSearchFunc = Callable[[str], Awaitable[list[PaperMetadata]]]
