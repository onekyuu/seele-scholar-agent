from collections.abc import Awaitable, Callable
from typing import Any

from pydantic import BaseModel

from .policy.execution_strategy import GenerationMode, SectionExecutionPolicy
from .state import DocumentChunk, PaperMetadata, SectionDraft


class PromptsConfig(BaseModel):
    planner_system_prompt: str
    planner_user_prompt: str
    writer_system_prompt: str
    writer_user_prompt: str
    proposal_writer_user_prompt: str | None = None
    proposal_revision_user_prompt: str | None = None
    academic_revision_user_prompt: str | None = None
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

    terminology_check_system_prompt: str
    terminology_check_user_prompt: str
    logic_check_system_prompt: str
    logic_check_user_prompt: str
    reference_consistency_system_prompt: str
    reference_consistency_user_prompt: str


RAGRetrieverFunc = Callable[[str], Awaitable[list[DocumentChunk]]]

PaperSearchFunc = Callable[[str], Awaitable[list[PaperMetadata]]]
BudgetAllocatorFunc = Callable[
    [Any, list[SectionDraft], int],
    Awaitable[Any],
]


class GraphConfig(BaseModel):
    generation_mode: GenerationMode = GenerationMode.FULL_DOCUMENT
    auto_advance_sections: bool = True
    require_section_approval: bool = False
    enable_topic_proposer: bool = True
    require_topic_approval: bool = True
    require_outline_approval: bool = True
    enable_outline_quality_gate: bool = True
    enable_budget_gate: bool = True
    enable_draft_integration: bool = True
    enable_exemplar_context: bool = False
    enable_similarity_gate: bool = False
    enable_reference_generator: bool = True
    enable_consistency_checker: bool = True
    enable_finalizer: bool = True
    enable_integrity_gate: bool = True

    def section_execution_policy(self) -> SectionExecutionPolicy:
        return SectionExecutionPolicy(
            generation_mode=self.generation_mode,
            auto_advance_sections=self.auto_advance_sections,
            require_section_approval=self.require_section_approval,
            stop_after_section_review=self.generation_mode == GenerationMode.SINGLE_SECTION,
        )
