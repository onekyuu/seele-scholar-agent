from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal, Protocol

from ..document_profile import is_research_proposal
from ..state import OutlineStructure, QualityIssue, ReviewIssue, ReviewResult, SectionOutline

WriterMode = Literal["draft", "academic_revision", "profile_draft", "profile_revision"]
DEFAULT_PROFILE_NAME = "default"
PROFILE_DRAFT_MODE: Literal["profile_draft"] = "profile_draft"
PROFILE_REVISION_MODE: Literal["profile_revision"] = "profile_revision"


@dataclass(frozen=True)
class ProfileWriterPrompts:
    draft_user_prompt: str
    revision_user_prompt: str


class DocumentProfile(Protocol):
    name: str
    allow_empty_references: bool
    skip_auto_finalizer: bool
    uses_profile_outline_quality: bool
    uses_strict_academic_integrity: bool
    review_document_type: str

    def effective_paper_type(self, requested: str) -> str: ...

    def effective_structure_pattern(self, requested: str) -> str: ...

    def default_outline(self, topic: str, lang: str) -> dict[str, Any] | None: ...

    def normalize_outline(
        self, outline: OutlineStructure, topic: str
    ) -> OutlineStructure: ...

    def planner_context_suffix(self, target_word_count: str) -> str: ...

    def writer_mode(self, has_revision_context: bool) -> WriterMode: ...

    def writer_prompts(self, prompts: Any) -> ProfileWriterPrompts | None: ...

    def review_policy_text(self) -> str: ...

    def missing_core_tasks(self, section_title: str, content: str) -> list[str]: ...

    def empty_reference_issue(self) -> QualityIssue | None: ...

    def structural_review_issues(
        self, section_id: str, section_title: str, content: str
    ) -> tuple[list[ReviewIssue], list[QualityIssue]]: ...

    def apply_review_policy(
        self, review: ReviewResult, quality_issues: list[QualityIssue]
    ) -> tuple[ReviewResult, list[QualityIssue]]: ...

    def outline_section_issues(
        self, section: SectionOutline, *, is_last: bool
    ) -> list[QualityIssue]: ...

    def outline_structure_issues(self, outline: OutlineStructure) -> list[QualityIssue]: ...

    def skip_methodology_audit(self, content: str) -> bool: ...


class DefaultDocumentProfile:
    name = DEFAULT_PROFILE_NAME
    allow_empty_references = False
    skip_auto_finalizer = False
    uses_profile_outline_quality = False
    uses_strict_academic_integrity = True
    review_document_type = "academic_paper"

    def effective_paper_type(self, requested: str) -> str:
        return requested

    def effective_structure_pattern(self, requested: str) -> str:
        return requested

    def default_outline(self, topic: str, lang: str) -> dict[str, Any] | None:
        return None

    def normalize_outline(
        self, outline: OutlineStructure, topic: str
    ) -> OutlineStructure:
        return outline

    def planner_context_suffix(self, target_word_count: str) -> str:
        return ""

    def writer_mode(self, has_revision_context: bool) -> WriterMode:
        return "academic_revision" if has_revision_context else "draft"

    def writer_prompts(self, prompts: Any) -> ProfileWriterPrompts | None:
        return None

    def review_policy_text(self) -> str:
        return "Review as an academic paper section."

    def missing_core_tasks(self, section_title: str, content: str) -> list[str]:
        return []

    def empty_reference_issue(self) -> QualityIssue | None:
        return None

    def structural_review_issues(
        self, section_id: str, section_title: str, content: str
    ) -> tuple[list[ReviewIssue], list[QualityIssue]]:
        return [], []

    def apply_review_policy(
        self, review: ReviewResult, quality_issues: list[QualityIssue]
    ) -> tuple[ReviewResult, list[QualityIssue]]:
        return review, quality_issues

    def outline_section_issues(
        self, section: SectionOutline, *, is_last: bool
    ) -> list[QualityIssue]:
        return []

    def outline_structure_issues(self, outline: OutlineStructure) -> list[QualityIssue]:
        return []

    def skip_methodology_audit(self, content: str) -> bool:
        return False


def get_document_profile(state: Mapping[str, Any]) -> DocumentProfile:
    if is_research_proposal(state):
        from .research_proposal import ResearchProposalProfile

        return ResearchProposalProfile()
    return DefaultDocumentProfile()


def get_default_specialized_writer_prompts(prompts: Any) -> ProfileWriterPrompts:
    from .research_proposal import ResearchProposalProfile

    profile_prompts = ResearchProposalProfile().writer_prompts(prompts)
    if profile_prompts is None:
        raise RuntimeError("specialized writer prompts are not configured")
    return profile_prompts
