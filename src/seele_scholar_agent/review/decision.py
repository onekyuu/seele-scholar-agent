from typing import Any, Literal

from pydantic import BaseModel, Field

from ..policy.execution_strategy import GenerationMode
from ..policy.writing_policy import WritingPolicy
from ..state import QualityIssue, ReviewResult


class ReviewDecision(BaseModel):
    action: Literal["pass", "revise", "accept_best", "human_required"]
    blocking_issues: list[QualityIssue] = Field(default_factory=list)
    revision_scope: Literal["none", "targeted", "full_section"] = "none"
    reason: str = ""


class SectionCandidate(BaseModel):
    section_id: str
    content: str
    revision_count: int
    review_score: int
    blocking_issues: list[QualityIssue] = Field(default_factory=list)
    warnings: list[QualityIssue] = Field(default_factory=list)
    length: int = 0
    diagnostics: dict[str, Any] = Field(default_factory=dict)


def decide_review_action(
    review: ReviewResult,
    quality_issues: list[QualityIssue],
    *,
    revision_count: int,
    max_revisions: int,
    writing_policy: WritingPolicy,
    generation_mode: GenerationMode,
) -> ReviewDecision:
    blocking = [issue for issue in quality_issues if issue.blocking or issue.severity == "blocking"]
    if review.approved and not blocking:
        return ReviewDecision(action="pass", reason="review_approved")

    if revision_count >= max_revisions:
        if (
            writing_policy.on_max_revisions == "accept_best_with_report"
            and generation_mode == GenerationMode.FULL_DOCUMENT
        ):
            return ReviewDecision(
                action="accept_best",
                blocking_issues=blocking,
                revision_scope="none",
                reason="max_revisions_accept_best_with_report",
            )
        return ReviewDecision(
            action="human_required",
            blocking_issues=blocking,
            revision_scope="none",
            reason="max_revisions_requires_human",
        )

    return ReviewDecision(
        action="revise",
        blocking_issues=blocking,
        revision_scope="targeted",
        reason="review_failed",
    )


def select_best_candidate(candidates: list[SectionCandidate]) -> SectionCandidate:
    if not candidates:
        raise ValueError("cannot select a best candidate from an empty list")
    return max(candidates, key=_candidate_score)


def _candidate_score(candidate: SectionCandidate) -> float:
    blocking_penalty = 2.0 * len(candidate.blocking_issues)
    warning_penalty = 0.5 * len(candidate.warnings)
    citation_penalty = sum(
        1.5
        for issue in [*candidate.blocking_issues, *candidate.warnings]
        if "CITATION" in issue.code or "CLAIM" in issue.code
    )
    return candidate.review_score - blocking_penalty - warning_penalty - citation_penalty
