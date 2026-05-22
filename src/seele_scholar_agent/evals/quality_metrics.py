import re
from dataclasses import dataclass
from difflib import SequenceMatcher

from ..nodes import CITATION_PATTERN
from ..state import AgentState, ClaimEvidenceBinding, OutlineStructure, ReferenceEntry, SectionDraft

_WORD_RE = re.compile(r"[\w\u4e00-\u9fff]+")
_TARGET_CLAIM_COVERAGE_THRESHOLD = 0.4


@dataclass(frozen=True)
class QualityMetrics:
    """Small benchmark metrics for generated academic paper quality."""

    citation_validity_rate: float
    chunk_support_rate: float
    reviewer_pass_rate: float
    duplicate_paragraph_ratio: float
    section_target_coverage: float
    user_edit_ratio: float | None = None


def evaluate_quality(
    state: AgentState,
    *,
    user_edited_text: str | None = None,
    generated_text: str | None = None,
) -> QualityMetrics:
    """Evaluate a generated paper state with lightweight deterministic metrics."""

    sections = state.get("sections", [])
    references = state.get("references", [])
    bindings = state.get("claim_evidence_bindings", [])
    review_history = state.get("review_history", [])
    outline = state.get("outline")

    combined_generated_text = generated_text or "\n\n".join(section.content for section in sections)
    return QualityMetrics(
        citation_validity_rate=_citation_validity_rate(sections, references),
        chunk_support_rate=_chunk_support_rate(bindings),
        reviewer_pass_rate=_reviewer_pass_rate(review_history),
        duplicate_paragraph_ratio=_duplicate_paragraph_ratio(combined_generated_text),
        section_target_coverage=_section_target_coverage(sections, outline),
        user_edit_ratio=_user_edit_ratio(combined_generated_text, user_edited_text),
    )


def _citation_validity_rate(
    sections: list[SectionDraft], references: list[ReferenceEntry]
) -> float:
    cited_numbers = [
        int(num) for section in sections for num in CITATION_PATTERN.findall(section.content)
    ]
    if not cited_numbers:
        return 0.0

    valid_numbers = {reference.number for reference in references}
    valid_count = sum(1 for number in cited_numbers if number in valid_numbers)
    return valid_count / len(cited_numbers)


def _chunk_support_rate(bindings: list[ClaimEvidenceBinding]) -> float:
    if not bindings:
        return 0.0

    supported = sum(
        1
        for binding in bindings
        if binding.chunk_id and binding.verdict == "supported" and binding.support_score > 0.0
    )
    return supported / len(bindings)


def _reviewer_pass_rate(review_history: list[dict[str, object]]) -> float:
    if not review_history:
        return 0.0

    approved_count = sum(1 for review in review_history if review.get("approved") is True)
    return approved_count / len(review_history)


def _duplicate_paragraph_ratio(text: str) -> float:
    paragraphs = [_normalize_paragraph(paragraph) for paragraph in text.split("\n\n")]
    paragraphs = [paragraph for paragraph in paragraphs if paragraph]
    if len(paragraphs) <= 1:
        return 0.0

    duplicate_count = len(paragraphs) - len(set(paragraphs))
    return duplicate_count / len(paragraphs)


def _section_target_coverage(
    sections: list[SectionDraft], outline: OutlineStructure | None
) -> float:
    if outline is None:
        return 0.0

    section_by_title = {section.title: section for section in sections}
    target_scores: list[float] = []
    for planned_section in outline.sections:
        section = section_by_title.get(planned_section.title)
        if section is None:
            target_scores.extend(0.0 for _ in planned_section.target_claims)
            continue

        content_tokens = _tokens(section.content)
        for claim in planned_section.target_claims:
            target_scores.append(_claim_covered(claim, content_tokens))

    if not target_scores:
        return 0.0
    return sum(target_scores) / len(target_scores)


def _claim_covered(claim: str, content_tokens: set[str]) -> float:
    claim_tokens = _tokens(claim)
    if not claim_tokens:
        return 0.0

    overlap = claim_tokens & content_tokens
    overlap_ratio = len(overlap) / len(claim_tokens)
    return 1.0 if overlap_ratio >= _TARGET_CLAIM_COVERAGE_THRESHOLD else 0.0


def _user_edit_ratio(generated_text: str, user_edited_text: str | None) -> float | None:
    if user_edited_text is None:
        return None

    if not generated_text and not user_edited_text:
        return 0.0

    similarity = SequenceMatcher(None, generated_text, user_edited_text).ratio()
    return 1.0 - similarity


def _normalize_paragraph(paragraph: str) -> str:
    return " ".join(paragraph.casefold().split())


def _tokens(text: str) -> set[str]:
    return {match.group(0).casefold() for match in _WORD_RE.finditer(text)}
