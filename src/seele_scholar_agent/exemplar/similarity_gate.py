from difflib import SequenceMatcher
from typing import Any

from ..state import AgentState, QualityIssue
from .models import ExemplarPolicy, coerce_exemplar_context

_COPY_RISK_CODE = "EXEMPLAR_COPY_RISK"


class SimilarityGateNode:
    """Warn when generated section text is too close to an exemplar chunk."""

    def __init__(self, policy: ExemplarPolicy | None = None) -> None:
        self.policy = policy or ExemplarPolicy()

    def check(self, state: AgentState) -> dict[str, Any]:
        if not self.policy.enabled:
            return {}

        sections = state.get("sections", [])
        current_index = state.get("current_section_index", 0)
        if current_index < 0 or current_index >= len(sections):
            return {"exemplar_similarity_diagnostics": {"status": "no_current_section"}}

        section = sections[current_index]
        context = coerce_exemplar_context(state.get("exemplar_context"))
        diagnostics = {
            "status": "ok",
            "section_id": section.section_id,
            "threshold": self.policy.max_similarity_ratio,
            "max_similarity_ratio": 0.0,
            "matched_exemplar_chunk_id": None,
        }
        existing_issues = _without_current_copy_risk(
            list(state.get("quality_issues", []) or []),
            section.section_id,
        )

        if not section.content or not context.section_examples:
            return {
                "quality_issues": existing_issues,
                "exemplar_similarity_diagnostics": diagnostics,
            }

        best_ratio = 0.0
        best_chunk_id: str | None = None
        for chunk in context.section_examples:
            ratio = _similarity_ratio(section.content, chunk.text)
            if ratio > best_ratio:
                best_ratio = ratio
                best_chunk_id = chunk.chunk_id

        diagnostics["max_similarity_ratio"] = best_ratio
        diagnostics["matched_exemplar_chunk_id"] = best_chunk_id
        if best_ratio <= self.policy.max_similarity_ratio:
            return {
                "quality_issues": existing_issues,
                "exemplar_similarity_diagnostics": diagnostics,
            }

        diagnostics["status"] = "copy_risk"
        issue = QualityIssue(
            code=_COPY_RISK_CODE,
            message=(
                "Current section is too similar to an exemplar; rewrite wording while "
                "keeping only the intended structure/style signal."
            ),
            severity="warning",
            location=section.section_id,
            blocking=False,
            details=diagnostics,
        )
        return {
            "quality_issues": [*existing_issues, issue],
            "quality_issue_history": [issue],
            "exemplar_similarity_diagnostics": diagnostics,
        }


def _similarity_ratio(left: str, right: str) -> float:
    left_normalized = " ".join(left.split())
    right_normalized = " ".join(right.split())
    if not left_normalized or not right_normalized:
        return 0.0
    return SequenceMatcher(None, left_normalized, right_normalized).ratio()


def _without_current_copy_risk(
    issues: list[QualityIssue], section_id: str
) -> list[QualityIssue]:
    return [
        issue
        for issue in issues
        if not (
            issue.code == _COPY_RISK_CODE
            and (issue.location == section_id or issue.details.get("section_id") == section_id)
        )
    ]
