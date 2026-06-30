import re
from typing import Any

from ..state import AgentState, QualityIssue, SectionDraft
from ..writing.writer_input import WriterInput
from .mapping import build_draft_section_context
from .models import DraftSectionContext, DraftSegment, coerce_draft_integration_state

_WORD_RE = re.compile(r"[A-Za-z0-9_\u4e00-\u9fff]+")
_PRESERVATION_CODE = "DRAFT_PROTECTED_SEGMENT_REMOVED"
_COVERAGE_CODE = "DRAFT_COVERAGE_GAP"
_CONFLICT_CODE = "DRAFT_OUTLINE_CONFLICT"


class PreservationGate:
    """Report protected draft segments that disappeared from the generated section."""

    def check(self, state: AgentState) -> dict[str, Any]:
        context = _draft_context(state)
        section = _current_section(state)
        if context is None or section is None:
            return {}

        protected_segments = [
            segment
            for segment in _all_context_segments(context)
            if segment.segment_id in context.preserve_policy.protected_segment_ids
        ]
        missing = [
            segment.segment_id
            for segment in protected_segments
            if not _segment_represented(segment, section.content)
        ]
        return _gate_result(
            state,
            section=section,
            code=_PRESERVATION_CODE,
            missing_ids=missing,
            message="Protected draft segments are not represented in the generated section.",
            diagnostics_key="preservation",
        )


class CoverageGate:
    """Report mapped draft content that appears uncovered by the generated section."""

    def check(self, state: AgentState) -> dict[str, Any]:
        context = _draft_context(state)
        section = _current_section(state)
        if context is None or section is None:
            return {}
        if context.user_intent == "reference_only":
            return {
                "draft_gate_diagnostics": {
                    "coverage": {"status": "skipped", "reason": "reference_only"}
                }
            }

        missing = [
            segment.segment_id
            for segment in context.mapped_segments
            if not _segment_represented(segment, section.content, threshold=0.2)
        ]
        return _gate_result(
            state,
            section=section,
            code=_COVERAGE_CODE,
            missing_ids=missing,
            message="Mapped draft segments may not be covered by the generated section.",
            diagnostics_key="coverage",
        )


class ConflictGate:
    """Surface conflicts detected while integrating user draft input."""

    def check(self, state: AgentState) -> dict[str, Any]:
        draft_state = coerce_draft_integration_state(state.get("draft_integration"))
        section = _current_section(state)
        if draft_state is None or section is None or not draft_state.conflicts:
            return {}

        existing_issues = _without_code(
            list(state.get("quality_issues", []) or []), _CONFLICT_CODE, section.section_id
        )
        issue = QualityIssue(
            code=_CONFLICT_CODE,
            message="User draft conflicts should be reviewed before finalizing this section.",
            severity="warning",
            location=section.section_id,
            blocking=False,
            details={
                "section_id": section.section_id,
                "conflicts": list(draft_state.conflicts),
                "recommended_action": "Review draft-to-outline mapping and resolve conflicts.",
            },
        )
        return {
            "quality_issues": [*existing_issues, issue],
            "quality_issue_history": [issue],
            "draft_gate_diagnostics": {
                "conflict": {"status": "conflict", "conflicts": list(draft_state.conflicts)}
            },
        }


def _draft_context(state: AgentState) -> DraftSectionContext | None:
    writer_input = state.get("writer_input")
    if isinstance(writer_input, WriterInput):
        return writer_input.draft_context
    if isinstance(writer_input, dict):
        try:
            return WriterInput.model_validate(writer_input).draft_context
        except ValueError:
            return None

    draft_state = coerce_draft_integration_state(state.get("draft_integration"))
    return build_draft_section_context(
        draft_state,
        sections=list(state.get("sections", []) or []),
        current_index=int(state.get("current_section_index", 0)),
    )


def _current_section(state: AgentState) -> SectionDraft | None:
    sections = list(state.get("sections", []) or [])
    current_index = int(state.get("current_section_index", 0))
    if current_index < 0 or current_index >= len(sections):
        return None
    return sections[current_index]


def _all_context_segments(context: DraftSectionContext) -> list[DraftSegment]:
    return [*context.mapped_segments, *context.unmapped_related_segments]


def _segment_represented(
    segment: DraftSegment, content: str, *, threshold: float = 0.3
) -> bool:
    if not segment.text.strip() or not content.strip():
        return False
    if segment.text.strip() in content:
        return True
    return _token_overlap_score(segment.text, content) >= threshold


def _gate_result(
    state: AgentState,
    *,
    section: SectionDraft,
    code: str,
    missing_ids: list[str],
    message: str,
    diagnostics_key: str,
) -> dict[str, Any]:
    existing_issues = _without_code(
        list(state.get("quality_issues", []) or []), code, section.section_id
    )
    diagnostics = {
        "status": "ok" if not missing_ids else "warning",
        "section_id": section.section_id,
        "missing_segment_ids": missing_ids,
    }
    if not missing_ids:
        return {
            "quality_issues": existing_issues,
            "draft_gate_diagnostics": {diagnostics_key: diagnostics},
        }

    issue = QualityIssue(
        code=code,
        message=message,
        severity="warning",
        location=section.section_id,
        blocking=False,
        details={
            **diagnostics,
            "recommended_action": "Review generated content against preserved draft segments.",
        },
    )
    return {
        "quality_issues": [*existing_issues, issue],
        "quality_issue_history": [issue],
        "draft_gate_diagnostics": {diagnostics_key: diagnostics},
    }


def _without_code(
    issues: list[QualityIssue], code: str, section_id: str
) -> list[QualityIssue]:
    return [
        issue
        for issue in issues
        if not (
            issue.code == code
            and (issue.location == section_id or issue.details.get("section_id") == section_id)
        )
    ]


def _token_overlap_score(left: str, right: str) -> float:
    left_tokens = {token.lower() for token in _WORD_RE.findall(left)}
    right_tokens = {token.lower() for token in _WORD_RE.findall(right)}
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / len(left_tokens)
