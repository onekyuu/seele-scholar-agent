from typing import Any

from pydantic import ValidationError

from ..state import AgentState, QualityIssue
from .mapping import build_draft_state
from .models import ExistingContentRef, coerce_existing_content_ref

_DRAFT_INPUT_INVALID = "DRAFT_INPUT_INVALID"
_DRAFT_SEGMENTS_REQUIRED = "DRAFT_SEGMENTS_REQUIRED"


class DraftIntegrationNode:
    """Validate and map structured user draft input."""

    def integrate(self, state: AgentState) -> dict[str, Any]:
        if state.get("existing_content") is None:
            return {}

        existing_content = coerce_existing_content_ref(state.get("existing_content"))
        if existing_content is None:
            return _blocking_result(
                state,
                code=_DRAFT_INPUT_INVALID,
                message=(
                    "existing_content must match ExistingContentRef, including "
                    "draft_id, version_id, segments, preserve_policy, and user_intent."
                ),
            )

        if not existing_content.segments:
            preview_available = bool(
                existing_content.normalized_content or existing_content.raw_content_preview
            )
            message = (
                "Structured draft input requires paragraph-level DraftSegment entries; "
                "raw or normalized full text alone is not accepted."
                if preview_available
                else "Structured draft input requires at least one DraftSegment."
            )
            return _blocking_result(
                state,
                code=_DRAFT_SEGMENTS_REQUIRED,
                message=message,
            )

        try:
            ExistingContentRef.model_validate(existing_content)
        except ValidationError as exc:
            return _blocking_result(
                state,
                code=_DRAFT_INPUT_INVALID,
                message=f"existing_content validation failed: {exc}",
            )

        draft_state = build_draft_state(
            existing_content,
            outline=state.get("outline"),
            sections=list(state.get("sections", []) or []),
        )
        return {
            "draft_integration": draft_state,
            "draft_diagnostics": {
                "status": "ok",
                "draft_id": existing_content.draft_id,
                "segment_count": len(existing_content.segments),
                "mapped_count": sum(
                    1 for mapping in draft_state.mappings if mapping.section_id is not None
                ),
                "outline_action": (
                    draft_state.outline_decision.action
                    if draft_state.outline_decision is not None
                    else None
                ),
            },
        }


def _blocking_result(
    state: AgentState,
    *,
    code: str,
    message: str,
) -> dict[str, Any]:
    issue = QualityIssue(
        code=code,
        message=message,
        severity="blocking",
        blocking=True,
        details={"recommended_action": "Provide structured DraftSegment input."},
    )
    issues = [issue for issue in state.get("quality_issues", []) if issue.code != code]
    issues.append(issue)
    return {
        "quality_issues": issues,
        "quality_issue_history": [issue],
        "draft_diagnostics": {"status": "invalid", "code": code},
        "status": "failed",
    }
