from typing import Any, Literal

from pydantic import BaseModel, Field

from ..state import QualityIssue


class QualityReport(BaseModel):
    document_status: Literal["passed", "completed_with_issues"]
    section_statuses: dict[str, Literal["passed", "accepted_with_issues"]]
    unresolved_issues: list[QualityIssue] = Field(default_factory=list)
    recommended_actions: list[str] = Field(default_factory=list)


def build_quality_report(state: dict[str, Any]) -> QualityReport:
    sections = state.get("sections", [])
    unresolved_issues = list(state.get("quality_issues") or [])
    section_statuses: dict[str, Literal["passed", "accepted_with_issues"]] = {}

    for section in sections:
        status = getattr(section, "status", "")
        if status == "accepted_with_issues":
            section_statuses[getattr(section, "section_id", getattr(section, "title", ""))] = (
                "accepted_with_issues"
            )
        elif status == "approved":
            section_statuses[getattr(section, "section_id", getattr(section, "title", ""))] = (
                "passed"
            )

    has_issues = bool(unresolved_issues) or any(
        status == "accepted_with_issues" for status in section_statuses.values()
    )
    return QualityReport(
        document_status="completed_with_issues" if has_issues else "passed",
        section_statuses=section_statuses,
        unresolved_issues=unresolved_issues,
        recommended_actions=_recommended_actions(unresolved_issues),
    )


def _recommended_actions(issues: list[QualityIssue]) -> list[str]:
    actions: list[str] = []
    seen: set[str] = set()
    for issue in issues:
        action = issue.details.get("recommended_action")
        if isinstance(action, str) and action and action not in seen:
            seen.add(action)
            actions.append(action)
            continue
        fallback = f"Review {issue.location or issue.code}: {issue.message}"
        if fallback not in seen:
            seen.add(fallback)
            actions.append(fallback)
    return actions
