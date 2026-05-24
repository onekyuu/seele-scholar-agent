from collections.abc import AsyncIterator
from typing import Any

from ..logging import get_logger
from ..state import AgentState, OutlineStructure, QualityIssue, SectionEvidencePlan, SectionOutline
from . import NodeStreamEvent

logger = get_logger(__name__)

_NON_EMPIRICAL_PAPER_TYPES = {
    "literature_review",
    "theoretical",
    "theoretical_analysis",
    "policy_brief",
    "survey",
    "review",
}
_IMRAD_PATTERNS = {"imrad", "experimental", "empirical"}
_EXPERIMENTAL_TITLE_MARKERS = (
    "method",
    "methodology",
    "experiment",
    "experiments",
    "result",
    "results",
    "方法",
    "实验",
    "结果",
    "実験",
    "結果",
)


class OutlineQualityGateNode:
    """Runtime gate for planner outline completeness and structure fit."""

    async def check(self, state: AgentState) -> dict[str, Any]:
        outline = state.get("outline")
        if outline is None:
            issue = self._blocking_issue(
                "OUTLINE_MISSING",
                "Planner did not produce an outline.",
                "outline",
            )
            return self._blocked([issue])

        issues = self._outline_issues(outline, state)
        blocking_issues = [
            issue for issue in issues if issue.blocking or issue.severity == "blocking"
        ]
        if blocking_issues:
            logger.warning(
                "outline quality gate blocked writing",
                issue_codes=[issue.code for issue in blocking_issues],
            )
            return self._blocked(issues)

        logger.info("outline quality gate passed", warning_count=len(issues))
        return {"quality_issues": issues}

    async def astream(self, state: AgentState) -> AsyncIterator[NodeStreamEvent]:
        yield NodeStreamEvent(type="progress", progress="checking_outline_quality")
        result = await self.check(state)
        yield NodeStreamEvent(type="result", result=result)

    def _outline_issues(
        self, outline: OutlineStructure, state: AgentState
    ) -> list[QualityIssue]:
        issues: list[QualityIssue] = []
        if not outline.sections:
            return [
                self._blocking_issue(
                    "OUTLINE_EMPTY_SECTIONS",
                    "Outline has no sections.",
                    "outline.sections",
                )
            ]

        evidence_plan_by_title = {plan.section_title: plan for plan in outline.evidence_map}
        for index, section in enumerate(sorted(outline.sections, key=lambda item: item.order)):
            issues.extend(
                self._section_issues(
                    section,
                    evidence_plan_by_title.get(section.title),
                    is_last=index == len(outline.sections) - 1,
                )
            )

        structure_issue = self._structure_fit_issue(outline, state)
        if structure_issue is not None:
            issues.append(structure_issue)
        return issues

    def _section_issues(
        self,
        section: SectionOutline,
        evidence_plan: SectionEvidencePlan | None,
        *,
        is_last: bool,
    ) -> list[QualityIssue]:
        issues: list[QualityIssue] = []
        location = f"outline.sections.{section.order}"
        if not section.purpose.strip():
            issues.append(
                self._blocking_issue(
                    "OUTLINE_MISSING_PURPOSE",
                    f"Section '{section.title}' is missing purpose.",
                    location,
                )
            )
        if not is_last and not section.transition_to_next.strip():
            issues.append(
                self._blocking_issue(
                    "OUTLINE_MISSING_TRANSITION",
                    f"Section '{section.title}' is missing transition_to_next.",
                    location,
                )
            )
        if not section.target_claims:
            issues.append(
                self._blocking_issue(
                    "OUTLINE_MISSING_TARGET_CLAIMS",
                    f"Section '{section.title}' has no target claims.",
                    location,
                )
            )
        elif not section.key_sources and not section.citation_plan:
            issues.append(
                self._blocking_issue(
                    "OUTLINE_CLAIMS_WITHOUT_EVIDENCE_PLAN",
                    (
                        f"Section '{section.title}' has target claims without sources "
                        "or citation plan."
                    ),
                    location,
                )
            )

        if evidence_plan is None:
            issues.append(
                self._blocking_issue(
                    "OUTLINE_MISSING_EVIDENCE_MAP",
                    f"Section '{section.title}' has no evidence map entry.",
                    location,
                )
            )
        else:
            issues.extend(self._evidence_map_issues(section, evidence_plan, location))

        if section.evidence_gaps:
            issues.append(
                QualityIssue(
                    code="OUTLINE_EVIDENCE_GAPS",
                    message=f"Section '{section.title}' still has evidence gaps.",
                    severity="warning",
                    location=location,
                    blocking=False,
                    details={"evidence_gaps": section.evidence_gaps},
                )
            )
        return issues

    def _evidence_map_issues(
        self, section: SectionOutline, evidence_plan: SectionEvidencePlan, location: str
    ) -> list[QualityIssue]:
        issues: list[QualityIssue] = []
        missing_claims = [
            claim for claim in section.target_claims if claim not in evidence_plan.target_claims
        ]
        if missing_claims:
            issues.append(
                self._blocking_issue(
                    "OUTLINE_EVIDENCE_MAP_MISSING_CLAIMS",
                    f"Evidence map for section '{section.title}' does not cover all target claims.",
                    location,
                    details={"missing_claims": missing_claims},
                )
            )
        if (
            section.target_claims
            and not evidence_plan.citation_plan
            and not evidence_plan.key_sources
        ):
            issues.append(
                self._blocking_issue(
                    "OUTLINE_EVIDENCE_MAP_WITHOUT_SOURCES",
                    f"Evidence map for section '{section.title}' has no source or citation plan.",
                    location,
                )
            )
        return issues

    def _structure_fit_issue(
        self, outline: OutlineStructure, state: AgentState
    ) -> QualityIssue | None:
        requested_type = str(state.get("paper_type", "") or "").casefold()
        outline_type = (outline.paper_type or "").casefold()
        structure_pattern = (outline.structure_pattern or "").casefold()
        effective_type = (
            requested_type if requested_type and requested_type != "auto" else outline_type
        )
        if effective_type not in _NON_EMPIRICAL_PAPER_TYPES:
            return None

        section_titles = " ".join(section.title.casefold() for section in outline.sections)
        experimental_marker_count = sum(
            1 for marker in _EXPERIMENTAL_TITLE_MARKERS if marker in section_titles
        )
        if structure_pattern in _IMRAD_PATTERNS or experimental_marker_count >= 2:
            return self._blocking_issue(
                "OUTLINE_EXPERIMENTAL_TEMPLATE_MISMATCH",
                (
                    "Outline appears to use an empirical/IMRaD structure for a "
                    f"non-empirical paper type '{effective_type}'."
                ),
                "outline.structure_pattern",
                details={
                    "paper_type": effective_type,
                    "structure_pattern": outline.structure_pattern,
                },
            )
        return None

    def _blocked(self, issues: list[QualityIssue]) -> dict[str, Any]:
        first_blocking = next(
            (issue for issue in issues if issue.blocking or issue.severity == "blocking"),
            issues[0],
        )
        return {
            "status": "waiting_human",
            "error_message": first_blocking.message,
            "quality_issues": issues,
        }

    def _blocking_issue(
        self,
        code: str,
        message: str,
        location: str,
        *,
        details: dict[str, Any] | None = None,
    ) -> QualityIssue:
        return QualityIssue(
            code=code,
            message=message,
            severity="blocking",
            location=location,
            blocking=True,
            details=details or {},
        )
