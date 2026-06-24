from collections.abc import AsyncIterator
from typing import Any

from ..document_profile import is_research_proposal, is_schedule_section, missing_schedule_phases
from ..logging import get_logger
from ..state import AgentState, OutlineStructure, QualityIssue, SectionEvidencePlan, SectionOutline
from . import CITATION_PATTERN, NodeStreamEvent
from .material_registry import (
    find_material_entry,
    find_required_entry_number,
    get_material_registry,
    material_display_name,
    required_entries,
)

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

        proposal_profile = is_research_proposal(state)
        evidence_plan_by_title = {plan.section_title: plan for plan in outline.evidence_map}
        for index, section in enumerate(sorted(outline.sections, key=lambda item: item.order)):
            if proposal_profile:
                issues.extend(
                    self._proposal_section_issues(
                        section,
                        is_last=index == len(outline.sections) - 1,
                    )
                )
            else:
                issues.extend(
                    self._section_issues(
                        section,
                        evidence_plan_by_title.get(section.title),
                        is_last=index == len(outline.sections) - 1,
                    )
                )

        if proposal_profile:
            issues.extend(self._proposal_structure_issues(outline))
        else:
            structure_issue = self._structure_fit_issue(outline, state)
            if structure_issue is not None:
                issues.append(structure_issue)
        issues.extend(self._material_registry_issues(outline, state))
        return issues

    def _proposal_section_issues(
        self,
        section: SectionOutline,
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
                QualityIssue(
                    code="OUTLINE_MISSING_TRANSITION",
                    message=f"Section '{section.title}' is missing transition_to_next.",
                    severity="warning",
                    location=location,
                    blocking=False,
                )
            )
        if section.target_words is None:
            issues.append(
                QualityIssue(
                    code="OUTLINE_MISSING_TARGET_WORDS",
                    message=f"Section '{section.title}' has no proposal length budget.",
                    severity="warning",
                    location=location,
                    blocking=False,
                )
            )
        return issues

    def _proposal_structure_issues(self, outline: OutlineStructure) -> list[QualityIssue]:
        issues: list[QualityIssue] = []
        sections = sorted(outline.sections, key=lambda item: item.order)
        if len(sections) < 4 or len(sections) > 5:
            issues.append(
                QualityIssue(
                    code="PROPOSAL_SECTION_COUNT_OUT_OF_RANGE",
                    message="Research proposal outline should usually have 4-5 sections.",
                    severity="warning",
                    location="outline.sections",
                    blocking=False,
                    details={"section_count": len(sections)},
                )
            )

        schedule_sections = [section for section in sections if is_schedule_section(section.title)]
        if not schedule_sections:
            issues.append(
                self._blocking_issue(
                    "PROPOSAL_SCHEDULE_SECTION_MISSING",
                    "Research proposal outline is missing a schedule section.",
                    "outline.sections",
                )
            )
            return issues

        schedule = schedule_sections[0]
        schedule_text = "\n".join(
            [
                schedule.title,
                schedule.description,
                schedule.content_summary,
                " ".join(schedule.key_points),
            ]
        )
        missing = missing_schedule_phases(schedule_text)
        if missing:
            issues.append(
                self._blocking_issue(
                    "PROPOSAL_SCHEDULE_PHASES_MISSING",
                    "Schedule outline is missing phases: " + ", ".join(missing),
                    f"outline.sections.{schedule.order}",
                    details={"missing_phases": missing},
                )
            )
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

    def _material_registry_issues(
        self, outline: OutlineStructure, state: AgentState
    ) -> list[QualityIssue]:
        registry = get_material_registry(state)
        if registry is None or not registry.entries:
            return []

        papers = state.get("papers", [])
        planned_numbers = self._planned_citation_numbers(outline)
        issues: list[QualityIssue] = []

        for number in sorted(planned_numbers):
            if number < 1 or number > len(papers):
                issues.append(
                    self._blocking_issue(
                        "OUTLINE_CITES_UNKNOWN_MATERIAL",
                        f"Outline evidence plan cites [{number}], but no such paper exists.",
                        f"[{number}]",
                    )
                )
                continue

            paper = papers[number - 1]
            entry = find_material_entry(paper, registry)
            if entry is None:
                continue
            if entry.citation_role != "citable":
                issues.append(
                    self._blocking_issue(
                        "OUTLINE_CITES_NON_CITABLE_MATERIAL",
                        (
                            f"Outline cites [{number}] '{paper.title}', but the material "
                            f"is marked as {entry.citation_role}."
                        ),
                        f"[{number}]",
                        details={"paper_id": paper.paper_id, "citation_role": entry.citation_role},
                    )
                )
            elif entry.confidence == "low":
                issues.append(
                    QualityIssue(
                        code="OUTLINE_LOW_CONFIDENCE_MATERIAL",
                        message=f"Outline plans to cite low-confidence material [{number}].",
                        severity="warning",
                        location=f"[{number}]",
                        blocking=False,
                        details={"paper_id": paper.paper_id},
                    )
                )

        for entry in required_entries(registry):
            required_number = find_required_entry_number(entry, papers)
            if required_number is None:
                issues.append(
                    self._blocking_issue(
                        "REQUIRED_MATERIAL_NOT_FOUND",
                        (
                            "Required user material was not found in retrieved papers: "
                            f"{material_display_name(entry)}."
                        ),
                        "material_registry",
                    )
                )
                continue

            if entry.citation_role != "citable":
                issues.append(
                    self._blocking_issue(
                        "REQUIRED_MATERIAL_NOT_CITABLE",
                        (
                            f"Required material [{required_number}] is marked as "
                            f"{entry.citation_role}, so it cannot be cited."
                        ),
                        f"[{required_number}]",
                    )
                )
                continue

            if required_number not in planned_numbers:
                issues.append(
                    self._blocking_issue(
                        "REQUIRED_MATERIAL_NOT_PLANNED",
                        (
                            f"Required user material [{required_number}] is not included "
                            "in the outline evidence plan."
                        ),
                        f"[{required_number}]",
                    )
                )

            paper = papers[required_number - 1]
            if (
                state.get("check_required_material_relevance", False)
                and paper.relevance_score < 0.15
                and paper.query_overlap_score < 0.1
            ):
                issues.append(
                    QualityIssue(
                        code="REQUIRED_MATERIAL_LOW_RELEVANCE",
                        message=(
                            f"Required material [{required_number}] has low relevance "
                            "signals for the current topic."
                        ),
                        severity="warning",
                        location=f"[{required_number}]",
                        blocking=False,
                        details={
                            "paper_id": paper.paper_id,
                            "relevance_score": paper.relevance_score,
                            "query_overlap_score": paper.query_overlap_score,
                        },
                    )
                )
        return issues

    def _planned_citation_numbers(self, outline: OutlineStructure) -> set[int]:
        planned_text_parts: list[str] = []
        for section in outline.sections:
            planned_text_parts.extend(section.key_sources)
            planned_text_parts.extend(section.citation_plan)
        for evidence_plan in outline.evidence_map:
            planned_text_parts.extend(evidence_plan.key_sources)
            planned_text_parts.extend(evidence_plan.citation_plan)
        return {int(number) for number in CITATION_PATTERN.findall("\n".join(planned_text_parts))}

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
