from ..budget import BudgetState, SectionBudget
from ..exemplar.models import ExemplarContext, coerce_exemplar_context
from ..state import (
    AgentState,
    EvidencePacket,
    OutlineStructure,
    SectionDraft,
    SectionOutline,
)
from .writer_input import OutlineContext, SectionBrief, SectionWritingSpec, WriterInput


class WriterInputBuilder:
    def build(
        self,
        state: AgentState,
        *,
        current_index: int,
        evidence_packets: list[EvidencePacket],
        style_context: str,
    ) -> WriterInput:
        sections = state["sections"]
        section = sections[current_index]
        outline = state.get("outline")
        section_outline = _find_section_outline(outline, section)
        return WriterInput(
            topic=state["topic"],
            language=state.get("language", "zh"),
            outline_context=_outline_context(outline, sections),
            current_section=_section_spec(section, section_outline, state),
            previous_section_summaries=_previous_summaries(state, current_index),
            citation_sources=_citation_sources(state),
            papers=list(state.get("papers", []) or []),
            evidence_packets=evidence_packets,
            review_comments=list(section.review_comments),
            style_context=style_context,
            exemplar_context=_exemplar_context(state),
        )


def _outline_context(
    outline: OutlineStructure | None, sections: list[SectionDraft]
) -> OutlineContext:
    if outline is None:
        return OutlineContext(
            title="",
            sections=[
                SectionBrief(
                    section_id=section.section_id,
                    title=section.title,
                    order=section.order_index,
                )
                for section in sections
            ],
        )
    return OutlineContext(
        title=outline.title,
        abstract=outline.abstract,
        keywords=list(outline.keywords),
        paper_type=outline.paper_type,
        structure_pattern=outline.structure_pattern,
        rationale=outline.rationale,
        sections=[
            SectionBrief(
                section_id=f"section_{index}",
                title=section.title,
                order=section.order,
                purpose=section.purpose,
                content_summary=section.content_summary,
            )
            for index, section in enumerate(outline.sections)
        ],
    )


def _section_spec(
    section: SectionDraft,
    section_outline: SectionOutline | None,
    state: AgentState,
) -> SectionWritingSpec:
    budget = _section_budget(section.section_id, section_outline, state)
    if section_outline is None:
        return SectionWritingSpec(
            section_id=section.section_id,
            title=section.title,
            order=section.order_index,
            description=section.description,
            budget=budget,
        )
    return SectionWritingSpec(
        section_id=section.section_id,
        title=section.title,
        order=section.order_index,
        description=section.description or section_outline.description,
        purpose=section_outline.purpose,
        content_summary=section_outline.content_summary,
        key_points=list(section_outline.key_points),
        target_claims=list(section_outline.target_claims),
        key_sources=list(section_outline.key_sources),
        citation_plan=list(section_outline.citation_plan),
        evidence_gaps=list(section_outline.evidence_gaps),
        transition_to_next=section_outline.transition_to_next,
        suggested_figures=list(section_outline.suggested_figures),
        section_style=section_outline.section_style,
        budget=budget,
    )


def _find_section_outline(
    outline: OutlineStructure | None, section: SectionDraft
) -> SectionOutline | None:
    if outline is None:
        return None
    return next(
        (
            section_outline
            for section_outline in outline.sections
            if section_outline.title == section.title
        ),
        None,
    )


def _previous_summaries(state: AgentState, current_index: int) -> list[str]:
    summaries = list(state.get("section_summaries") or [])
    return [summary for summary in summaries[:current_index] if summary]


def _citation_sources(state: AgentState) -> list[object]:
    raw = state.get("citation_sources", [])
    return raw if isinstance(raw, list) else []


def _exemplar_context(state: AgentState) -> ExemplarContext | None:
    raw = state.get("exemplar_context")
    if raw is None:
        return None
    return coerce_exemplar_context(raw)


def _section_budget(
    section_id: str, section_outline: SectionOutline | None, state: AgentState
) -> SectionBudget | None:
    budget_state = _budget_state_from_state(state)
    if budget_state is not None:
        budget = budget_state.sections.get(section_id)
        if isinstance(budget, SectionBudget):
            return budget
        if isinstance(budget, dict):
            return SectionBudget(**budget)
    if section_outline is not None and section_outline.target_words is not None:
        return SectionBudget(section_id=section_id, target=section_outline.target_words)
    return None


def _budget_state_from_state(state: AgentState) -> BudgetState | None:
    raw = state.get("budget_state")
    if raw is None:
        return None
    if isinstance(raw, BudgetState):
        return raw
    if isinstance(raw, dict):
        return BudgetState(**raw)
    return None
