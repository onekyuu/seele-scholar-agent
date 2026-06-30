from seele_scholar_agent.budget import BudgetState, SectionBudget
from seele_scholar_agent.draft import (
    DraftIntegrationNode,
    DraftSegment,
    ExistingContentRef,
    PreservePolicy,
)
from seele_scholar_agent.exemplar import ExemplarContext
from seele_scholar_agent.writing import WriterInputBuilder


def test_writer_input_builder_injects_outline_and_budget(state_with_outline):
    budget = SectionBudget(section_id="section_0", target=120, hard_limit=150, unit="chars")
    state = {
        **state_with_outline,
        "budget_state": BudgetState(sections={"section_0": budget}),
        "section_summaries": ["[Previous]\nSummary."],
    }

    writer_input = WriterInputBuilder().build(
        state,
        current_index=0,
        evidence_packets=[],
        style_context="Style guidance.",
    )

    assert writer_input.topic == state_with_outline["topic"]
    assert writer_input.outline_context.title == state_with_outline["outline"].title
    assert writer_input.current_section.title == "Introduction"
    assert writer_input.current_section.budget == budget
    assert writer_input.style_context == "Style guidance."
    assert writer_input.exemplar_context is None


def test_writer_input_builder_uses_previous_section_summaries(state_with_outline):
    state = {
        **state_with_outline,
        "section_summaries": ["[Introduction]\nIntro summary.", "[Related]\nRelated summary."],
        "current_section_index": 2,
    }

    writer_input = WriterInputBuilder().build(
        state,
        current_index=2,
        evidence_packets=[],
        style_context="",
    )

    assert writer_input.previous_section_summaries == [
        "[Introduction]\nIntro summary.",
        "[Related]\nRelated summary.",
    ]


def test_writer_input_builder_injects_exemplar_context(state_with_outline):
    exemplar_context = ExemplarContext(
        outline_patterns=["Move from gap to contribution."],
        style_notes=["Prefer compact transitions."],
    )
    state = {**state_with_outline, "exemplar_context": exemplar_context}

    writer_input = WriterInputBuilder().build(
        state,
        current_index=0,
        evidence_packets=[],
        style_context="",
    )

    assert writer_input.exemplar_context == exemplar_context


def test_writer_input_builder_injects_draft_context(state_with_outline):
    existing_content = ExistingContentRef(
        draft_id="draft-1",
        version_id="v1",
        segments=[
            DraftSegment(
                segment_id="seg-intro",
                detected_heading="Introduction",
                text="Draft introduction content to preserve and expand.",
                order=1,
            )
        ],
        preserve_policy=PreservePolicy(protected_segment_ids=["seg-intro"]),
        user_intent="expand",
    )
    draft_state = DraftIntegrationNode().integrate(
        {**state_with_outline, "existing_content": existing_content}
    )["draft_integration"]
    state = {**state_with_outline, "draft_integration": draft_state}

    writer_input = WriterInputBuilder().build(
        state,
        current_index=0,
        evidence_packets=[],
        style_context="",
    )

    assert writer_input.draft_context is not None
    assert writer_input.draft_context.mapped_segments[0].segment_id == "seg-intro"
    assert writer_input.draft_context.preserve_policy.protected_segment_ids == ["seg-intro"]
