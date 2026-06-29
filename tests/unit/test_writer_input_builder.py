from seele_scholar_agent.budget import BudgetState, SectionBudget
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
