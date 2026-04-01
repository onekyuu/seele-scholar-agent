"""Integration tests: node-to-node state flow — I-01 through I-06."""

from typing import cast
from unittest.mock import AsyncMock, patch

import pytest
from langchain_core.messages import AIMessage

from seele_scholar_agent.nodes.planner import PlannerNode
from seele_scholar_agent.nodes.reviewer import ReviewerNode
from seele_scholar_agent.nodes.writer import WriterNode
from seele_scholar_agent.state import AgentState, SectionDraft


def _make_planner_result():
    return {
        "title": "LLM Survey",
        "abstract": "A survey.",
        "sections": [
            {"title": "Introduction", "description": "Intro", "order": 1, "key_points": []},
            {"title": "Conclusion", "description": "Summary", "order": 2, "key_points": []},
        ],
        "keywords": ["LLM"],
    }


# ---------------------------------------------------------------------------
# I-01: PlannerNode output feeds WriterNode correctly
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_planner_output_feeds_writer(mock_llm, mock_prompts, state_with_papers):
    with patch(
        "seele_scholar_agent.nodes.planner.invoke_with_retry",
        new_callable=AsyncMock,
        return_value=_make_planner_result(),
    ):
        planner = PlannerNode(llm=mock_llm, prompts=mock_prompts)
        plan_result = await planner.plan(state_with_papers)

    writer_state = cast(
        AgentState,
        {
            **state_with_papers,
            **plan_result,
        },
    )

    with patch(
        "seele_scholar_agent.nodes.writer.invoke_with_retry",
        new_callable=AsyncMock,
        return_value=AIMessage(content="Introduction content."),
    ):
        writer = WriterNode(llm=mock_llm, prompts=mock_prompts)
        write_result = await writer.write(writer_state)

    assert write_result["sections"][0].content == "Introduction content."
    assert write_result["status"] == "reviewing"


# ---------------------------------------------------------------------------
# I-02: WriterNode output feeds ReviewerNode correctly
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_writer_output_feeds_reviewer(mock_llm, mock_prompts, state_with_outline):
    with patch(
        "seele_scholar_agent.nodes.writer.invoke_with_retry",
        new_callable=AsyncMock,
        return_value=AIMessage(content="Section content here."),
    ):
        writer = WriterNode(llm=mock_llm, prompts=mock_prompts)
        write_result = await writer.write(state_with_outline)

    reviewer_state = cast(AgentState, {**state_with_outline, **write_result})

    review_data = {"approved": True, "score": 8, "issues": [], "summary": "Good."}
    with patch(
        "seele_scholar_agent.nodes.reviewer.invoke_with_retry",
        new_callable=AsyncMock,
        return_value=review_data,
    ):
        reviewer = ReviewerNode(llm=mock_llm, prompts=mock_prompts)
        review_result = await reviewer.review(reviewer_state)

    assert review_result["sections"][0].status == "approved"


# ---------------------------------------------------------------------------
# I-03: Reviewer rejection → review_comments passed back to writer
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_reviewer_rejection_comments_reach_writer(mock_llm, mock_prompts, state_with_outline):
    with patch(
        "seele_scholar_agent.nodes.writer.invoke_with_retry",
        new_callable=AsyncMock,
        return_value=AIMessage(content="Initial draft."),
    ):
        writer = WriterNode(llm=mock_llm, prompts=mock_prompts)
        write_result = await writer.write(state_with_outline)

    reviewer_state = cast(AgentState, {**state_with_outline, **write_result})

    review_data = {
        "approved": False,
        "score": 4,
        "issues": [{"type": "weak_argument", "description": "Weak.", "suggestion": "Strengthen."}],
        "summary": "Needs improvement.",
    }
    with patch(
        "seele_scholar_agent.nodes.reviewer.invoke_with_retry",
        new_callable=AsyncMock,
        return_value=review_data,
    ):
        reviewer = ReviewerNode(llm=mock_llm, prompts=mock_prompts)
        review_result = await reviewer.review(reviewer_state)

    assert review_result["status"] == "writing"
    rejected_section = review_result["sections"][0]
    assert len(rejected_section.review_comments) > 0

    writer_state_2 = cast(AgentState, {**reviewer_state, **review_result})
    captured_invoke_args: list = []

    async def capture(*args, **kwargs):
        captured_invoke_args.append(args[1] if len(args) > 1 else kwargs.get("input_data", {}))
        return AIMessage(content="Revised draft.")

    with patch("seele_scholar_agent.nodes.writer.invoke_with_retry", side_effect=capture):
        writer2 = WriterNode(llm=mock_llm, prompts=mock_prompts)
        revision_result = await writer2.write(writer_state_2)

    assert revision_result["status"] == "reviewing"
    if captured_invoke_args:
        review_comments_arg = captured_invoke_args[0].get("review_comments", "")
        assert review_comments_arg != "无"


# ---------------------------------------------------------------------------
# I-04: Full planner → writer → reviewer cycle for single section
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_full_single_section_cycle(mock_llm, mock_prompts, state_with_papers):
    with patch(
        "seele_scholar_agent.nodes.planner.invoke_with_retry",
        new_callable=AsyncMock,
        return_value={
            "title": "Test Paper",
            "abstract": "Abstract.",
            "sections": [
                {"title": "Introduction", "description": "Intro", "order": 1, "key_points": []}
            ],
            "keywords": ["test"],
        },
    ):
        planner = PlannerNode(llm=mock_llm, prompts=mock_prompts)
        plan_result = await planner.plan(state_with_papers)

    assert plan_result["status"] == "waiting_human"

    writer_state = cast(AgentState, {**state_with_papers, **plan_result, "status": "writing"})

    with patch(
        "seele_scholar_agent.nodes.writer.invoke_with_retry",
        new_callable=AsyncMock,
        return_value=AIMessage(content="Introduction."),
    ):
        writer = WriterNode(llm=mock_llm, prompts=mock_prompts)
        write_result = await writer.write(writer_state)

    assert write_result["status"] == "reviewing"

    reviewer_state = cast(AgentState, {**writer_state, **write_result})

    with patch(
        "seele_scholar_agent.nodes.reviewer.invoke_with_retry",
        new_callable=AsyncMock,
        return_value={"approved": True, "score": 9, "issues": [], "summary": "Excellent."},
    ):
        reviewer = ReviewerNode(llm=mock_llm, prompts=mock_prompts)
        review_result = await reviewer.review(reviewer_state)

    assert review_result["status"] == "completed"


# ---------------------------------------------------------------------------
# I-05: status field propagates correctly through state merges
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_status_propagates_through_nodes(mock_llm, mock_prompts, state_with_papers):
    with patch(
        "seele_scholar_agent.nodes.planner.invoke_with_retry",
        new_callable=AsyncMock,
        return_value=_make_planner_result(),
    ):
        planner = PlannerNode(llm=mock_llm, prompts=mock_prompts)
        plan_result = await planner.plan(state_with_papers)

    assert plan_result["status"] == "waiting_human"

    combined = {**state_with_papers, **plan_result, "status": "writing"}
    assert combined["status"] == "writing"


# ---------------------------------------------------------------------------
# I-06: sections list maintains order after planner → writer pipeline
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_sections_order_maintained_across_nodes(mock_llm, mock_prompts, state_with_papers):
    llm_sections = [
        {"title": "Conclusion", "description": "Summary", "order": 3, "key_points": []},
        {"title": "Introduction", "description": "Intro", "order": 1, "key_points": []},
        {"title": "Methods", "description": "Method", "order": 2, "key_points": []},
    ]
    with patch(
        "seele_scholar_agent.nodes.planner.invoke_with_retry",
        new_callable=AsyncMock,
        return_value={
            "title": "Paper",
            "abstract": "Abs.",
            "sections": llm_sections,
            "keywords": [],
        },
    ):
        planner = PlannerNode(llm=mock_llm, prompts=mock_prompts)
        plan_result = await planner.plan(state_with_papers)

    section_titles = [s.title for s in plan_result["sections"]]
    assert section_titles[0] == "Introduction"
    assert section_titles[1] == "Methods"
    assert section_titles[2] == "Conclusion"
