"""Integration tests: node-to-node state flow — I-01 through I-09."""

from typing import cast
from unittest.mock import AsyncMock, patch

import pytest
from langchain_core.messages import AIMessage

from seele_scholar_agent.nodes.planner import PlannerNode
from seele_scholar_agent.nodes.reviewer import ReviewerNode
from seele_scholar_agent.nodes.writer import WriterNode
from seele_scholar_agent.nodes.finalizer import FinalizerNode
from seele_scholar_agent.nodes.consistency_checker import ConsistencyCheckerNode
from seele_scholar_agent.nodes.reference_generator import ReferenceGeneratorNode
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


# ---------------------------------------------------------------------------
# I-07: writer → reviewer → finalizer — pending abstract section gets finalized
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_writer_reviewer_finalizer_chain(mock_llm, mock_prompts, state_with_outline):
    sections = list(state_with_outline["sections"])
    sections.append(
        SectionDraft(
            section_id="s_abstract",
            title="Abstract",
            description="Summary",
            order_index=0,
            content="",
            status="pending",
        )
    )
    state = cast(AgentState, {**state_with_outline, "sections": sections})

    with patch(
        "seele_scholar_agent.nodes.writer.invoke_with_retry",
        new_callable=AsyncMock,
        return_value=AIMessage(content="Introduction content."),
    ):
        writer = WriterNode(llm=mock_llm, prompts=mock_prompts)
        write_result = await writer.write(state)

    reviewer_state = cast(AgentState, {**state, **write_result})

    with patch(
        "seele_scholar_agent.nodes.reviewer.invoke_with_retry",
        new_callable=AsyncMock,
        return_value={"approved": True, "score": 8, "issues": [], "summary": "Good."},
    ):
        reviewer = ReviewerNode(llm=mock_llm, prompts=mock_prompts)
        review_result = await reviewer.review(reviewer_state)

    finalizer_state = cast(AgentState, {**reviewer_state, **review_result})

    with patch(
        "seele_scholar_agent.nodes.finalizer.invoke_with_retry",
        new_callable=AsyncMock,
        return_value=AIMessage(content="Generated abstract."),
    ):
        finalizer = FinalizerNode(llm=mock_llm, prompts=mock_prompts)
        finalize_result = await finalizer.finalize(finalizer_state)

    assert finalize_result["status"] == "completed"
    abstract = next((s for s in finalize_result["sections"] if s.title == "Abstract"), None)
    assert abstract is not None
    assert abstract.status == "auto_generated"


# ---------------------------------------------------------------------------
# I-08: finalizer → consistency_checker — issues written to state
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_finalizer_output_feeds_consistency_checker(mock_llm, mock_prompts, base_state):
    sections = [
        SectionDraft(
            section_id="s0",
            title="Introduction",
            description="",
            order_index=1,
            content="Intro content.",
            status="approved",
        ),
        SectionDraft(
            section_id="s1",
            title="Conclusion",
            description="",
            order_index=2,
            content="",
            status="pending",
        ),
    ]
    state = cast(AgentState, {**base_state, "sections": sections})

    with patch(
        "seele_scholar_agent.nodes.finalizer.invoke_with_retry",
        new_callable=AsyncMock,
        return_value=AIMessage(content="Generated conclusion."),
    ):
        finalizer = FinalizerNode(llm=mock_llm, prompts=mock_prompts)
        finalize_result = await finalizer.finalize(state)

    checker_state = cast(AgentState, {**state, **finalize_result})

    with patch(
        "seele_scholar_agent.nodes.consistency_checker.invoke_with_retry",
        new_callable=AsyncMock,
        return_value={
            "issues": [
                {
                    "issue_type": "logic",
                    "description": "Minor inconsistency.",
                    "sections_involved": ["Introduction", "Conclusion"],
                    "suggestion": "Align claims.",
                }
            ]
        },
    ):
        checker = ConsistencyCheckerNode(llm=mock_llm, prompts=mock_prompts)
        check_result = await checker.check(checker_state)

    assert check_result["consistency_checked"] is True
    assert len(check_result["consistency_issues"]) == 1
    assert check_result["consistency_issues"][0].issue_type == "logic"


# ---------------------------------------------------------------------------
# I-09: consistency_checker → reference_generator — references written to state
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_consistency_checker_output_feeds_reference_generator(
    mock_llm, mock_prompts, base_state, sample_papers
):
    sections = [
        SectionDraft(
            section_id="s0",
            title="Introduction",
            description="",
            order_index=1,
            content="See [1] for background.",
            status="approved",
        ),
        SectionDraft(
            section_id="s1",
            title="Methods",
            description="",
            order_index=2,
            content="We follow [2].",
            status="approved",
        ),
    ]
    state = cast(AgentState, {**base_state, "papers": sample_papers, "sections": sections})

    with patch(
        "seele_scholar_agent.nodes.consistency_checker.invoke_with_retry",
        new_callable=AsyncMock,
        return_value={"issues": []},
    ):
        checker = ConsistencyCheckerNode(llm=mock_llm, prompts=mock_prompts)
        check_result = await checker.check(state)

    ref_gen_state = cast(AgentState, {**state, **check_result})

    ref_gen = ReferenceGeneratorNode()
    ref_result = await ref_gen.generate(ref_gen_state)

    assert ref_result["status"] == "completed"
    refs = ref_result["references"]
    numbers = {r.number for r in refs}
    assert 1 in numbers
    assert 2 in numbers
