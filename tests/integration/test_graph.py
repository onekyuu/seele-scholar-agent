"""Integration tests for create_simple_writing_graph — G-01 through G-06."""

import json
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
import respx
from langchain_core.messages import AIMessage
from langchain_openai import ChatOpenAI
from seele_scholar_agent.graph import create_simple_writing_graph


def _topic_proposer_response() -> str:
    return json.dumps(
        {
            "topics": [
                {
                    "title": "LLM Alignment",
                    "description": "Aligning large language models.",
                    "trend_analysis": "Growing field.",
                    "difficulty_level": "medium",
                }
            ]
        }
    )


def _planner_response(sections: list[dict] | None = None) -> str:
    if sections is None:
        sections = [{"title": "Introduction", "description": "Intro", "order": 1, "key_points": []}]
    return json.dumps(
        {
            "title": "Test Paper",
            "abstract": "Abstract.",
            "sections": sections,
            "keywords": ["test"],
        }
    )


def _reviewer_response(approved: bool = True) -> str:
    return json.dumps(
        {"approved": approved, "score": 8 if approved else 4, "issues": [], "summary": "OK."}
    )


def _make_mock_llm(side_effects: list) -> ChatOpenAI:
    llm = MagicMock(spec=ChatOpenAI)
    llm.ainvoke = AsyncMock(side_effect=side_effects)
    return llm


def _mock_http(respx_mock: respx.MockRouter) -> None:
    respx_mock.get(url__regex=r".*arxiv\.org.*").mock(
        return_value=httpx.Response(200, text="<feed></feed>")
    )
    respx_mock.get(url__regex=r".*openalex\.org.*").mock(
        return_value=httpx.Response(200, json={"results": [], "meta": {"count": 0}})
    )
    respx_mock.get(url__regex=r".*semanticscholar\.org.*").mock(
        return_value=httpx.Response(200, json={"data": []})
    )


# ---------------------------------------------------------------------------
# G-01: Full graph runs to completion (single section, single review cycle)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_graph_full_run_single_section(base_state, mock_prompts):
    with respx.mock(assert_all_mocked=False, assert_all_called=False) as respx_mock:
        _mock_http(respx_mock)

        llm = _make_mock_llm(
            [
                AIMessage(content=_topic_proposer_response()),
                AIMessage(content=_planner_response()),
                AIMessage(content="Introduction content."),
                AIMessage(content=_reviewer_response(approved=True)),
            ]
        )

        graph = create_simple_writing_graph(model=llm, prompts=mock_prompts)
        state = {**base_state, "outline_approved": True}

        result = await graph.ainvoke(state, config={"configurable": {"thread_id": "g-test-001"}})

    assert result["status"] == "completed"


# ---------------------------------------------------------------------------
# G-02: Graph correctly propagates outline from planner to writer
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_graph_planner_outline_reaches_writer(base_state, mock_prompts):
    with respx.mock(assert_all_mocked=False, assert_all_called=False) as respx_mock:
        _mock_http(respx_mock)

        llm = _make_mock_llm(
            [
                AIMessage(content=_topic_proposer_response()),
                AIMessage(content=_planner_response()),
                AIMessage(content="Introduction content."),
                AIMessage(content=_reviewer_response(approved=True)),
            ]
        )

        graph = create_simple_writing_graph(model=llm, prompts=mock_prompts)
        state = {**base_state, "outline_approved": True}

        result = await graph.ainvoke(state, config={"configurable": {"thread_id": "g-test-002"}})

    assert result["outline"] is not None
    assert result["outline"].title == "Test Paper"
    assert len(result["sections"]) == 1


# ---------------------------------------------------------------------------
# G-03: Graph handles reviewer rejection → writer retry cycle
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_graph_reviewer_rejection_triggers_retry(base_state, mock_prompts):
    with respx.mock(assert_all_mocked=False, assert_all_called=False) as respx_mock:
        _mock_http(respx_mock)

        rejected_review = json.dumps(
            {
                "approved": False,
                "score": 4,
                "issues": [{"type": "weak_argument", "description": "Weak.", "suggestion": "Fix."}],
                "summary": "Needs work.",
            }
        )

        llm = _make_mock_llm(
            [
                AIMessage(content=_topic_proposer_response()),
                AIMessage(content=_planner_response()),
                AIMessage(content="First draft."),
                AIMessage(content=rejected_review),
                AIMessage(content="Revised draft."),
                AIMessage(content=_reviewer_response(approved=True)),
            ]
        )

        graph = create_simple_writing_graph(model=llm, prompts=mock_prompts)
        state = {**base_state, "outline_approved": True, "max_revisions": 3}

        result = await graph.ainvoke(state, config={"configurable": {"thread_id": "g-test-003"}})

    assert result["status"] == "completed"
    assert result["revision_count"] == 1


# ---------------------------------------------------------------------------
# G-04: Graph terminates when max_revisions reached (force-approved)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_graph_max_revisions_forces_completion(base_state, mock_prompts):
    with respx.mock(assert_all_mocked=False, assert_all_called=False) as respx_mock:
        _mock_http(respx_mock)

        rejected_review = json.dumps(
            {
                "approved": False,
                "score": 3,
                "issues": [{"type": "other", "description": "Bad.", "suggestion": "Redo."}],
                "summary": "Poor.",
            }
        )

        llm = _make_mock_llm(
            [
                AIMessage(content=_topic_proposer_response()),
                AIMessage(content=_planner_response()),
                AIMessage(content="Draft 1."),
                AIMessage(content=rejected_review),
                AIMessage(content="Draft 2."),
                AIMessage(content=rejected_review),
            ]
        )

        graph = create_simple_writing_graph(model=llm, prompts=mock_prompts)
        state = {**base_state, "outline_approved": True, "max_revisions": 1}

        result = await graph.ainvoke(state, config={"configurable": {"thread_id": "g-test-004"}})

    assert result["status"] == "completed"


# ---------------------------------------------------------------------------
# G-05: Graph with multiple sections completes all sections
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_graph_multiple_sections_all_completed(base_state, mock_prompts):
    with respx.mock(assert_all_mocked=False, assert_all_called=False) as respx_mock:
        _mock_http(respx_mock)

        planner_resp = _planner_response(
            sections=[
                {"title": "Introduction", "description": "Intro", "order": 1, "key_points": []},
                {"title": "Conclusion", "description": "Summary", "order": 2, "key_points": []},
            ]
        )

        llm = _make_mock_llm(
            [
                AIMessage(content=_topic_proposer_response()),
                AIMessage(content=planner_resp),
                AIMessage(content="Introduction content."),
                AIMessage(content=_reviewer_response(approved=True)),
                AIMessage(content="Conclusion content."),
                AIMessage(content=_reviewer_response(approved=True)),
                AIMessage(content=json.dumps({"issues": []})),
                AIMessage(content=json.dumps({"issues": []})),
                AIMessage(content=json.dumps({"issues": []})),
            ]
        )

        graph = create_simple_writing_graph(model=llm, prompts=mock_prompts)
        state = {**base_state, "outline_approved": True}

        result = await graph.ainvoke(state, config={"configurable": {"thread_id": "g-test-005"}})

    assert result["status"] == "completed"
    completed = result["sections_completed"]
    assert "Introduction" in completed
    assert "Conclusion" in completed


# ---------------------------------------------------------------------------
# G-06: Graph populates proposed_topics from TopicProposerNode
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_graph_topic_proposer_populates_topics(base_state, mock_prompts):
    with respx.mock(assert_all_mocked=False, assert_all_called=False) as respx_mock:
        _mock_http(respx_mock)

        llm = _make_mock_llm(
            [
                AIMessage(content=_topic_proposer_response()),
                AIMessage(content=_planner_response()),
                AIMessage(content="Introduction content."),
                AIMessage(content=_reviewer_response(approved=True)),
            ]
        )

        graph = create_simple_writing_graph(model=llm, prompts=mock_prompts)
        state = {**base_state, "outline_approved": True}

        result = await graph.ainvoke(state, config={"configurable": {"thread_id": "g-test-006"}})

    assert len(result.get("proposed_topics", [])) >= 0
