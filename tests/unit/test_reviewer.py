"""Unit tests for ReviewerNode — RV-01 through RV-09."""

from datetime import datetime
from typing import cast
from unittest.mock import AsyncMock, patch

import pytest

from seele_scholar_agent.nodes.reviewer import ReviewerNode
from seele_scholar_agent.state import AgentState, SectionDraft


def _make_state_with_written_section(
    base_state, sections, index=0, revision_count=0, max_revisions=3
):
    return cast(
        AgentState,
        {
            **base_state,
            "sections": sections,
            "current_section_index": index,
            "revision_count": revision_count,
            "max_revisions": max_revisions,
            "status": "reviewing",
        },
    )


def _written_section(title="Introduction", content="Some content.", index=0):
    return SectionDraft(
        section_id=f"section_{index}",
        title=title,
        description="Desc",
        order_index=index + 1,
        content=content,
        status="review",
    )


# ---------------------------------------------------------------------------
# RV-01: approved=True, not last section → status="writing", index+1
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_reviewer_approved_not_last_section(mock_llm, mock_prompts, base_state):
    sections = [
        _written_section("Introduction", index=0),
        _written_section("Related Work", index=1),
    ]
    state = _make_state_with_written_section(base_state, sections, index=0)

    review_result = {"approved": True, "score": 8, "issues": [], "summary": "Looks good."}
    with patch(
        "seele_scholar_agent.nodes.reviewer.invoke_with_retry",
        new_callable=AsyncMock,
        return_value=review_result,
    ):
        node = ReviewerNode(llm=mock_llm, prompts=mock_prompts)
        result = await node.review(state)

    assert result["sections"][0].status == "approved"
    assert result["current_section_index"] == 1
    assert result["status"] == "writing"


# ---------------------------------------------------------------------------
# RV-02: approved=True, last section → status="completed"
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_reviewer_approved_last_section(mock_llm, mock_prompts, base_state):
    sections = [_written_section("Introduction", index=0)]
    state = _make_state_with_written_section(base_state, sections, index=0)

    review_result = {"approved": True, "score": 9, "issues": [], "summary": "Great."}
    with patch(
        "seele_scholar_agent.nodes.reviewer.invoke_with_retry",
        new_callable=AsyncMock,
        return_value=review_result,
    ):
        node = ReviewerNode(llm=mock_llm, prompts=mock_prompts)
        result = await node.review(state)

    assert result["status"] == "completed"


# ---------------------------------------------------------------------------
# RV-03: approved=False, 2 issues → review_comments appended, revision_count+1, status="writing"
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_reviewer_rejected_appends_comments_and_increments_revision(
    mock_llm, mock_prompts, base_state
):
    sections = [_written_section("Introduction", index=0)]
    state = _make_state_with_written_section(base_state, sections, index=0, revision_count=0)

    review_result = {
        "approved": False,
        "score": 4,
        "issues": [
            {"type": "factual_error", "description": "Incorrect claim.", "suggestion": "Fix it."},
            {"type": "weak_argument", "description": "Needs evidence.", "suggestion": "Add refs."},
        ],
        "summary": "Needs improvement.",
    }
    with patch(
        "seele_scholar_agent.nodes.reviewer.invoke_with_retry",
        new_callable=AsyncMock,
        return_value=review_result,
    ):
        node = ReviewerNode(llm=mock_llm, prompts=mock_prompts)
        result = await node.review(state)

    assert result["revision_count"] == 1
    assert result["status"] == "writing"
    updated_section = result["sections"][0]
    assert len(updated_section.review_comments) > 0


# ---------------------------------------------------------------------------
# RV-04: revision_count >= max_revisions, rejected → force approved, status="completed"
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_reviewer_max_revisions_forces_approval(mock_llm, mock_prompts, base_state):
    sections = [_written_section("Introduction", index=0)]
    state = _make_state_with_written_section(
        base_state, sections, index=0, revision_count=3, max_revisions=3
    )

    review_result = {
        "approved": False,
        "score": 3,
        "issues": [{"type": "other", "description": "Bad.", "suggestion": "Redo."}],
        "summary": "Poor quality.",
    }
    with patch(
        "seele_scholar_agent.nodes.reviewer.invoke_with_retry",
        new_callable=AsyncMock,
        return_value=review_result,
    ):
        node = ReviewerNode(llm=mock_llm, prompts=mock_prompts)
        result = await node.review(state)

    assert result["status"] == "completed"
    assert result["sections"][0].status == "approved"


# ---------------------------------------------------------------------------
# RV-05: review_comments include round marker and opinion prefix (zh)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_reviewer_comments_contain_round_marker_zh(mock_llm, mock_prompts, base_state):
    sections = [_written_section("Introduction", index=0)]
    state = cast(
        AgentState,
        {
            **_make_state_with_written_section(base_state, sections, index=0, revision_count=0),
            "language": "zh",
        },
    )

    review_result = {
        "approved": False,
        "score": 5,
        "issues": [],
        "summary": "Need more depth.",
    }
    with patch(
        "seele_scholar_agent.nodes.reviewer.invoke_with_retry",
        new_callable=AsyncMock,
        return_value=review_result,
    ):
        node = ReviewerNode(llm=mock_llm, prompts=mock_prompts)
        result = await node.review(state)

    comments = result["sections"][0].review_comments
    combined = " ".join(comments)
    assert "轮审稿" in combined or "审稿" in combined
    assert "意见" in combined


# ---------------------------------------------------------------------------
# RV-06: LLM raises exception → fallback ReviewResult(approved=False, score=5)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_reviewer_llm_exception_fallback(mock_llm, mock_prompts, base_state):
    sections = [_written_section("Introduction", index=0)]
    state = _make_state_with_written_section(base_state, sections, index=0, revision_count=0)

    with patch(
        "seele_scholar_agent.nodes.reviewer.invoke_with_retry",
        side_effect=Exception("LLM timeout"),
    ):
        node = ReviewerNode(llm=mock_llm, prompts=mock_prompts)
        result = await node.review(state)

    assert result["status"] == "writing"
    assert result["revision_count"] == 1


# ---------------------------------------------------------------------------
# RV-07: issues=[] → no crash, normal processing
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_reviewer_empty_issues_no_crash(mock_llm, mock_prompts, base_state):
    sections = [_written_section("Introduction", index=0)]
    state = _make_state_with_written_section(base_state, sections, index=0)

    review_result = {"approved": False, "score": 6, "issues": [], "summary": "Needs work."}
    with patch(
        "seele_scholar_agent.nodes.reviewer.invoke_with_retry",
        new_callable=AsyncMock,
        return_value=review_result,
    ):
        node = ReviewerNode(llm=mock_llm, prompts=mock_prompts)
        result = await node.review(state)

    assert result["status"] == "writing"


# ---------------------------------------------------------------------------
# RV-08: ReviewIssue type field correctly assigned
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_reviewer_issue_type_correctly_assigned(mock_llm, mock_prompts, base_state):
    sections = [_written_section("Introduction", index=0)]
    state = _make_state_with_written_section(base_state, sections, index=0)

    review_result = {
        "approved": False,
        "score": 4,
        "issues": [
            {
                "type": "factual_error",
                "description": "Wrong claim.",
                "suggestion": "Fix it.",
            }
        ],
        "summary": "Fix factual error.",
    }
    with patch(
        "seele_scholar_agent.nodes.reviewer.invoke_with_retry",
        new_callable=AsyncMock,
        return_value=review_result,
    ):
        node = ReviewerNode(llm=mock_llm, prompts=mock_prompts)
        result = await node.review(state)

    current_review = result.get("current_review") or {}
    issues = current_review.get("issues", [])
    assert len(issues) == 1
    assert issues[0]["type"] == "factual_error"


# ---------------------------------------------------------------------------
# RV-09: review_history record contains section, score, approved, timestamp
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_reviewer_history_record_structure(mock_llm, mock_prompts, base_state):
    sections = [_written_section("Introduction", index=0)]
    state = _make_state_with_written_section(base_state, sections, index=0)

    review_result = {"approved": True, "score": 8, "issues": [], "summary": "Good."}
    with patch(
        "seele_scholar_agent.nodes.reviewer.invoke_with_retry",
        new_callable=AsyncMock,
        return_value=review_result,
    ):
        node = ReviewerNode(llm=mock_llm, prompts=mock_prompts)
        result = await node.review(state)

    history = result["review_history"]
    assert len(history) == 1
    record = history[0]
    assert "section" in record
    assert "score" in record
    assert "approved" in record
    assert "timestamp" in record
    assert record["section"] == "Introduction"
    assert record["score"] == 8
    assert record["approved"] is True
    datetime.fromisoformat(record["timestamp"])
