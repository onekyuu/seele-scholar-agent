"""Unit tests for ReviewerNode — RV-01 through RV-13."""

from datetime import datetime
from typing import cast
from unittest.mock import AsyncMock, patch

import pytest
from seele_scholar_agent.nodes.reviewer import ReviewerNode
from seele_scholar_agent.state import AgentState, PaperMetadata, SectionDraft


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


def _written_section(title="Introduction", content="Some content.", index=0, revision_count=0):
    return SectionDraft(
        section_id=f"section_{index}",
        title=title,
        description="Desc",
        order_index=index + 1,
        content=content,
        status="review",
        revision_count=revision_count,
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
    assert updated_section.revision_count == 1
    assert len(updated_section.review_comments) > 0


# ---------------------------------------------------------------------------
# RV-04: revision_count >= max_revisions, rejected → blocking issue, waiting_human
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_reviewer_max_revisions_blocks_approval(mock_llm, mock_prompts, base_state):
    sections = [_written_section("Introduction", index=0, revision_count=3)]
    state = _make_state_with_written_section(base_state, sections, index=0, max_revisions=3)

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

    assert result["status"] == "waiting_human"
    assert result["sections"][0].status == "review"
    assert result["quality_issues"][0].code == "MAX_REVISIONS_REACHED"
    assert result["quality_issues"][0].blocking is True


# ---------------------------------------------------------------------------
# RV-04b: revision_count >= max_revisions, rejected, NOT last section → blocks current section
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_reviewer_max_revisions_blocks_not_last_section(
    mock_llm, mock_prompts, base_state
):
    sections = [
        _written_section("Introduction", index=0, revision_count=2),
        _written_section("Related Work", index=1),
    ]
    state = _make_state_with_written_section(base_state, sections, index=0, max_revisions=2)

    review_result = {
        "approved": False,
        "score": 4,
        "issues": [{"type": "weak_argument", "description": "Weak.", "suggestion": "Add more."}],
        "summary": "Needs work.",
    }
    with patch(
        "seele_scholar_agent.nodes.reviewer.invoke_with_retry",
        new_callable=AsyncMock,
        return_value=review_result,
    ):
        node = ReviewerNode(llm=mock_llm, prompts=mock_prompts)
        result = await node.review(state)

    assert result["status"] == "waiting_human"
    assert result["sections"][0].status == "review"
    assert "current_section_index" not in result
    assert "sections_completed" not in result
    assert result["quality_issues"][0].code == "MAX_REVISIONS_REACHED"


# ---------------------------------------------------------------------------
# RV-04c: max_revisions is evaluated per section, not from global revision_count
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_reviewer_max_revisions_uses_section_revision_count(
    mock_llm, mock_prompts, base_state
):
    sections = [_written_section("Related Work", index=1, revision_count=0)]
    state = _make_state_with_written_section(
        base_state,
        sections,
        index=0,
        revision_count=10,
        max_revisions=2,
    )

    review_result = {
        "approved": False,
        "score": 4,
        "issues": [{"type": "weak_argument", "description": "Weak.", "suggestion": "Add more."}],
        "summary": "Needs work.",
    }
    with patch(
        "seele_scholar_agent.nodes.reviewer.invoke_with_retry",
        new_callable=AsyncMock,
        return_value=review_result,
    ):
        node = ReviewerNode(llm=mock_llm, prompts=mock_prompts)
        result = await node.review(state)

    assert result["status"] == "writing"
    assert result["sections"][0].status == "writing"
    assert result["sections"][0].revision_count == 1
    assert result["revision_count"] == 11


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
    assert result["sections"][0].revision_count == 1


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


# ---------------------------------------------------------------------------
# RV-10: Valid citations + LLM returns issues → citation_mismatch issues added
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_reviewer_citation_alignment_issues_added(mock_llm, mock_prompts, base_state):
    papers = [
        PaperMetadata(
            paper_id="p1",
            title="Attention Is All You Need",
            authors=["Vaswani"],
            abstract="Transformer architecture.",
            source="arxiv",
        ),
    ]
    sections = [
        _written_section("Introduction", content="The model in [1] is widely used.", index=0)
    ]
    state = cast(
        AgentState,
        {
            **_make_state_with_written_section(base_state, sections, index=0),
            "papers": papers,
        },
    )

    main_review = {"approved": True, "score": 8, "issues": [], "summary": "Good."}
    citation_result = {
        "issues": [
            {
                "description": "Citation [1] context mismatch.",
                "suggestion": "Verify claim.",
                "citation_number": 1,
            }
        ]
    }

    call_count = 0

    async def mock_invoke(chain, input_data, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return main_review
        return citation_result

    with patch(
        "seele_scholar_agent.nodes.reviewer.invoke_with_retry",
        side_effect=mock_invoke,
    ):
        node = ReviewerNode(llm=mock_llm, prompts=mock_prompts)
        result = await node.review(state)

    assert call_count == 2
    current_review = result.get("current_review") or {}
    all_issues = current_review.get("issues", [])
    citation_issues = [i for i in all_issues if i.get("type") == "citation_mismatch"]
    assert len(citation_issues) >= 1


# ---------------------------------------------------------------------------
# RV-11: Citation alignment LLM returns empty issues → review result unaffected
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_reviewer_citation_alignment_empty_issues_no_effect(
    mock_llm, mock_prompts, base_state
):
    papers = [
        PaperMetadata(
            paper_id="p1",
            title="BERT",
            authors=["Devlin"],
            abstract="Bidirectional encoder.",
            source="semantic_scholar",
        ),
    ]
    sections = [_written_section("Methods", content="We follow [1].", index=0)]
    state = cast(
        AgentState,
        {
            **_make_state_with_written_section(base_state, sections, index=0),
            "papers": papers,
        },
    )

    call_count = 0

    async def mock_invoke(chain, input_data, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return {"approved": True, "score": 9, "issues": [], "summary": "Fine."}
        return {"issues": []}

    with patch(
        "seele_scholar_agent.nodes.reviewer.invoke_with_retry",
        side_effect=mock_invoke,
    ):
        node = ReviewerNode(llm=mock_llm, prompts=mock_prompts)
        result = await node.review(state)

    assert result["sections"][0].status == "approved"
    assert result["status"] in ("completed", "writing")


# ---------------------------------------------------------------------------
# RV-12: papers=[] → citation alignment check not triggered
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_reviewer_no_citation_alignment_when_no_papers(mock_llm, mock_prompts, base_state):
    sections = [_written_section("Introduction", content="Some content.", index=0)]
    state = cast(
        AgentState,
        {
            **_make_state_with_written_section(base_state, sections, index=0),
            "papers": [],
        },
    )

    invoke_count = 0

    async def mock_invoke(chain, input_data, **kwargs):
        nonlocal invoke_count
        invoke_count += 1
        return {"approved": True, "score": 8, "issues": [], "summary": "OK."}

    with patch(
        "seele_scholar_agent.nodes.reviewer.invoke_with_retry",
        side_effect=mock_invoke,
    ):
        node = ReviewerNode(llm=mock_llm, prompts=mock_prompts)
        await node.review(state)

    assert invoke_count == 1


# ---------------------------------------------------------------------------
# RV-13: Citation alignment LLM fails → main review flow unaffected
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_reviewer_citation_alignment_failure_does_not_block_review(
    mock_llm, mock_prompts, base_state
):
    papers = [
        PaperMetadata(
            paper_id="p1",
            title="GPT-3",
            authors=["Brown"],
            abstract="Few-shot learner.",
            source="openalex",
        ),
    ]
    sections = [_written_section("Discussion", content="See [1] for details.", index=0)]
    state = cast(
        AgentState,
        {
            **_make_state_with_written_section(base_state, sections, index=0),
            "papers": papers,
        },
    )

    call_count = 0

    async def mock_invoke(chain, input_data, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return {"approved": True, "score": 8, "issues": [], "summary": "Good."}
        raise Exception("citation LLM failed")

    with patch(
        "seele_scholar_agent.nodes.reviewer.invoke_with_retry",
        side_effect=mock_invoke,
    ):
        node = ReviewerNode(llm=mock_llm, prompts=mock_prompts)
        result = await node.review(state)

    assert result["status"] in ("completed", "writing")
