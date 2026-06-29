"""Unit tests for ReviewerNode — RV-01 through RV-13."""

from datetime import datetime
from typing import cast
from unittest.mock import AsyncMock, patch

import pytest
from seele_scholar_agent.nodes.reviewer import ReviewerNode
from seele_scholar_agent.policy import WritingPolicy
from seele_scholar_agent.state import (
    AgentState,
    ClaimEvidenceBinding,
    PaperMetadata,
    SectionDraft,
)


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
# RV-04: revision_count >= max_revisions, rejected → accept best with report
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_reviewer_max_revisions_accepts_best_with_report(
    mock_llm, mock_prompts, base_state
):
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

    assert result["status"] == "completed"
    assert result["sections"][0].status == "accepted_with_issues"
    assert result["quality_issues"][0].code == "MAX_REVISIONS_REACHED"
    assert result["quality_issues"][0].blocking is False
    assert result["quality_issues"][0].details["accepted_with_issues"] is True
    assert result["review_decision"]["action"] == "accept_best"


# ---------------------------------------------------------------------------
# RV-04b: revision_count >= max_revisions, rejected, NOT last section → continues next section
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_reviewer_max_revisions_accepts_best_not_last_section(
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

    assert result["status"] == "writing"
    assert result["sections"][0].status == "accepted_with_issues"
    assert result["current_section_index"] == 1
    assert result["sections_completed"] == ["Introduction"]
    assert result["quality_issues"][0].code == "MAX_REVISIONS_REACHED"


@pytest.mark.asyncio
async def test_reviewer_max_revisions_can_block_with_policy(
    mock_llm, mock_prompts, base_state
):
    sections = [_written_section("Introduction", index=0, revision_count=1)]
    state = _make_state_with_written_section(base_state, sections, index=0, max_revisions=1)

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
        node = ReviewerNode(
            llm=mock_llm,
            prompts=mock_prompts,
            writing_policy=WritingPolicy(on_max_revisions="block"),
        )
        result = await node.review(state)

    assert result["status"] == "waiting_human"
    assert result["sections"][0].status == "review"
    assert result["quality_issues"][0].blocking is True
    assert result["review_decision"]["action"] == "human_required"


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
            "claim_evidence_bindings": [
                ClaimEvidenceBinding(
                    section_id=sections[0].section_id,
                    claim_text="The model in [1] is widely used.",
                    citation_number=1,
                    chunk_id="c1",
                    support_score=0.8,
                    verdict="supported",
                )
            ],
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
            "claim_evidence_bindings": [
                ClaimEvidenceBinding(
                    section_id=sections[0].section_id,
                    claim_text="We follow [1].",
                    citation_number=1,
                    chunk_id="c1",
                    support_score=0.8,
                    verdict="supported",
                )
            ],
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


@pytest.mark.asyncio
async def test_reviewer_rejects_unsupported_claim_source_binding(
    mock_llm, mock_prompts, base_state
):
    papers = [
        PaperMetadata(
            paper_id="p1",
            title="Evidence Paper",
            authors=["Author"],
            abstract="Evidence abstract.",
            source="openalex",
        )
    ]
    section = _written_section("Discussion", content="A strong claim is made [1].", index=0)
    binding = ClaimEvidenceBinding(
        section_id=section.section_id,
        claim_text="A strong claim is made [1].",
        citation_number=1,
        support_score=0.0,
        verdict="unsupported",
    )
    state = cast(
        AgentState,
        {
            **_make_state_with_written_section(base_state, [section], index=0),
            "papers": papers,
            "claim_evidence_bindings": [binding],
        },
    )

    with patch(
        "seele_scholar_agent.nodes.reviewer.invoke_with_retry",
        new_callable=AsyncMock,
        return_value={"approved": True, "score": 8, "issues": [], "summary": "Good."},
    ):
        node = ReviewerNode(llm=mock_llm, prompts=mock_prompts)
        result = await node.review(state)

    assert result["status"] == "writing"
    assert result["sections"][0].status == "writing"
    current_review = result["current_review"]
    assert current_review["approved"] is False
    assert current_review["issues"][0]["type"] == "citation_mismatch"


@pytest.mark.asyncio
async def test_reviewer_accepts_supported_claim_source_binding(
    mock_llm, mock_prompts, base_state
):
    papers = [
        PaperMetadata(
            paper_id="p1",
            title="Evidence Paper",
            authors=["Author"],
            abstract="Evidence abstract.",
            source="openalex",
        )
    ]
    section = _written_section("Discussion", content="A supported claim is made [1].", index=0)
    binding = ClaimEvidenceBinding(
        section_id=section.section_id,
        claim_text="A supported claim is made [1].",
        citation_number=1,
        chunk_id="c1",
        support_score=0.8,
        verdict="supported",
    )
    state = cast(
        AgentState,
        {
            **_make_state_with_written_section(base_state, [section], index=0),
            "papers": papers,
            "claim_evidence_bindings": [binding],
        },
    )

    with patch(
        "seele_scholar_agent.nodes.reviewer.invoke_with_retry",
        new_callable=AsyncMock,
        return_value={"approved": True, "score": 8, "issues": [], "summary": "Good."},
    ):
        node = ReviewerNode(llm=mock_llm, prompts=mock_prompts)
        result = await node.review(state)

    assert result["status"] == "completed"
    assert result["sections"][0].status == "approved"


@pytest.mark.asyncio
async def test_reviewer_rejects_uncited_factual_claim_with_quality_issue(
    mock_llm, mock_prompts, base_state
):
    section = _written_section(
        "Results",
        content="The model improves accuracy by 12%. This section summarizes implications.",
        index=0,
    )
    state = _make_state_with_written_section(base_state, [section], index=0)

    with patch(
        "seele_scholar_agent.nodes.reviewer.invoke_with_retry",
        new_callable=AsyncMock,
        return_value={"approved": True, "score": 8, "issues": [], "summary": "Good."},
    ):
        node = ReviewerNode(llm=mock_llm, prompts=mock_prompts)
        result = await node.review(state)

    assert result["status"] == "writing"
    assert result["current_review"]["approved"] is False
    assert result["current_review"]["issues"][0]["type"] == "missing_citation"
    assert result["quality_issues"][0].code == "UNSUPPORTED_CLAIM"
    assert result["quality_issues"][0].details["citation_numbers"] == []


@pytest.mark.asyncio
async def test_reviewer_rejects_cited_claim_without_evidence_packet_quality_issue(
    mock_llm, mock_prompts, base_state
):
    papers = [
        PaperMetadata(
            paper_id="p1",
            title="Evidence Paper",
            authors=["Author"],
            abstract="Evidence abstract.",
            source="openalex",
        )
    ]
    section = _written_section(
        "Results",
        content="Prior work shows attention improves sequence modeling [1].",
        index=0,
    )
    state = cast(
        AgentState,
        {
            **_make_state_with_written_section(base_state, [section], index=0),
            "papers": papers,
            "claim_evidence_bindings": [],
        },
    )

    with patch(
        "seele_scholar_agent.nodes.reviewer.invoke_with_retry",
        new_callable=AsyncMock,
        return_value={"approved": True, "score": 8, "issues": [], "summary": "Good."},
    ):
        node = ReviewerNode(llm=mock_llm, prompts=mock_prompts)
        result = await node.review(state)

    assert result["status"] == "writing"
    assert result["current_review"]["approved"] is False
    assert result["current_review"]["issues"][0]["type"] == "citation_mismatch"
    assert result["quality_issues"][0].code == "CLAIM_MISSING_EVIDENCE_PACKET"


@pytest.mark.asyncio
async def test_reviewer_rejects_stale_binding_for_different_claim(
    mock_llm, mock_prompts, base_state
):
    section = _written_section(
        "Results",
        content="Prior work shows attention improves sequence modeling [1].",
        index=0,
    )
    stale_binding = ClaimEvidenceBinding(
        section_id=section.section_id,
        claim_text="A different old claim is supported [1].",
        citation_number=1,
        chunk_id="old-chunk",
        support_score=0.9,
        verdict="supported",
    )
    state = cast(
        AgentState,
        {
            **_make_state_with_written_section(base_state, [section], index=0),
            "papers": [
                PaperMetadata(
                    paper_id="p1",
                    title="Evidence Paper",
                    authors=["Author"],
                    abstract="Evidence abstract.",
                    source="openalex",
                )
            ],
            "claim_evidence_bindings": [stale_binding],
        },
    )

    with patch(
        "seele_scholar_agent.nodes.reviewer.invoke_with_retry",
        new_callable=AsyncMock,
        return_value={"approved": True, "score": 8, "issues": [], "summary": "Good."},
    ):
        node = ReviewerNode(llm=mock_llm, prompts=mock_prompts)
        result = await node.review(state)

    assert result["status"] == "writing"
    assert result["current_review"]["approved"] is False
    assert result["quality_issues"][0].code == "CLAIM_MISSING_EVIDENCE_PACKET"


@pytest.mark.asyncio
async def test_reviewer_rejects_methodology_statistical_gaps(
    mock_llm, mock_prompts, base_state
):
    section = _written_section(
        "Results",
        content="Our model significantly improves accuracy on the dataset.",
        index=0,
    )
    state = cast(
        AgentState,
        {
            **_make_state_with_written_section(base_state, [section], index=0),
            "paper_type": "empirical",
        },
    )

    with patch(
        "seele_scholar_agent.nodes.reviewer.invoke_with_retry",
        new_callable=AsyncMock,
        return_value={"approved": True, "score": 8, "issues": [], "summary": "Good."},
    ):
        node = ReviewerNode(llm=mock_llm, prompts=mock_prompts)
        result = await node.review(state)

    codes = {issue.code for issue in result["quality_issues"]}
    assert result["status"] == "writing"
    assert result["current_review"]["approved"] is False
    assert "METHODOLOGY_BASELINE_FAIRNESS_MISSING" in codes
    assert "METHODOLOGY_METRIC_DEFINITION_MISSING" in codes
    assert "METHODOLOGY_SIGNIFICANCE_UNCERTAINTY_MISSING" in codes


@pytest.mark.asyncio
async def test_reviewer_rejects_paragraph_style_quality_gaps(
    mock_llm, mock_prompts, base_state
):
    paragraph = "In today's world, this section discusses the topic in broad terms."
    section = _written_section(
        "Discussion",
        content=f"{paragraph}\n\n{paragraph}",
        index=0,
    )
    state = _make_state_with_written_section(base_state, [section], index=0)

    with patch(
        "seele_scholar_agent.nodes.reviewer.invoke_with_retry",
        new_callable=AsyncMock,
        return_value={"approved": True, "score": 8, "issues": [], "summary": "Good."},
    ):
        node = ReviewerNode(llm=mock_llm, prompts=mock_prompts)
        result = await node.review(state)

    codes = {issue.code for issue in result["quality_issues"]}
    assert result["status"] == "writing"
    assert result["current_review"]["approved"] is False
    assert "PARAGRAPH_DUPLICATE" in codes
    assert "PARAGRAPH_GENERIC_TEMPLATE" in codes


@pytest.mark.asyncio
async def test_proposal_schedule_plan_sentence_does_not_trigger_missing_citation(
    mock_llm, mock_prompts, base_state
):
    section = _written_section(
        "研究計画・スケジュール",
        content="1年次後期にプロトタイプを実装し、評価手法を検証する。",
        index=0,
    )
    state = cast(
        AgentState,
        {
            **_make_state_with_written_section(base_state, [section], index=0),
            "document_type": "research_proposal",
            "language": "ja",
        },
    )

    with patch(
        "seele_scholar_agent.nodes.reviewer.invoke_with_retry",
        new_callable=AsyncMock,
        return_value={"approved": True, "score": 8, "issues": [], "summary": "Good."},
    ):
        node = ReviewerNode(llm=mock_llm, prompts=mock_prompts)
        result = await node.review(state)

    issues = result["current_review"]["issues"]
    assert not any(issue["type"] == "missing_citation" for issue in issues)
    assert any(issue["type"] == "format_issue" for issue in issues)


@pytest.mark.asyncio
async def test_reviewer_still_rejects_academic_uncited_factual_claim(
    mock_llm, mock_prompts, base_state
):
    section = _written_section(
        "Results",
        content="The model improves accuracy by 12%.",
        index=0,
    )
    state = _make_state_with_written_section(base_state, [section], index=0)

    with patch(
        "seele_scholar_agent.nodes.reviewer.invoke_with_retry",
        new_callable=AsyncMock,
        return_value={"approved": True, "score": 8, "issues": [], "summary": "Good."},
    ):
        node = ReviewerNode(llm=mock_llm, prompts=mock_prompts)
        result = await node.review(state)

    assert result["status"] == "writing"
    assert result["current_review"]["approved"] is False
    assert any(
        issue["type"] == "missing_citation" for issue in result["current_review"]["issues"]
    )


@pytest.mark.asyncio
async def test_proposal_short_paragraph_not_blocked_for_missing_evidence_analysis(
    mock_llm, mock_prompts, base_state
):
    section = _written_section(
        "研究方法",
        content=(
            "本研究では、ユーザー定義感情曲線に基づくゲーム音響設計支援手法を"
            "設計し、Wwise 上で実装可能な制作フローとして整理する。"
        ),
        index=0,
    )
    state = cast(
        AgentState,
        {
            **_make_state_with_written_section(base_state, [section], index=0),
            "document_type": "research_proposal",
            "language": "ja",
        },
    )

    with patch(
        "seele_scholar_agent.nodes.reviewer.invoke_with_retry",
        new_callable=AsyncMock,
        return_value={"approved": True, "score": 8, "issues": [], "summary": "Good."},
    ):
        node = ReviewerNode(llm=mock_llm, prompts=mock_prompts)
        result = await node.review(state)

    assert result["status"] == "completed"
    assert result["current_review"]["approved"] is True
    codes = {issue.code for issue in result.get("quality_issues", [])}
    assert "PARAGRAPH_STRUCTURE_INCOMPLETE" not in codes


@pytest.mark.asyncio
async def test_proposal_invalid_citation_number_still_reports_missing_citation(
    mock_llm, mock_prompts, base_state
):
    papers = [
        PaperMetadata(
            paper_id="p1",
            title="Evidence Paper",
            authors=["Author"],
            abstract="Evidence abstract.",
            source="openalex",
        )
    ]
    section = _written_section(
        "研究背景",
        content="先行研究は音響設計の重要性を示している [2]。",
        index=0,
    )
    state = cast(
        AgentState,
        {
            **_make_state_with_written_section(base_state, [section], index=0),
            "document_type": "research_proposal",
            "papers": papers,
            "language": "ja",
        },
    )

    with patch(
        "seele_scholar_agent.nodes.reviewer.invoke_with_retry",
        new_callable=AsyncMock,
        return_value={"approved": True, "score": 8, "issues": [], "summary": "Good."},
    ):
        node = ReviewerNode(llm=mock_llm, prompts=mock_prompts)
        result = await node.review(state)

    issues = result["current_review"]["issues"]
    assert result["status"] == "writing"
    assert any(
        issue["type"] == "missing_citation" and "[2]" in issue["description"]
        for issue in issues
    )


@pytest.mark.asyncio
async def test_proposal_unverified_binding_is_deferred_warning(
    mock_llm, mock_prompts, base_state
):
    papers = [
        PaperMetadata(
            paper_id="p1",
            title="Evidence Paper",
            authors=["Author"],
            abstract="Evidence abstract.",
            source="openalex",
        )
    ]
    section = _written_section(
        "研究背景",
        content="先行研究は音響体験に関する重要な知見を示している [1]。",
        index=0,
    )
    binding = ClaimEvidenceBinding(
        section_id=section.section_id,
        claim_text=section.content,
        citation_number=1,
        source_paper_id="p1",
        verdict="unverified",
        diagnostics={"evidence_packet_count": 0, "candidate_count": 0},
    )
    state = cast(
        AgentState,
        {
            **_make_state_with_written_section(base_state, [section], index=0),
            "document_type": "research_proposal",
            "papers": papers,
            "claim_evidence_bindings": [binding],
            "language": "ja",
        },
    )

    with patch(
        "seele_scholar_agent.nodes.reviewer.invoke_with_retry",
        new_callable=AsyncMock,
        return_value={"approved": True, "score": 8, "issues": [], "summary": "Good."},
    ):
        node = ReviewerNode(llm=mock_llm, prompts=mock_prompts)
        result = await node.review(state)

    assert result["status"] == "completed"
    assert result["current_review"]["approved"] is True
    assert result["quality_issues"][0].severity == "warning"
    assert result["quality_issues"][0].details["deferred"] is True
    assert result["review_diagnostics"]["deferred_quality_issue_count"] == 1


@pytest.mark.asyncio
async def test_proposal_method_plan_overview_can_pass_with_score_seven(
    mock_llm, mock_prompts, base_state
):
    section = _written_section(
        "研究方法・計画",
        content=(
            "本研究では、ゲーム音響制作の事例資料と Wwise を用い、感情曲線に沿った"
            "音響設計手順を整理する。分析では既存作品の音響変化を比較し、制作した"
            "プロトタイプを少人数の試聴で検証する。修士課程では前期に文献整理と"
            "資料収集を行い、後期に実装と検証を進める計画である。"
        ),
        index=0,
    )
    state = cast(
        AgentState,
        {
            **_make_state_with_written_section(base_state, [section], index=0),
            "document_type": "research_proposal",
            "language": "ja",
        },
    )

    with patch(
        "seele_scholar_agent.nodes.reviewer.invoke_with_retry",
        new_callable=AsyncMock,
        return_value={
            "approved": False,
            "score": 7,
            "issues": [
                {
                    "type": "weak_argument",
                    "description": "Method lacks paper-level protocol details.",
                    "suggestion": "Add variables and statistical tests.",
                }
            ],
            "summary": "Compact but feasible.",
        },
    ):
        node = ReviewerNode(llm=mock_llm, prompts=mock_prompts)
        result = await node.review(state)

    assert result["status"] == "completed"
    assert result["current_review"]["approved"] is True
    assert result["review_diagnostics"]["blocking_issue_count"] == 0
    assert result["review_diagnostics"]["compound_title_detected"] is True


@pytest.mark.asyncio
async def test_proposal_high_score_citation_warning_is_not_blocking(
    mock_llm, mock_prompts, base_state
):
    section = _written_section(
        "研究背景",
        content="研究背景として、ゲーム音響設計では感情変化の扱いが重要である。",
        index=0,
    )
    state = cast(
        AgentState,
        {
            **_make_state_with_written_section(base_state, [section], index=0),
            "document_type": "research_proposal",
            "language": "ja",
        },
    )

    with patch(
        "seele_scholar_agent.nodes.reviewer.invoke_with_retry",
        new_callable=AsyncMock,
        return_value={
            "approved": False,
            "score": 8,
            "issues": [
                {
                    "type": "missing_citation",
                    "description": "Background claim should cite prior work.",
                    "suggestion": "Add a citation.",
                }
            ],
            "summary": "Mostly acceptable.",
        },
    ):
        node = ReviewerNode(llm=mock_llm, prompts=mock_prompts)
        result = await node.review(state)

    assert result["status"] == "completed"
    assert result["current_review"]["approved"] is True
    issue = result["current_review"]["issues"][0]
    assert issue["blocking"] is False
    assert issue["category"] == "citation_warning"


@pytest.mark.asyncio
async def test_proposal_novelty_outcome_compound_accepts_brief_outcome(
    mock_llm, mock_prompts, base_state
):
    section = _written_section(
        "新規性・期待される成果",
        content=(
            "本研究の新規性は、感情曲線を音響設計の操作可能な指針として扱う点にある。"
            "期待される成果は、修士研究として再利用可能な制作手順と、申請先研究室での"
            "作品分析に応用できる知見を示すことである。"
        ),
        index=0,
    )
    state = cast(
        AgentState,
        {
            **_make_state_with_written_section(base_state, [section], index=0),
            "document_type": "research_proposal",
            "language": "ja",
        },
    )

    with patch(
        "seele_scholar_agent.nodes.reviewer.invoke_with_retry",
        new_callable=AsyncMock,
        return_value={"approved": True, "score": 8, "issues": [], "summary": "Good."},
    ):
        node = ReviewerNode(llm=mock_llm, prompts=mock_prompts)
        result = await node.review(state)

    assert result["status"] == "completed"
    assert result["current_review"]["approved"] is True
    assert result["review_diagnostics"]["missing_core_tasks"] == []


@pytest.mark.asyncio
async def test_proposal_enumeration_mismatch_blocks_revision(
    mock_llm, mock_prompts, base_state
):
    section = _written_section(
        "期待される成果",
        content=(
            "期待される成果は三点である。第一に、制作手順を整理する。"
            "第二に、研究室で議論できる分析観点を提示する。"
        ),
        index=0,
    )
    state = cast(
        AgentState,
        {
            **_make_state_with_written_section(base_state, [section], index=0),
            "document_type": "research_proposal",
            "language": "ja",
        },
    )

    with patch(
        "seele_scholar_agent.nodes.reviewer.invoke_with_retry",
        new_callable=AsyncMock,
        return_value={"approved": True, "score": 8, "issues": [], "summary": "Good."},
    ):
        node = ReviewerNode(llm=mock_llm, prompts=mock_prompts)
        result = await node.review(state)

    assert result["status"] == "writing"
    assert result["current_review"]["approved"] is False
    assert result["review_diagnostics"]["blocking_issue_count"] >= 1
    assert any(
        issue["blocking"] for issue in result["review_diagnostics"]["issue_categories"]["blocking"]
    )


@pytest.mark.asyncio
async def test_proposal_japanese_stage_enumeration_is_recognized(
    mock_llm, mock_prompts, base_state
):
    section = _written_section(
        "研究方法・計画",
        content=(
            "本研究は三段階で進める。第一段階ではWwise上でRTPC制御を実装する。"
            "第二段階では感情次元から音楽特徴量への対応を整理する。"
            "第三段階では参加者評価により没入感と感情一致度を検証する。"
        ),
        index=0,
    )
    state = cast(
        AgentState,
        {
            **_make_state_with_written_section(base_state, [section], index=0),
            "document_type": "research_proposal",
            "language": "ja",
        },
    )

    with patch(
        "seele_scholar_agent.nodes.reviewer.invoke_with_retry",
        new_callable=AsyncMock,
        return_value={"approved": True, "score": 8, "issues": [], "summary": "Good."},
    ):
        node = ReviewerNode(llm=mock_llm, prompts=mock_prompts)
        result = await node.review(state)

    assert result["status"] == "completed"
    assert result["current_review"]["approved"] is True
    assert not any(
        issue["code"] == "PROPOSAL_ENUMERATION_INCONSISTENT"
        for issue in result.get("quality_issues", [])
    )


@pytest.mark.asyncio
async def test_proposal_truncated_or_missing_core_task_blocks(
    mock_llm, mock_prompts, base_state
):
    sections = [
        _written_section(
            "研究方法・計画",
            content="本研究の目的は音響体験を明らかにすることである。",
            index=0,
        )
    ]
    state = cast(
        AgentState,
        {
            **_make_state_with_written_section(base_state, sections, index=0),
            "document_type": "research_proposal",
            "language": "ja",
        },
    )

    with patch(
        "seele_scholar_agent.nodes.reviewer.invoke_with_retry",
        new_callable=AsyncMock,
        return_value={"approved": True, "score": 8, "issues": [], "summary": "Good."},
    ):
        node = ReviewerNode(llm=mock_llm, prompts=mock_prompts)
        result = await node.review(state)

    assert result["status"] == "writing"
    assert result["current_review"]["approved"] is False
    assert set(result["review_diagnostics"]["missing_core_tasks"]) >= {"method", "plan"}

    truncated = sections[0].model_copy(
        update={
            "content": (
                "本研究では、事例資料と制作ツールを用いて分析し、前期に資料収集を進める"
            )
        }
    )
    truncated_state = cast(
        AgentState,
        {
            **_make_state_with_written_section(base_state, [truncated], index=0),
            "document_type": "research_proposal",
            "language": "ja",
        },
    )
    with patch(
        "seele_scholar_agent.nodes.reviewer.invoke_with_retry",
        new_callable=AsyncMock,
        return_value={"approved": True, "score": 8, "issues": [], "summary": "Good."},
    ):
        result = await ReviewerNode(llm=mock_llm, prompts=mock_prompts).review(
            truncated_state
        )

    assert result["status"] == "writing"
    assert result["current_review"]["approved"] is False
    assert any(
        "truncated" in issue["description"]
        for issue in result["review_diagnostics"]["issue_categories"]["blocking"]
    )
