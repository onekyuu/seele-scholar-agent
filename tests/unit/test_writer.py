"""Unit tests for WriterNode — W-01 through W-17."""

from typing import cast
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import AIMessage

from seele_scholar_agent.nodes.writer import WriterNode
from seele_scholar_agent.state import AgentState, SectionDraft


# ---------------------------------------------------------------------------
# W-01: Normal write → sections[0].content filled, status="reviewing"
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_writer_normal_write(mock_llm, mock_prompts, state_with_outline):
    with patch(
        "seele_scholar_agent.nodes.writer.invoke_with_retry",
        new_callable=AsyncMock,
        return_value=AIMessage(content="This is the introduction content."),
    ):
        node = WriterNode(llm=mock_llm, prompts=mock_prompts)
        result = await node.write(state_with_outline)

    assert result["sections"][0].content == "This is the introduction content."
    assert result["sections"][0].status == "review"
    assert result["status"] == "reviewing"


# ---------------------------------------------------------------------------
# W-02: sections[0].status="approved" → skip, current_section_index becomes 1
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_writer_skips_approved_section(mock_llm, mock_prompts, state_with_outline):
    sections = list(state_with_outline["sections"])
    sections[0] = sections[0].model_copy(update={"status": "approved"})
    state = cast(AgentState, {**state_with_outline, "sections": sections})

    node = WriterNode(llm=mock_llm, prompts=mock_prompts)
    result = await node.write(state)

    assert result["current_section_index"] == 1
    assert result["status"] == "writing"


# ---------------------------------------------------------------------------
# W-03: current_index >= len(sections) → status="completed"
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_writer_out_of_bounds_index_returns_completed(
    mock_llm, mock_prompts, state_with_outline
):
    state = cast(AgentState, {**state_with_outline, "current_section_index": 999})

    node = WriterNode(llm=mock_llm, prompts=mock_prompts)
    result = await node.write(state)

    assert result["status"] == "completed"


# ---------------------------------------------------------------------------
# W-04: _move_to_next when last section already approved → status="completed"
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_writer_move_to_next_last_section_completed(
    mock_llm, mock_prompts, state_with_outline
):
    sections = list(state_with_outline["sections"])
    last_index = len(sections) - 1
    sections[last_index] = sections[last_index].model_copy(update={"status": "approved"})
    state = cast(
        AgentState,
        {**state_with_outline, "sections": sections, "current_section_index": last_index},
    )

    node = WriterNode(llm=mock_llm, prompts=mock_prompts)
    result = await node.write(state)

    assert result["status"] == "completed"


# ---------------------------------------------------------------------------
# W-05: LLM failure → status="failed", error_message set, sections[0].status="pending"
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_writer_llm_failure_returns_failed(mock_llm, mock_prompts, state_with_outline):
    with patch(
        "seele_scholar_agent.nodes.writer.invoke_with_retry",
        side_effect=Exception("network error"),
    ):
        node = WriterNode(llm=mock_llm, prompts=mock_prompts)
        result = await node.write(state_with_outline)

    assert result["status"] == "failed"
    assert result["error_message"] is not None
    assert "network error" in result["error_message"]
    assert result["sections"][0].status == "pending"


# ---------------------------------------------------------------------------
# W-06: _clean_content removes ``` code block markers
# ---------------------------------------------------------------------------


def test_writer_clean_content_removes_code_fences(mock_llm, mock_prompts):
    node = WriterNode(llm=mock_llm, prompts=mock_prompts)
    raw = "```\nsome code\n```\nActual content here."
    cleaned = node._clean_content(raw)

    assert "```" not in cleaned
    assert "some code" in cleaned
    assert "Actual content here." in cleaned


# ---------------------------------------------------------------------------
# W-07: _clean_content preserves regular content
# ---------------------------------------------------------------------------


def test_writer_clean_content_preserves_regular_text(mock_llm, mock_prompts):
    node = WriterNode(llm=mock_llm, prompts=mock_prompts)
    raw = "Line one.\nLine two.\nLine three."
    cleaned = node._clean_content(raw)

    assert "Line one." in cleaned
    assert "Line two." in cleaned
    assert "Line three." in cleaned


# ---------------------------------------------------------------------------
# W-08: _build_review_comments with issues → "- " prefix on each
# ---------------------------------------------------------------------------


def test_writer_build_review_comments_with_comments(mock_llm, mock_prompts):
    node = WriterNode(llm=mock_llm, prompts=mock_prompts)
    section = MagicMock()
    section.review_comments = ["Needs more detail.", "Fix citation."]
    result = node._build_review_comments(section)

    assert result == "- Needs more detail.\n- Fix citation."


# ---------------------------------------------------------------------------
# W-09: _build_review_comments no issues → returns "无"
# ---------------------------------------------------------------------------


def test_writer_build_review_comments_empty(mock_llm, mock_prompts):
    node = WriterNode(llm=mock_llm, prompts=mock_prompts)
    section = MagicMock()
    section.review_comments = []
    result = node._build_review_comments(section)

    assert result == "无"


# ---------------------------------------------------------------------------
# W-10: _build_outline_json with outline → multi-line containing section titles
# ---------------------------------------------------------------------------


def test_writer_build_outline_json_with_outline(mock_llm, mock_prompts, sample_outline):
    node = WriterNode(llm=mock_llm, prompts=mock_prompts)
    result = node._build_outline_json(sample_outline)

    assert "Introduction" in result
    assert "Related Work" in result
    assert "Conclusion" in result
    assert "\n" in result


# ---------------------------------------------------------------------------
# W-11: _build_outline_json outline=None → returns ""
# ---------------------------------------------------------------------------


def test_writer_build_outline_json_none(mock_llm, mock_prompts):
    node = WriterNode(llm=mock_llm, prompts=mock_prompts)
    result = node._build_outline_json(None)

    assert result == ""


# ---------------------------------------------------------------------------
# W-12: rag_retriever exists → used instead of state rag_context
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_writer_uses_rag_retriever_when_provided(mock_llm, mock_prompts, state_with_outline):
    rag_called_with: list[str] = []

    async def mock_rag_retriever(query: str):
        rag_called_with.append(query)
        return []

    with patch(
        "seele_scholar_agent.nodes.writer.invoke_with_retry",
        new_callable=AsyncMock,
        return_value=AIMessage(content="Content."),
    ):
        node = WriterNode(llm=mock_llm, prompts=mock_prompts, rag_retriever=mock_rag_retriever)
        await node.write(state_with_outline)

    assert len(rag_called_with) == 1
    assert state_with_outline["topic"] in rag_called_with[0]


# ---------------------------------------------------------------------------
# W-13: After writing, revision_count unchanged (stays 0)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_writer_revision_count_unchanged_after_write(
    mock_llm, mock_prompts, state_with_outline
):
    with patch(
        "seele_scholar_agent.nodes.writer.invoke_with_retry",
        new_callable=AsyncMock,
        return_value=AIMessage(content="Introduction text."),
    ):
        node = WriterNode(llm=mock_llm, prompts=mock_prompts)
        result = await node.write(state_with_outline)

    assert result["sections"][0].revision_count == 0


# ---------------------------------------------------------------------------
# W-14: _build_previous_sections_context with completed sections → contains title and content
# ---------------------------------------------------------------------------


def test_writer_build_previous_sections_with_content(mock_llm, mock_prompts):
    node = WriterNode(llm=mock_llm, prompts=mock_prompts)
    sections = [
        SectionDraft(
            section_id="s0",
            title="Introduction",
            description="",
            order_index=1,
            content="This is the introduction content about LLMs.",
            status="approved",
        ),
        SectionDraft(
            section_id="s1",
            title="Methods",
            description="",
            order_index=2,
            content="",
            status="pending",
        ),
    ]
    result = node._build_previous_sections_context(sections, current_index=1)

    assert "Introduction" in result
    assert "introduction content" in result


# ---------------------------------------------------------------------------
# W-15: _build_previous_sections_context no completed sections → returns "无"
# ---------------------------------------------------------------------------


def test_writer_build_previous_sections_empty(mock_llm, mock_prompts):
    node = WriterNode(llm=mock_llm, prompts=mock_prompts)
    sections = [
        SectionDraft(
            section_id="s0",
            title="Introduction",
            description="",
            order_index=1,
            content="",
            status="pending",
        ),
    ]
    result = node._build_previous_sections_context(sections, current_index=0)

    assert result == "无"


# ---------------------------------------------------------------------------
# W-16: _build_numbered_papers with papers → contains [N] numbered entries
# ---------------------------------------------------------------------------


def test_writer_build_numbered_papers_with_papers(mock_llm, mock_prompts, sample_papers):
    node = WriterNode(llm=mock_llm, prompts=mock_prompts)
    result = node._build_numbered_papers(sample_papers)

    assert "[1]" in result
    assert "[2]" in result
    assert "[3]" in result
    assert "Attention Is All You Need" in result


# ---------------------------------------------------------------------------
# W-17: _build_numbered_papers with empty list → returns "无"
# ---------------------------------------------------------------------------


def test_writer_build_numbered_papers_empty(mock_llm, mock_prompts):
    node = WriterNode(llm=mock_llm, prompts=mock_prompts)
    result = node._build_numbered_papers([])

    assert result == "无"
