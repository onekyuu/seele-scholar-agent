"""Unit tests for FinalizerNode — FIN-01 through FIN-05."""

from typing import Literal, cast
from unittest.mock import AsyncMock, patch

import pytest
from langchain_core.messages import AIMessage

from seele_scholar_agent.nodes.finalizer import FinalizerNode
from seele_scholar_agent.state import AgentState, SectionDraft


def _make_section(
    title: str,
    status: Literal["pending", "writing", "review", "approved", "auto_generated"] = "pending",
    content: str = "",
    section_id: str | None = None,
    order_index: int = 1,
) -> SectionDraft:
    return SectionDraft(
        section_id=section_id or f"s_{title}",
        title=title,
        description="",
        order_index=order_index,
        content=content,
        status=status,
    )


# ---------------------------------------------------------------------------
# FIN-01: pending "摘要" section → LLM called, status becomes "auto_generated"
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_finalizer_generates_abstract_zh(mock_llm, mock_prompts, base_state, sample_papers):
    sections = [
        _make_section("Introduction", status="approved", content="Intro content.", order_index=1),
        _make_section("摘要", status="pending", order_index=2),
    ]
    state = cast(
        AgentState,
        {**base_state, "papers": sample_papers, "sections": sections, "language": "zh"},
    )

    with patch(
        "seele_scholar_agent.nodes.finalizer.invoke_with_retry",
        new_callable=AsyncMock,
        return_value=AIMessage(content="This is the generated abstract."),
    ):
        node = FinalizerNode(llm=mock_llm, prompts=mock_prompts)
        result = await node.finalize(state)

    updated = result["sections"]
    abstract_section = next(s for s in updated if s.title == "摘要")
    assert abstract_section.status == "auto_generated"
    assert abstract_section.content == "This is the generated abstract."
    assert result["status"] == "completed"


# ---------------------------------------------------------------------------
# FIN-02: pending "结论" section → LLM called, status becomes "auto_generated"
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_finalizer_generates_conclusion_zh(mock_llm, mock_prompts, base_state):
    sections = [
        _make_section("Introduction", status="approved", content="Body content.", order_index=1),
        _make_section("结论", status="pending", order_index=2),
    ]
    state = cast(AgentState, {**base_state, "sections": sections, "language": "zh"})

    with patch(
        "seele_scholar_agent.nodes.finalizer.invoke_with_retry",
        new_callable=AsyncMock,
        return_value=AIMessage(content="Generated conclusion text."),
    ):
        node = FinalizerNode(llm=mock_llm, prompts=mock_prompts)
        result = await node.finalize(state)

    conclusion = next(s for s in result["sections"] if s.title == "结论")
    assert conclusion.status == "auto_generated"
    assert "Generated conclusion text." in conclusion.content


# ---------------------------------------------------------------------------
# FIN-03: No pending abstract/conclusion sections → LLM not called, status="completed"
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_finalizer_skips_when_no_pending_abstract_or_conclusion(
    mock_llm, mock_prompts, base_state
):
    sections = [
        _make_section("Introduction", status="approved", content="Intro.", order_index=1),
        _make_section("Methods", status="pending", order_index=2),
    ]
    state = cast(AgentState, {**base_state, "sections": sections})

    invoke_called = False

    async def mock_invoke(*_args, **_kwargs):
        nonlocal invoke_called
        invoke_called = True
        return AIMessage(content="Should not be called.")

    with patch("seele_scholar_agent.nodes.finalizer.invoke_with_retry", side_effect=mock_invoke):
        node = FinalizerNode(llm=mock_llm, prompts=mock_prompts)
        result = await node.finalize(state)

    assert not invoke_called
    assert result["status"] == "completed"
    assert "sections" not in result


# ---------------------------------------------------------------------------
# FIN-04: LLM raises exception → section remains unchanged, no exception propagates
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_finalizer_llm_failure_leaves_section_unchanged(mock_llm, mock_prompts, base_state):
    sections = [
        _make_section("Abstract", status="pending", order_index=1),
    ]
    state = cast(AgentState, {**base_state, "sections": sections, "language": "en"})

    with patch(
        "seele_scholar_agent.nodes.finalizer.invoke_with_retry",
        side_effect=Exception("LLM timeout"),
    ):
        node = FinalizerNode(llm=mock_llm, prompts=mock_prompts)
        result = await node.finalize(state)

    assert result["status"] == "completed"
    assert "sections" not in result


# ---------------------------------------------------------------------------
# FIN-05: English titles "Abstract" and "Conclusion" are matched (case-insensitive)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_finalizer_matches_english_titles(mock_llm, mock_prompts, base_state):
    sections = [
        _make_section("Introduction", status="approved", content="Intro.", order_index=1),
        _make_section("Abstract", status="pending", order_index=2),
        _make_section("Conclusion", status="pending", order_index=3),
    ]
    state = cast(AgentState, {**base_state, "sections": sections, "language": "en"})

    call_count = 0

    async def mock_invoke(*_args, **_kwargs):
        nonlocal call_count
        call_count += 1
        return AIMessage(content=f"Generated content {call_count}.")

    with patch("seele_scholar_agent.nodes.finalizer.invoke_with_retry", side_effect=mock_invoke):
        node = FinalizerNode(llm=mock_llm, prompts=mock_prompts)
        result = await node.finalize(state)

    assert call_count == 2
    auto_generated = [s for s in result["sections"] if s.status == "auto_generated"]
    assert len(auto_generated) == 2
    titles = {s.title for s in auto_generated}
    assert titles == {"Abstract", "Conclusion"}
