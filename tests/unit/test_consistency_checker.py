"""Unit tests for ConsistencyCheckerNode — CC-01 through CC-05."""

from typing import cast
from unittest.mock import AsyncMock, patch

import pytest
from seele_scholar_agent.nodes.consistency_checker import ConsistencyCheckerNode
from seele_scholar_agent.state import AgentState, ConsistencyIssue, SectionDraft


def _approved_section(title: str, content: str, order_index: int = 1) -> SectionDraft:
    return SectionDraft(
        section_id=f"s_{title}",
        title=title,
        description="",
        order_index=order_index,
        content=content,
        status="approved",
    )


# ---------------------------------------------------------------------------
# CC-01: 2+ approved sections → LLM called, issues returned
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_consistency_checker_returns_issues_from_llm(mock_llm, mock_prompts, base_state):
    sections = [
        _approved_section("Introduction", "Intro text about transformers.", order_index=1),
        _approved_section("Related Work", "Prior work on attention mechanisms.", order_index=2),
    ]
    state = cast(AgentState, {**base_state, "sections": sections})

    llm_response = {
        "issues": [
            {
                "issue_type": "terminology",
                "description": "Inconsistent use of 'attention'.",
                "sections_involved": ["Introduction", "Related Work"],
                "suggestion": "Use consistent terminology.",
            }
        ]
    }

    with patch(
        "seele_scholar_agent.nodes.consistency_checker.invoke_with_retry",
        new_callable=AsyncMock,
        return_value=llm_response,
    ):
        node = ConsistencyCheckerNode(llm=mock_llm, prompts=mock_prompts)
        result = await node.check(state)

    assert result["consistency_checked"] is True
    issues = result["consistency_issues"]
    # 3 parallel sub-checks (terminology, logic, citation) each return 1 issue from the mock
    assert len(issues) == 3
    assert all(i.issue_type == "terminology" for i in issues)


# ---------------------------------------------------------------------------
# CC-02: LLM returns empty issues → consistency_issues=[], checked=True
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_consistency_checker_empty_issues(mock_llm, mock_prompts, base_state):
    sections = [
        _approved_section("Introduction", "Content A.", order_index=1),
        _approved_section("Methods", "Content B.", order_index=2),
    ]
    state = cast(AgentState, {**base_state, "sections": sections})

    with patch(
        "seele_scholar_agent.nodes.consistency_checker.invoke_with_retry",
        new_callable=AsyncMock,
        return_value={"issues": []},
    ):
        node = ConsistencyCheckerNode(llm=mock_llm, prompts=mock_prompts)
        result = await node.check(state)

    assert result["consistency_checked"] is True
    assert result["consistency_issues"] == []


# ---------------------------------------------------------------------------
# CC-03: Fewer than 2 approved sections → LLM skipped, checked=True, issues=[]
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_consistency_checker_skips_when_not_enough_sections(
    mock_llm, mock_prompts, base_state
):
    sections = [
        _approved_section("Introduction", "Only one approved section.", order_index=1),
    ]
    state = cast(AgentState, {**base_state, "sections": sections})

    invoke_called = False

    async def mock_invoke(*_args, **_kwargs):
        nonlocal invoke_called
        invoke_called = True
        return {"issues": []}

    with patch(
        "seele_scholar_agent.nodes.consistency_checker.invoke_with_retry", side_effect=mock_invoke
    ):
        node = ConsistencyCheckerNode(llm=mock_llm, prompts=mock_prompts)
        result = await node.check(state)

    assert not invoke_called
    assert result["consistency_checked"] is True
    assert result["consistency_issues"] == []


# ---------------------------------------------------------------------------
# CC-04: LLM raises exception → issues=[], checked=True, no exception propagates
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_consistency_checker_llm_failure_returns_empty(mock_llm, mock_prompts, base_state):
    sections = [
        _approved_section("Introduction", "Content X.", order_index=1),
        _approved_section("Conclusion", "Content Y.", order_index=2),
    ]
    state = cast(AgentState, {**base_state, "sections": sections})

    with patch(
        "seele_scholar_agent.nodes.consistency_checker.invoke_with_retry",
        side_effect=Exception("LLM error"),
    ):
        node = ConsistencyCheckerNode(llm=mock_llm, prompts=mock_prompts)
        result = await node.check(state)

    assert result["consistency_checked"] is True
    assert result["consistency_issues"] == []


# ---------------------------------------------------------------------------
# CC-05: ConsistencyIssue fields correctly mapped from LLM response
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_consistency_checker_issue_fields_mapped_correctly(
    mock_llm, mock_prompts, base_state
):
    sections = [
        _approved_section("Introduction", "Some intro.", order_index=1),
        _approved_section("Discussion", "Discussion content.", order_index=2),
    ]
    state = cast(AgentState, {**base_state, "sections": sections})

    llm_response = {
        "issues": [
            {
                "issue_type": "logic",
                "description": "Contradicting claim about model size.",
                "sections_involved": ["Introduction", "Discussion"],
                "suggestion": "Align the claims in both sections.",
            }
        ]
    }

    with patch(
        "seele_scholar_agent.nodes.consistency_checker.invoke_with_retry",
        new_callable=AsyncMock,
        return_value=llm_response,
    ):
        node = ConsistencyCheckerNode(llm=mock_llm, prompts=mock_prompts)
        result = await node.check(state)

    issue: ConsistencyIssue = result["consistency_issues"][0]
    assert issue.issue_type == "logic"
    assert issue.description == "Contradicting claim about model size."
    assert issue.sections_involved == ["Introduction", "Discussion"]
    assert issue.suggestion == "Align the claims in both sections."
