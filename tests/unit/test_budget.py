from datetime import datetime

import pytest
from seele_scholar_agent.budget import (
    BudgetAllocatorNode,
    BudgetPolicy,
    BudgetState,
    LengthGateNode,
    SectionBudget,
)
from seele_scholar_agent.budget.length_gate import count_length
from seele_scholar_agent.state import AgentState, SectionDraft


def test_count_length_words_and_chars():
    assert count_length("Alpha beta, gamma.", "words") == 3
    assert count_length("本研究 では 音響設計を扱う。", "chars") == 13


@pytest.mark.asyncio
async def test_length_gate_reports_over_budget():
    section = SectionDraft(
        section_id="section_0",
        title="Introduction",
        order_index=0,
        content="one two three four five",
        status="review",
    )
    state: AgentState = {
        "thread_id": "budget-test",
        "topic": "Budget control",
        "language": "en",
        "created_at": datetime.now(),
        "tenant_id": None,
        "broad_papers": [],
        "proposed_topics": [],
        "papers": [],
        "search_queries": [],
        "outline": None,
        "outline_approved": True,
        "sections": [section],
        "current_section_index": 0,
        "sections_completed": [],
        "review_history": [],
        "section_candidates": [],
        "current_review": None,
        "rag_context": [],
        "evidence_packets": [],
        "claim_evidence_bindings": [],
        "section_summaries": [],
        "paper_summaries": [],
        "status": "reviewing",
        "error_message": None,
        "max_revisions": 3,
        "revision_count": 0,
        "references": [],
        "consistency_issues": [],
        "consistency_checked": False,
        "quality_issues": [],
        "quality_issue_history": [],
        "budget_state": BudgetState(
            total_target=10,
            sections={
                "section_0": SectionBudget(
                    section_id="section_0",
                    target=3,
                    hard_limit=3,
                    unit="words",
                )
            },
        ),
    }

    result = await LengthGateNode(BudgetPolicy()).check(state)

    assert result["budget_diagnostics"]["over_budget"] is True
    assert result["budget_diagnostics"]["actual"] == 5
    assert result["quality_issues"][0].code == "SECTION_OVER_BUDGET"
    assert result["budget_state"].section_actuals["section_0"] == 5


@pytest.mark.asyncio
async def test_budget_allocator_node_uses_caller_hook():
    section = SectionDraft(section_id="section_0", title="Intro", order_index=0)
    budget_state = BudgetState(
        total_target=100,
        remaining=70,
        sections={"section_0": SectionBudget(section_id="section_0", target=50)},
    )
    state: AgentState = {
        "thread_id": "budget-allocator-test",
        "topic": "Budget control",
        "language": "en",
        "created_at": datetime.now(),
        "tenant_id": None,
        "broad_papers": [],
        "proposed_topics": [],
        "papers": [],
        "search_queries": [],
        "outline": None,
        "outline_approved": True,
        "sections": [section],
        "current_section_index": 0,
        "sections_completed": [],
        "review_history": [],
        "section_candidates": [],
        "current_review": None,
        "rag_context": [],
        "evidence_packets": [],
        "claim_evidence_bindings": [],
        "section_summaries": [],
        "paper_summaries": [],
        "status": "writing",
        "error_message": None,
        "max_revisions": 3,
        "revision_count": 0,
        "references": [],
        "consistency_issues": [],
        "consistency_checked": False,
        "quality_issues": [],
        "quality_issue_history": [],
        "budget_state": budget_state,
    }

    async def allocator(
        current: BudgetState, _sections: list[SectionDraft], _index: int
    ) -> BudgetState:
        return current.model_copy(update={"remaining": 42})

    result = await BudgetAllocatorNode(allocator).allocate(state)

    assert result["budget_state"].remaining == 42
