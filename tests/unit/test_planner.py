"""Unit tests for PlannerNode — P-01 through P-14."""

from typing import cast
from unittest.mock import AsyncMock, patch

import pytest
from seele_scholar_agent.nodes.planner import PlannerNode
from seele_scholar_agent.state import AgentState, PaperMetadata

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_llm_result(
    title: str = "Test Paper",
    abstract: str = "Abstract.",
    sections: list[dict] | None = None,
    keywords: list[str] | None = None,
) -> dict:
    if sections is None:
        sections = [
            {"title": "Introduction", "description": "Intro", "order": 1, "key_points": []},
            {"title": "Related Work", "description": "Prior art", "order": 2, "key_points": []},
            {"title": "Conclusion", "description": "Summary", "order": 3, "key_points": []},
        ]
    return {
        "title": title,
        "abstract": abstract,
        "sections": sections,
        "keywords": keywords or ["test"],
    }


# ---------------------------------------------------------------------------
# P-01: Normal planning — outline returned with correct section count
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_planner_normal_output(mock_llm, mock_prompts, state_with_papers):
    """P-01: Normal planning produces outline with correct section count."""
    result_data = _make_llm_result(
        title="LLM Survey",
        sections=[
            {"title": "Introduction", "description": "Intro", "order": 1, "key_points": []},
            {"title": "Related Work", "description": "Prior", "order": 2, "key_points": []},
            {"title": "Conclusion", "description": "Summary", "order": 3, "key_points": []},
        ],
    )
    with patch(
        "seele_scholar_agent.nodes.planner.invoke_with_retry",
        new_callable=AsyncMock,
        return_value=result_data,
    ):
        node = PlannerNode(llm=mock_llm, prompts=mock_prompts)
        result = await node.plan(state_with_papers)

    assert result["outline"].title == "LLM Survey"
    assert len(result["sections"]) == 3


# ---------------------------------------------------------------------------
# P-02: Sections out of order — sorted by 'order' field
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_planner_sections_sorted_by_order(mock_llm, mock_prompts, state_with_papers):
    """P-02: Sections returned out of order are sorted correctly."""
    result_data = _make_llm_result(
        sections=[
            {"title": "Conclusion", "description": "Summary", "order": 3, "key_points": []},
            {"title": "Introduction", "description": "Intro", "order": 1, "key_points": []},
            {"title": "Related Work", "description": "Prior", "order": 2, "key_points": []},
        ]
    )
    with patch(
        "seele_scholar_agent.nodes.planner.invoke_with_retry",
        new_callable=AsyncMock,
        return_value=result_data,
    ):
        node = PlannerNode(llm=mock_llm, prompts=mock_prompts)
        result = await node.plan(state_with_papers)

    titles = [s.title for s in result["sections"]]
    assert titles == ["Introduction", "Related Work", "Conclusion"]


# ---------------------------------------------------------------------------
# P-03: LLM failure → _default_outline fallback, 5 sections
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_planner_llm_failure_uses_default_outline(mock_llm, mock_prompts, state_with_papers):
    """P-03: LLM failure triggers default outline with 5 sections."""
    with patch(
        "seele_scholar_agent.nodes.planner.invoke_with_retry",
        side_effect=Exception("timeout"),
    ):
        node = PlannerNode(llm=mock_llm, prompts=mock_prompts)
        result = await node.plan(state_with_papers)

    assert len(result["sections"]) == 5


# ---------------------------------------------------------------------------
# P-04: lang="zh" → first section title is "引言"
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_planner_zh_default_sections(mock_llm, mock_prompts, base_state):
    """P-04: zh fallback outline uses Chinese section titles."""
    state = cast(AgentState, {**base_state, "language": "zh", "papers": [], "status": "planning"})
    with patch(
        "seele_scholar_agent.nodes.planner.invoke_with_retry",
        side_effect=Exception("fail"),
    ):
        node = PlannerNode(llm=mock_llm, prompts=mock_prompts)
        result = await node.plan(state)

    assert result["sections"][0].title == "引言"


# ---------------------------------------------------------------------------
# P-05: lang="en" → first section title is "Introduction"
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_planner_en_default_sections(mock_llm, mock_prompts, base_state):
    """P-05: en fallback outline uses English section titles."""
    state = cast(AgentState, {**base_state, "language": "en", "papers": [], "status": "planning"})
    with patch(
        "seele_scholar_agent.nodes.planner.invoke_with_retry",
        side_effect=Exception("fail"),
    ):
        node = PlannerNode(llm=mock_llm, prompts=mock_prompts)
        result = await node.plan(state)

    assert result["sections"][0].title == "Introduction"


# ---------------------------------------------------------------------------
# P-06: lang="ja" → first section title is "序論"
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_planner_ja_default_sections(mock_llm, mock_prompts, base_state):
    """P-06: ja fallback outline uses Japanese section titles."""
    state = cast(AgentState, {**base_state, "language": "ja", "papers": [], "status": "planning"})
    with patch(
        "seele_scholar_agent.nodes.planner.invoke_with_retry",
        side_effect=Exception("fail"),
    ):
        node = PlannerNode(llm=mock_llm, prompts=mock_prompts)
        result = await node.plan(state)

    assert result["sections"][0].title == "序論"


# ---------------------------------------------------------------------------
# P-07: papers=[] → papers_summary uses "no_papers_found" i18n string
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_planner_empty_papers_summary(mock_llm, mock_prompts, base_state):
    """P-07: Empty papers list yields a valid outline (no crash)."""
    state = cast(AgentState, {**base_state, "language": "zh", "papers": [], "status": "planning"})
    result_data = _make_llm_result()
    with patch(
        "seele_scholar_agent.nodes.planner.invoke_with_retry",
        new_callable=AsyncMock,
        return_value=result_data,
    ):
        node = PlannerNode(llm=mock_llm, prompts=mock_prompts)
        result = await node.plan(state)

    # Should still produce an outline without crashing
    assert result["outline"] is not None


# ---------------------------------------------------------------------------
# P-08: papers > 15 → only first 15 passed to LLM
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_planner_truncates_papers_to_15(mock_llm, mock_prompts, base_state):
    """P-08: Only first 15 papers are included in the prompt."""
    many_papers = [
        PaperMetadata(
            paper_id=f"id:{i}",
            title=f"Paper {i}",
            authors=["Author"],
            abstract=f"Abstract {i}",
            source="arxiv",
        )
        for i in range(20)
    ]
    state = cast(AgentState, {**base_state, "papers": many_papers, "status": "planning"})

    captured_input: dict = {}

    async def capture_invoke(chain, input_data):  # type: ignore[override]
        captured_input.update(input_data)
        return _make_llm_result()

    with patch("seele_scholar_agent.nodes.planner.invoke_with_retry", side_effect=capture_invoke):
        node = PlannerNode(llm=mock_llm, prompts=mock_prompts)
        await node.plan(state)

    # papers_summary should contain exactly 15 paper titles
    papers_summary = captured_input["papers_summary"]
    assert papers_summary.count("Paper ") == 15


# ---------------------------------------------------------------------------
# P-09: Return dict contains outline, sections, current_section_index=0, status="waiting_human"
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_planner_return_keys_and_status(mock_llm, mock_prompts, state_with_papers):
    """P-09: Return dict has correct keys and status."""
    with patch(
        "seele_scholar_agent.nodes.planner.invoke_with_retry",
        new_callable=AsyncMock,
        return_value=_make_llm_result(),
    ):
        node = PlannerNode(llm=mock_llm, prompts=mock_prompts)
        result = await node.plan(state_with_papers)

    assert "outline" in result
    assert "sections" in result
    assert result["current_section_index"] == 0
    assert result["status"] == "waiting_human"


# ---------------------------------------------------------------------------
# P-10: section_id uses "section_{i}" format
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_planner_section_ids(mock_llm, mock_prompts, state_with_papers):
    """P-10: SectionDraft IDs follow 'section_{i}' pattern."""
    with patch(
        "seele_scholar_agent.nodes.planner.invoke_with_retry",
        new_callable=AsyncMock,
        return_value=_make_llm_result(
            sections=[
                {"title": "A", "description": "a", "order": 1, "key_points": []},
                {"title": "B", "description": "b", "order": 2, "key_points": []},
                {"title": "C", "description": "c", "order": 3, "key_points": []},
            ]
        ),
    ):
        node = PlannerNode(llm=mock_llm, prompts=mock_prompts)
        result = await node.plan(state_with_papers)

    ids = [s.section_id for s in result["sections"]]
    assert ids == ["section_0", "section_1", "section_2"]


# ---------------------------------------------------------------------------
# P-11: suggested_figures from LLM result → written into SectionOutline
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_planner_suggested_figures_parsed(mock_llm, mock_prompts, state_with_papers):
    """P-11: suggested_figures from LLM output are stored in SectionOutline."""
    result_data = _make_llm_result(
        sections=[
            {
                "title": "Introduction",
                "description": "Intro",
                "order": 1,
                "key_points": [],
                "suggested_figures": [
                    "Bar chart comparing model accuracy",
                    "Timeline of LLM releases",
                ],
            },
            {
                "title": "Conclusion",
                "description": "Summary",
                "order": 2,
                "key_points": [],
                "suggested_figures": [],
            },
        ]
    )
    with patch(
        "seele_scholar_agent.nodes.planner.invoke_with_retry",
        new_callable=AsyncMock,
        return_value=result_data,
    ):
        node = PlannerNode(llm=mock_llm, prompts=mock_prompts)
        result = await node.plan(state_with_papers)

    intro_outline = result["outline"].sections[0]
    assert intro_outline.suggested_figures == [
        "Bar chart comparing model accuracy",
        "Timeline of LLM releases",
    ]
    conclusion_outline = result["outline"].sections[1]
    assert conclusion_outline.suggested_figures == []


# ---------------------------------------------------------------------------
# P-12: section missing suggested_figures key → defaults to []
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_planner_suggested_figures_missing_key_defaults_empty(
    mock_llm, mock_prompts, state_with_papers
):
    """P-12: Section without suggested_figures key defaults to empty list."""
    result_data = _make_llm_result(
        sections=[
            {"title": "Introduction", "description": "Intro", "order": 1, "key_points": []},
        ]
    )
    with patch(
        "seele_scholar_agent.nodes.planner.invoke_with_retry",
        new_callable=AsyncMock,
        return_value=result_data,
    ):
        node = PlannerNode(llm=mock_llm, prompts=mock_prompts)
        result = await node.plan(state_with_papers)

    assert result["outline"].sections[0].suggested_figures == []


# ---------------------------------------------------------------------------
# P-13: suggested_figures preserved across multiple sections independently
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_planner_suggested_figures_per_section_independent(
    mock_llm, mock_prompts, state_with_papers
):
    """P-13: Each section carries its own suggested_figures independently."""
    result_data = _make_llm_result(
        sections=[
            {
                "title": "Methods",
                "description": "Methodology",
                "order": 1,
                "key_points": [],
                "suggested_figures": ["Flowchart of pipeline"],
            },
            {
                "title": "Results",
                "description": "Experiments",
                "order": 2,
                "key_points": [],
                "suggested_figures": ["Table: BLEU scores", "Figure: loss curves"],
            },
            {
                "title": "Discussion",
                "description": "Analysis",
                "order": 3,
                "key_points": [],
                "suggested_figures": [],
            },
        ]
    )
    with patch(
        "seele_scholar_agent.nodes.planner.invoke_with_retry",
        new_callable=AsyncMock,
        return_value=result_data,
    ):
        node = PlannerNode(llm=mock_llm, prompts=mock_prompts)
        result = await node.plan(state_with_papers)

    sections = result["outline"].sections
    assert sections[0].suggested_figures == ["Flowchart of pipeline"]
    assert sections[1].suggested_figures == ["Table: BLEU scores", "Figure: loss curves"]
    assert sections[2].suggested_figures == []


# ---------------------------------------------------------------------------
# P-14: default outline fallback → suggested_figures is [] for all sections
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_planner_default_outline_has_empty_suggested_figures(
    mock_llm, mock_prompts, state_with_papers
):
    """P-14: Fallback outline sections have empty suggested_figures."""
    with patch(
        "seele_scholar_agent.nodes.planner.invoke_with_retry",
        side_effect=Exception("timeout"),
    ):
        node = PlannerNode(llm=mock_llm, prompts=mock_prompts)
        result = await node.plan(state_with_papers)

    for section in result["outline"].sections:
        assert section.suggested_figures == []


# ---------------------------------------------------------------------------
# P-15: Planner preserves paper type, structure pattern, and section planning fields
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_planner_preserves_structure_and_section_strategy(
    mock_llm, mock_prompts, state_with_papers
):
    """P-15: Rich outline fields are parsed into OutlineStructure and SectionOutline."""
    result_data = _make_llm_result(
        sections=[
            {
                "title": "Thematic Foundations",
                "description": "Define the main research themes.",
                "order": 1,
                "purpose": "Establish the conceptual frame for the review.",
                "content_summary": (
                    "This section defines the core LLM concepts and explains why "
                    "they matter for the survey."
                ),
                "target_words": "850",
                "key_points": ["motivation"],
                "target_claims": ["LLM evaluation requires both capability and risk dimensions."],
                "key_sources": ["[1] Attention Is All You Need"],
                "citation_plan": ["Use [1] to ground Transformer architecture."],
                "evidence_gaps": ["Need a recent evaluation survey."],
                "transition_to_next": "Moves from definitions to prior evaluation frameworks.",
            }
        ],
    )
    result_data.update(
        {
            "paper_type": "literature_review",
            "structure_pattern": "thematic_review",
            "rationale": (
                "The topic is better handled as a thematic synthesis than as an "
                "experiment."
            ),
        }
    )

    with patch(
        "seele_scholar_agent.nodes.planner.invoke_with_retry",
        new_callable=AsyncMock,
        return_value=result_data,
    ):
        node = PlannerNode(llm=mock_llm, prompts=mock_prompts)
        result = await node.plan(
            cast(
                AgentState,
                {
                    **state_with_papers,
                    "paper_type": "literature_review",
                    "structure_pattern": "thematic_review",
                },
            )
        )

    outline = result["outline"]
    section = outline.sections[0]
    assert outline.paper_type == "literature_review"
    assert outline.structure_pattern == "thematic_review"
    assert outline.rationale.startswith("The topic")
    assert section.purpose == "Establish the conceptual frame for the review."
    assert section.content_summary.startswith("This section defines")
    assert section.target_words == 850
    assert section.target_claims == [
        "LLM evaluation requires both capability and risk dimensions."
    ]
    assert section.key_sources == ["[1] Attention Is All You Need"]
    assert section.evidence_gaps == ["Need a recent evaluation survey."]
    assert section.transition_to_next.startswith("Moves from")


# ---------------------------------------------------------------------------
# P-16: Section strategy fields are mirrored into evidence_map by default
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_planner_builds_evidence_map_from_section_fields(
    mock_llm, mock_prompts, state_with_papers
):
    """P-16: evidence_map is derived from section-level evidence planning."""
    result_data = _make_llm_result(
        sections=[
            {
                "title": "Related Work",
                "description": "Prior evaluation work.",
                "order": 1,
                "key_points": [],
                "target_claims": ["Existing benchmarks under-cover multilingual safety."],
                "key_sources": ["[2] BERT"],
                "citation_plan": ["Use [2] as a representative pretraining baseline."],
                "evidence_gaps": ["Need multilingual benchmark sources."],
            }
        ]
    )

    with patch(
        "seele_scholar_agent.nodes.planner.invoke_with_retry",
        new_callable=AsyncMock,
        return_value=result_data,
    ):
        node = PlannerNode(llm=mock_llm, prompts=mock_prompts)
        result = await node.plan(state_with_papers)

    evidence_plan = result["outline"].evidence_map[0]
    assert evidence_plan.section_title == "Related Work"
    assert evidence_plan.target_claims == [
        "Existing benchmarks under-cover multilingual safety."
    ]
    assert evidence_plan.key_sources == ["[2] BERT"]
    assert evidence_plan.evidence_gaps == ["Need multilingual benchmark sources."]


# ---------------------------------------------------------------------------
# P-17: Planner input uses paper_summaries and requested structure controls
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_planner_prompt_uses_paper_summaries_and_structure_controls(
    mock_llm, mock_prompts, state_with_papers
):
    """P-17: Planner receives richer literature summaries and structure hints."""
    captured_input: dict = {}

    async def capture_invoke(chain, input_data):  # type: ignore[override]
        captured_input.update(input_data)
        return _make_llm_result()

    state = cast(
        AgentState,
        {
            **state_with_papers,
            "paper_summaries": ["[1] Compact summary with key contribution."],
            "paper_type": "theoretical",
            "structure_pattern": "theoretical_analysis",
            "target_word_count": 6000,
        },
    )

    with patch("seele_scholar_agent.nodes.planner.invoke_with_retry", side_effect=capture_invoke):
        node = PlannerNode(llm=mock_llm, prompts=mock_prompts)
        await node.plan(state)

    assert captured_input["papers_summary"] == "[1] Compact summary with key contribution."
    assert captured_input["paper_type"] == "theoretical"
    assert captured_input["structure_pattern"] == "theoretical_analysis"
    assert captured_input["target_word_count"] == "6000"
