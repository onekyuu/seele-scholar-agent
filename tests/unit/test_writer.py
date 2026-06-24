"""Unit tests for WriterNode — W-01 through W-24."""

import re
from typing import cast
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import AIMessage
from seele_scholar_agent.nodes.writer import WriterNode
from seele_scholar_agent.state import (
    AgentState,
    DocumentChunk,
    EvidencePacket,
    MaterialRegistry,
    MaterialRegistryEntry,
    OutlineStructure,
    PaperMetadata,
    SectionDraft,
    SectionOutline,
)

# ---------------------------------------------------------------------------
# W-01: Normal write → sections[0].content filled, status="reviewing"
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_writer_normal_write(mock_llm, mock_prompts, state_with_outline):
    captured_input: dict = {}

    async def capture_invoke(chain, input_data):  # type: ignore[override]
        captured_input.update(input_data)
        return AIMessage(content="This is the introduction content.")

    with patch("seele_scholar_agent.nodes.writer.invoke_with_retry", side_effect=capture_invoke):
        node = WriterNode(llm=mock_llm, prompts=mock_prompts)
        result = await node.write(state_with_outline)

    assert result["sections"][0].content == "This is the introduction content."
    assert result["sections"][0].status == "review"
    assert result["status"] == "reviewing"
    assert "Writing locale: zh-CN" in captured_input["style_guidance"]


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


# ---------------------------------------------------------------------------
# W-18: _build_suggested_figures — outline=None → returns "无"
# ---------------------------------------------------------------------------


def test_writer_build_suggested_figures_no_outline(mock_llm, mock_prompts, base_state):
    node = WriterNode(llm=mock_llm, prompts=mock_prompts)
    section = SectionDraft(section_id="s0", title="Introduction", description="", order_index=1)
    state = cast(AgentState, {**base_state, "outline": None})

    result = node._build_suggested_figures(section, state)

    assert result == "无"


# ---------------------------------------------------------------------------
# W-19: _build_suggested_figures — section title matched → figures returned
# ---------------------------------------------------------------------------


def test_writer_build_suggested_figures_matched(mock_llm, mock_prompts, state_with_outline):
    from seele_scholar_agent.state import OutlineStructure, SectionOutline

    outline = OutlineStructure(
        title="Survey",
        abstract="",
        sections=[
            SectionOutline(
                title="Introduction",
                description="",
                order=1,
                suggested_figures=["Bar chart of accuracy", "Timeline of models"],
            ),
            SectionOutline(title="Conclusion", description="", order=2, suggested_figures=[]),
        ],
        keywords=[],
    )
    state = cast(AgentState, {**state_with_outline, "outline": outline})
    section = SectionDraft(section_id="s0", title="Introduction", description="", order_index=1)
    node = WriterNode(llm=mock_llm, prompts=mock_prompts)

    result = node._build_suggested_figures(section, state)

    assert "Bar chart of accuracy" in result
    assert "Timeline of models" in result
    assert result.startswith("- ")


# ---------------------------------------------------------------------------
# W-20: _build_suggested_figures — section title not in outline → returns "无"
# ---------------------------------------------------------------------------


def test_writer_build_suggested_figures_no_match(mock_llm, mock_prompts, state_with_outline):
    section = SectionDraft(
        section_id="s0", title="NonExistentSection", description="", order_index=1
    )
    node = WriterNode(llm=mock_llm, prompts=mock_prompts)

    result = node._build_suggested_figures(section, state_with_outline)

    assert result == "无"


# ---------------------------------------------------------------------------
# W-21: _build_suggested_figures — matched section has empty list → returns "无"
# ---------------------------------------------------------------------------


def test_writer_build_suggested_figures_empty_list(mock_llm, mock_prompts, state_with_outline):
    from seele_scholar_agent.state import OutlineStructure, SectionOutline

    outline = OutlineStructure(
        title="Survey",
        abstract="",
        sections=[
            SectionOutline(title="Introduction", description="", order=1, suggested_figures=[]),
        ],
        keywords=[],
    )
    state = cast(AgentState, {**state_with_outline, "outline": outline})
    section = SectionDraft(section_id="s0", title="Introduction", description="", order_index=1)
    node = WriterNode(llm=mock_llm, prompts=mock_prompts)

    result = node._build_suggested_figures(section, state)

    assert result == "无"


# ---------------------------------------------------------------------------
# W-22: _build_rag_context — carries chunk_id in [chunk_id:xxx] format
# ---------------------------------------------------------------------------


def test_writer_build_rag_context_includes_chunk_id(mock_llm, mock_prompts):
    chunks = [
        DocumentChunk(chunk_id="abc123", content="Transformer paper content.", source="arxiv"),
        DocumentChunk(chunk_id="def456", content="BERT paper content.", source="s2"),
    ]
    node = WriterNode(llm=mock_llm, prompts=mock_prompts)
    result = node._build_rag_context(chunks)

    assert "[chunk_id:abc123]" in result
    assert "Transformer paper content." in result
    assert "[chunk_id:def456]" in result
    assert "BERT paper content." in result


# ---------------------------------------------------------------------------
# W-23: _build_rag_context — empty input → returns "无"
# ---------------------------------------------------------------------------


def test_writer_build_rag_context_empty(mock_llm, mock_prompts):
    node = WriterNode(llm=mock_llm, prompts=mock_prompts)
    assert node._build_rag_context([]) == "无"
    assert node._build_rag_context(None) == "无"


# ---------------------------------------------------------------------------
# W-24: _clean_content — preserves figure/table placeholders intact
# ---------------------------------------------------------------------------


def test_writer_clean_content_preserves_figure_placeholder(mock_llm, mock_prompts):
    node = WriterNode(llm=mock_llm, prompts=mock_prompts)
    raw = (
        "Here is the introduction.\n"
        "{{FIGURE: Bar chart of accuracy | chunks:[abc123,def456]}}\n"
        "{{TABLE: Comparison of methods | chunks:[xyz789]}}\n"
        "More text follows."
    )
    cleaned = node._clean_content(raw)

    assert "{{FIGURE: Bar chart of accuracy | chunks:[abc123,def456]}}" in cleaned
    assert "{{TABLE: Comparison of methods | chunks:[xyz789]}}" in cleaned
    assert "More text follows." in cleaned


def test_writer_chunks_to_evidence_packets(mock_llm, mock_prompts):
    chunk = DocumentChunk(
        chunk_id="c1",
        content="Quoted evidence about Transformer architecture.",
        source="arxiv",
        metadata={
            "paper_id": "arxiv:2301.00001",
            "title": "Attention Is All You Need",
            "authors": ["Vaswani", "Shazeer"],
            "year": 2017,
            "page": "3",
            "section": "Model Architecture",
            "relevance_score": 0.9,
            "why_relevant": "Supports the architecture claim.",
            "quote": "The Transformer architecture uses attention mechanisms.",
        },
    )
    node = WriterNode(llm=mock_llm, prompts=mock_prompts)

    packets = node._chunks_to_evidence_packets(
        [chunk], section_title="Introduction", why_relevant="query"
    )

    assert packets[0].chunk_id == "c1"
    assert packets[0].source_paper_id == "arxiv:2301.00001"
    assert packets[0].title == "Attention Is All You Need"
    assert packets[0].authors == ["Vaswani", "Shazeer"]
    assert packets[0].quote == "The Transformer architecture uses attention mechanisms."


@pytest.mark.asyncio
async def test_writer_returns_evidence_packets_and_claim_bindings(
    mock_llm, mock_prompts, state_with_outline, sample_papers
):
    chunk = DocumentChunk(
        chunk_id="c1",
        content="The Transformer architecture uses attention mechanisms.",
        source="arxiv",
        metadata={
            "paper_id": sample_papers[0].paper_id,
            "title": sample_papers[0].title,
            "quote": "The Transformer architecture uses attention mechanisms.",
            "relevance_score": 0.9,
        },
    )

    async def rag_retriever(_query: str) -> list[DocumentChunk]:
        return [chunk]

    with patch(
        "seele_scholar_agent.nodes.writer.invoke_with_retry",
        new_callable=AsyncMock,
        return_value=AIMessage(content="The Transformer architecture uses attention [1]."),
    ):
        node = WriterNode(llm=mock_llm, prompts=mock_prompts, rag_retriever=rag_retriever)
        result = await node.write({**state_with_outline, "papers": sample_papers})

    assert result["evidence_packets"][0].chunk_id == "c1"
    binding = result["claim_evidence_bindings"][0]
    assert binding.citation_number == 1
    assert binding.chunk_id == "c1"
    assert binding.verdict == "supported"


def test_citation_binder_extracts_multiple_cited_factual_claims(mock_llm, mock_prompts):
    node = WriterNode(llm=mock_llm, prompts=mock_prompts)

    claims = node.citation_binder._extract_cited_claims(
        "Prior work shows attention improves sequence modeling [1]. "
        "This section explains the background. "
        "Experiments indicate better scaling [2]."
    )

    assert claims == [
        "Prior work shows attention improves sequence modeling [1].",
        "Experiments indicate better scaling [2].",
    ]


def test_writer_build_rag_context_formats_evidence_packet(mock_llm, mock_prompts):
    packet = EvidencePacket(
        chunk_id="packet-1",
        title="Evidence Paper",
        authors=["Author A"],
        year=2024,
        page="7",
        section="Findings",
        relevance_score=0.8,
        why_relevant="Supports a cited claim.",
        quote="Evidence quote.",
    )
    node = WriterNode(llm=mock_llm, prompts=mock_prompts)

    result = node._build_rag_context([packet])

    assert "[chunk_id:packet-1]" in result
    assert "title: Evidence Paper" in result
    assert "quote: Evidence quote." in result


def test_writer_numbered_papers_include_material_policy(mock_llm, mock_prompts, sample_papers):
    registry = MaterialRegistry(
        entries=[MaterialRegistryEntry(paper_id=sample_papers[0].paper_id, required=True)]
    )
    node = WriterNode(llm=mock_llm, prompts=mock_prompts)

    result = node._build_numbered_papers(
        [sample_papers[0]], {"material_registry": registry}  # type: ignore[arg-type]
    )

    assert "required_by_user=true" in result


# ---------------------------------------------------------------------------
# Figure placeholder regex tests
# ---------------------------------------------------------------------------

FIGURE_PATTERN = re.compile(r"\{\{(FIGURE|TABLE): (.+?) \| chunks:\[([^\]]*)\]\}\}")


def test_figure_pattern_matches_figure_with_chunks():
    content = "{{FIGURE: Bar chart comparing Top-1 accuracy | chunks:[abc123,def456]}}"
    matches = FIGURE_PATTERN.findall(content)

    assert len(matches) == 1
    fig_type, description, chunks_str = matches[0]
    assert fig_type == "FIGURE"
    assert description == "Bar chart comparing Top-1 accuracy"
    assert chunks_str == "abc123,def456"


def test_figure_pattern_matches_table_with_single_chunk():
    content = "{{TABLE: Comparison of model parameters | chunks:[xyz789]}}"
    matches = FIGURE_PATTERN.findall(content)

    assert len(matches) == 1
    fig_type, description, chunks_str = matches[0]
    assert fig_type == "TABLE"
    assert description == "Comparison of model parameters"
    chunk_ids = [c.strip() for c in chunks_str.split(",") if c.strip()]
    assert chunk_ids == ["xyz789"]


def test_figure_pattern_matches_empty_chunks():
    content = "{{FIGURE: Conceptual diagram of architecture | chunks:[]}}"
    matches = FIGURE_PATTERN.findall(content)

    assert len(matches) == 1
    _, _, chunks_str = matches[0]
    chunk_ids = [c.strip() for c in chunks_str.split(",") if c.strip()]
    assert chunk_ids == []


def test_figure_pattern_matches_multiple_placeholders():
    content = (
        "Some text.\n"
        "{{FIGURE: Accuracy bar chart | chunks:[c1,c2]}}\n"
        "More text.\n"
        "{{TABLE: Method comparison | chunks:[c3]}}\n"
        "End."
    )
    matches = FIGURE_PATTERN.findall(content)

    assert len(matches) == 2
    assert matches[0][0] == "FIGURE"
    assert matches[1][0] == "TABLE"


def test_figure_pattern_does_not_match_malformed_placeholder():
    malformed_cases = [
        "{{FIGURE: missing pipe and chunks}}",
        "{{GRAPH: wrong type | chunks:[c1]}}",
        "{FIGURE: single brace | chunks:[c1]}",
    ]
    for case in malformed_cases:
        matches = FIGURE_PATTERN.findall(case)
        assert matches == [], f"Should not match: {case}"


@pytest.mark.asyncio
async def test_proposal_schedule_rewrite_prompt_requires_two_year_timeline(
    mock_llm, mock_prompts, base_state
):
    captured_input: dict = {}

    async def capture_invoke(chain, input_data):  # type: ignore[override]
        captured_input.update(input_data)
        return AIMessage(
            content=(
                "1年次前期に先行研究を整理する。1年次後期にプロトタイプを開発する。"
                "2年次前期に評価実験を行う。2年次後期に論文を執筆する。"
            )
        )

    section = SectionDraft(
        section_id="schedule",
        title="研究計画・スケジュール",
        description="研究計画書全体 2000-2500 字の第5章。",
        order_index=5,
        content="1年次前期に文献調査を行う。",
        status="writing",
        revision_count=1,
        review_comments=["2年次の研究計画が欠落している。"],
    )
    outline = OutlineStructure(
        title="研究計画書",
        abstract="",
        sections=[
            SectionOutline(
                title=section.title,
                description=section.description,
                order=5,
            )
        ],
        paper_type="research_proposal",
    )
    state = cast(
        AgentState,
        {
            **base_state,
            "document_type": "research_proposal",
            "language": "ja",
            "topic": "ゲーム音響における感情曲線設計",
            "outline": outline,
            "sections": [section],
            "status": "writing",
        },
    )

    with patch("seele_scholar_agent.nodes.writer.invoke_with_retry", side_effect=capture_invoke):
        node = WriterNode(llm=mock_llm, prompts=mock_prompts)
        result = await node.write(state)

    rendered_user_prompt = node.proposal_revision_prompt.format_messages(**captured_input)[
        1
    ].content
    assert result["writer_diagnostics"]["revision_mode"] is True
    assert result["writer_diagnostics"]["proposal_profile"] is True
    assert captured_input["current_content"] == "1年次前期に文献調査を行う。"
    assert "2年次の研究計画が欠落" in captured_input["review_comments"]
    for phase in ("1年次前期", "1年次後期", "2年次前期", "2年次後期"):
        assert phase in rendered_user_prompt


def test_proposal_writer_budget_description_uses_budget_as_hard_constraint(
    mock_llm, mock_prompts
):
    section = SectionDraft(
        section_id="s1",
        title="研究背景",
        description="研究計画書全文 2000-2500 字のうち、本章は約400字。",
        order_index=1,
    )
    node = WriterNode(llm=mock_llm, prompts=mock_prompts)

    description = node._build_section_description(section)

    assert "only hard length limit" in description
    assert "Default length target: 200-500" not in description


def test_citation_binder_exposes_unverified_diagnostics(mock_llm, mock_prompts):
    section = SectionDraft(section_id="s1", title="背景", order_index=1)
    paper = PaperMetadata(
        paper_id="p1",
        title="Evidence Paper",
        authors=["Author"],
        abstract="Abstract.",
        source="openalex",
    )
    node = WriterNode(llm=mock_llm, prompts=mock_prompts)

    bindings = node.citation_binder.bind(
        section,
        "先行研究は重要な背景を示している [1]。",
        [paper],
        [],
    )

    assert bindings[0].verdict == "unverified"
    assert bindings[0].diagnostics["evidence_packet_count"] == 0
    assert bindings[0].diagnostics["candidate_count"] == 0
    assert bindings[0].diagnostics["source_paper_id"] == "p1"


@pytest.mark.asyncio
async def test_academic_revision_uses_revision_prompt(
    mock_llm, mock_prompts, state_with_outline
):
    captured_input: dict = {}

    async def capture_invoke(chain, input_data):  # type: ignore[override]
        captured_input.update(input_data)
        return AIMessage(content="Revised academic content with a valid citation [1].")

    sections = list(state_with_outline["sections"])
    sections[0] = sections[0].model_copy(
        update={
            "content": "Old content makes an unsupported claim.",
            "status": "writing",
            "revision_count": 1,
            "review_comments": [
                "Issue 1: missing_citation: unsupported claim.",
                "Suggestion: add a valid citation or remove the claim.",
            ],
        }
    )
    state = cast(
        AgentState,
        {
            **state_with_outline,
            "document_type": "academic_paper",
            "sections": sections,
            "current_section_index": 0,
        },
    )

    with patch("seele_scholar_agent.nodes.writer.invoke_with_retry", side_effect=capture_invoke):
        node = WriterNode(llm=mock_llm, prompts=mock_prompts)
        result = await node.write(state)

    rendered_user_prompt = node.academic_revision_prompt.format_messages(**captured_input)[
        1
    ].content
    assert result["writer_diagnostics"]["revision_mode"] is True
    assert result["writer_diagnostics"]["writer_mode"] == "academic_revision"
    assert result["writer_diagnostics"]["proposal_profile"] is False
    assert captured_input["current_content"] == "Old content makes an unsupported claim."
    assert "missing_citation" in captured_input["review_comments"]
    assert "unsupported claim" in rendered_user_prompt
    assert "有效 [N] 引用" in rendered_user_prompt


@pytest.mark.asyncio
async def test_academic_initial_draft_keeps_regular_writer_prompt(
    mock_llm, mock_prompts, state_with_outline
):
    with patch(
        "seele_scholar_agent.nodes.writer.invoke_with_retry",
        new_callable=AsyncMock,
        return_value=AIMessage(content="Initial academic draft."),
    ):
        node = WriterNode(llm=mock_llm, prompts=mock_prompts)
        result = await node.write({**state_with_outline, "document_type": "academic_paper"})

    assert result["writer_diagnostics"]["revision_mode"] is False
    assert result["writer_diagnostics"]["writer_mode"] == "draft"
    assert result["writer_diagnostics"]["proposal_profile"] is False


@pytest.mark.asyncio
async def test_proposal_initial_draft_uses_proposal_prompt(
    mock_llm, mock_prompts, base_state
):
    captured_input: dict = {}

    async def capture_invoke(chain, input_data):  # type: ignore[override]
        captured_input.update(input_data)
        return AIMessage(content="研究計画書の初稿本文。")

    section = SectionDraft(
        section_id="s1",
        title="研究目的・研究課題",
        description="研究目的を約450字で述べる。",
        order_index=1,
    )
    outline = OutlineStructure(
        title="研究計画書",
        abstract="",
        sections=[
            SectionOutline(
                title=section.title,
                description=section.description,
                order=1,
                target_words=450,
            )
        ],
        paper_type="research_proposal",
        structure_pattern="research_proposal",
    )
    state = cast(
        AgentState,
        {
            **base_state,
            "document_type": "research_proposal",
            "language": "ja",
            "outline": outline,
            "sections": [section],
            "status": "writing",
        },
    )

    with patch("seele_scholar_agent.nodes.writer.invoke_with_retry", side_effect=capture_invoke):
        node = WriterNode(llm=mock_llm, prompts=mock_prompts)
        result = await node.write(state)

    rendered_user_prompt = node.proposal_draft_prompt.format_messages(**captured_input)[
        1
    ].content
    assert result["writer_diagnostics"]["writer_mode"] == "proposal_draft"
    assert result["writer_diagnostics"]["revision_mode"] is False
    assert "日本大学院研究計画書" in rendered_user_prompt
    assert "申请者自身的问题意识" in rendered_user_prompt
