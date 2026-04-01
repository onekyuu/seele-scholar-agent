"""Unit tests for ReferenceGeneratorNode — REF-01 through REF-06."""

from typing import cast
from unittest.mock import AsyncMock

import pytest

from seele_scholar_agent.nodes.reference_generator import ReferenceGeneratorNode
from seele_scholar_agent.state import AgentState, PaperMetadata, SectionDraft


def _make_section(content: str, section_id: str = "s0", title: str = "Intro") -> SectionDraft:
    return SectionDraft(
        section_id=section_id,
        title=title,
        description="",
        order_index=1,
        content=content,
        status="approved",
    )


# ---------------------------------------------------------------------------
# REF-01: Normal — sections cite [1] and [2] → ReferenceEntry for each
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_reference_generator_cited_papers_returned(base_state, sample_papers):
    sections = [_make_section("See [1] and [2] for details.")]
    state = cast(AgentState, {**base_state, "papers": sample_papers, "sections": sections})

    node = ReferenceGeneratorNode()
    result = await node.generate(state)

    refs = result["references"]
    assert len(refs) == 2
    numbers = {r.number for r in refs}
    assert numbers == {1, 2}
    assert result["status"] == "completed"


# ---------------------------------------------------------------------------
# REF-02: No citation markers in sections → full reference list generated
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_reference_generator_no_citations_generates_full_list(base_state, sample_papers):
    sections = [_make_section("This section has no citation markers.")]
    state = cast(AgentState, {**base_state, "papers": sample_papers, "sections": sections})

    node = ReferenceGeneratorNode()
    result = await node.generate(state)

    refs = result["references"]
    assert len(refs) == len(sample_papers)
    assert result["status"] == "completed"


# ---------------------------------------------------------------------------
# REF-03: Citation number out of range → skipped, no error raised
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_reference_generator_out_of_range_citation_skipped(base_state, sample_papers):
    # sample_papers has 3 entries; [99] is out of range
    sections = [_make_section("See [1] and [99].")]
    state = cast(AgentState, {**base_state, "papers": sample_papers, "sections": sections})

    node = ReferenceGeneratorNode()
    result = await node.generate(state)

    refs = result["references"]
    numbers = {r.number for r in refs}
    assert 99 not in numbers
    assert 1 in numbers
    assert result["status"] == "completed"


# ---------------------------------------------------------------------------
# REF-04: papers=[] → references=[], status="completed"
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_reference_generator_empty_papers_returns_empty(base_state):
    sections = [_make_section("See [1].")]
    state = cast(AgentState, {**base_state, "papers": [], "sections": sections})

    node = ReferenceGeneratorNode()
    result = await node.generate(state)

    assert result["references"] == []
    assert result["status"] == "completed"


# ---------------------------------------------------------------------------
# REF-05: formatted field contains number, authors, and title
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_reference_generator_formatted_field_structure(base_state, sample_papers):
    sections = [_make_section("See [1].")]
    state = cast(AgentState, {**base_state, "papers": sample_papers, "sections": sections})

    node = ReferenceGeneratorNode()
    result = await node.generate(state)

    ref = result["references"][0]
    assert ref.number == 1
    assert "[1]" in ref.formatted
    assert "Vaswani" in ref.formatted
    assert "Attention Is All You Need" in ref.formatted


# ---------------------------------------------------------------------------
# REF-06: Multiple sections — citation numbers deduplicated across all content
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_reference_generator_deduplicates_citations_across_sections(
    base_state, sample_papers
):
    sections = [
        _make_section("See [1] here.", section_id="s0", title="Section A"),
        _make_section("Also [1] and [2].", section_id="s1", title="Section B"),
    ]
    state = cast(AgentState, {**base_state, "papers": sample_papers, "sections": sections})

    node = ReferenceGeneratorNode()
    result = await node.generate(state)

    refs = result["references"]
    numbers = [r.number for r in refs]
    # [1] should appear exactly once even though cited in two sections
    assert numbers.count(1) == 1
    assert set(numbers) == {1, 2}
