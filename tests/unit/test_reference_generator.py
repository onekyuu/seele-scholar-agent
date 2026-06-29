from typing import cast
from unittest.mock import AsyncMock, patch

import pytest
from seele_scholar_agent.citation import CitationSource, SourceQuality
from seele_scholar_agent.nodes.reference_generator import ReferenceGeneratorNode
from seele_scholar_agent.state import AgentState, PaperMetadata, SectionDraft
from seele_scholar_agent.tools.crossref import CrossRefMetadata, extract_doi_from_url


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

    with patch(
        "seele_scholar_agent.nodes.reference_generator.fetch_metadata",
        new_callable=AsyncMock,
        return_value=None,
    ):
        node = ReferenceGeneratorNode()
        result = await node.generate(state)

    refs = result["references"]
    assert len(refs) == 2
    numbers = {r.number for r in refs}
    assert numbers == {1, 2}
    assert result["status"] == "completed"


@pytest.mark.asyncio
async def test_reference_generator_uses_citation_source_ids(base_state, sample_papers):
    source = CitationSource(
        citation_id=7,
        paper=sample_papers[0],
        stable_url="https://doi.org/10.48550/arxiv.1706.03762",
        doi="10.48550/arxiv.1706.03762",
        source_quality=SourceQuality(
            citable=True,
            metadata_verified=True,
            verification_source="openalex",
        ),
    )
    sections = [_make_section("See [7] for details.")]
    state = cast(
        AgentState,
        {
            **base_state,
            "papers": [],
            "citation_sources": [source],
            "sections": sections,
        },
    )

    with patch(
        "seele_scholar_agent.nodes.reference_generator.fetch_metadata",
        new_callable=AsyncMock,
        return_value=None,
    ):
        node = ReferenceGeneratorNode()
        result = await node.generate(state)

    refs = result["references"]
    assert len(refs) == 1
    assert refs[0].number == 7
    assert refs[0].paper_id == sample_papers[0].paper_id
    assert refs[0].doi == "10.48550/arxiv.1706.03762"
    assert refs[0].metadata_verified is True


# ---------------------------------------------------------------------------
# REF-02: No citation markers in sections → no references and a blocking quality issue
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_reference_generator_no_citations_returns_quality_issue(base_state, sample_papers):
    sections = [_make_section("This section has no citation markers.")]
    state = cast(AgentState, {**base_state, "papers": sample_papers, "sections": sections})

    with patch(
        "seele_scholar_agent.nodes.reference_generator.fetch_metadata",
        new_callable=AsyncMock,
        return_value=None,
    ):
        node = ReferenceGeneratorNode()
        result = await node.generate(state)

    assert result["references"] == []
    assert result["quality_issues"][0].code == "NO_INLINE_CITATIONS"
    assert result["quality_issues"][0].blocking is True
    assert result["status"] == "completed"


@pytest.mark.asyncio
async def test_reference_generator_proposal_no_citations_is_warning(base_state, sample_papers):
    sections = [_make_section("本研究では二年間の計画を実施する。")]
    state = cast(
        AgentState,
        {
            **base_state,
            "document_type": "research_proposal",
            "papers": sample_papers,
            "sections": sections,
        },
    )

    node = ReferenceGeneratorNode()
    result = await node.generate(state)

    assert result["references"] == []
    assert result["quality_issues"][0].code == "PROPOSAL_NO_INLINE_CITATIONS"
    assert result["quality_issues"][0].blocking is False
    assert result["quality_issues"][0].severity == "warning"


# ---------------------------------------------------------------------------
# REF-03: Citation number out of range → skipped, no error raised
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_reference_generator_out_of_range_citation_skipped(base_state, sample_papers):
    sections = [_make_section("See [1] and [99].")]
    state = cast(AgentState, {**base_state, "papers": sample_papers, "sections": sections})

    with patch(
        "seele_scholar_agent.nodes.reference_generator.fetch_metadata",
        new_callable=AsyncMock,
        return_value=None,
    ):
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

    with patch(
        "seele_scholar_agent.nodes.reference_generator.fetch_metadata",
        new_callable=AsyncMock,
        return_value=None,
    ):
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

    with patch(
        "seele_scholar_agent.nodes.reference_generator.fetch_metadata",
        new_callable=AsyncMock,
        return_value=None,
    ):
        node = ReferenceGeneratorNode()
        result = await node.generate(state)

    refs = result["references"]
    numbers = [r.number for r in refs]
    assert numbers.count(1) == 1
    assert set(numbers) == {1, 2}


# ---------------------------------------------------------------------------
# REF-07: CrossRef returns metadata → year/venue/doi populated from API
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_reference_generator_crossref_enriches_year_venue_doi(base_state, sample_papers):
    sections = [_make_section("See [1].")]
    state = cast(AgentState, {**base_state, "papers": sample_papers, "sections": sections})

    cr_meta = CrossRefMetadata(
        doi="10.48550/arXiv.1706.03762",
        year=2017,
        venue="Advances in Neural Information Processing Systems",
        authors=["Vaswani, Ashish", "Shazeer, Noam"],
    )

    with patch(
        "seele_scholar_agent.nodes.reference_generator.fetch_metadata",
        new_callable=AsyncMock,
        return_value=cr_meta,
    ):
        node = ReferenceGeneratorNode()
        result = await node.generate(state)

    ref = result["references"][0]
    assert ref.year == 2017
    assert ref.venue == "Advances in Neural Information Processing Systems"
    assert ref.doi == "10.48550/arXiv.1706.03762"
    assert "Vaswani, Ashish" in ref.authors
    assert ref.metadata_verified is True
    assert ref.verification_source == "crossref"


# ---------------------------------------------------------------------------
# REF-08: CrossRef returns no authors → fallback to PaperMetadata authors
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_reference_generator_crossref_empty_authors_falls_back(base_state, sample_papers):
    sections = [_make_section("See [1].")]
    state = cast(AgentState, {**base_state, "papers": sample_papers, "sections": sections})

    cr_meta = CrossRefMetadata(
        doi="10.48550/arXiv.1706.03762", year=2017, venue="NeurIPS", authors=[]
    )

    with patch(
        "seele_scholar_agent.nodes.reference_generator.fetch_metadata",
        new_callable=AsyncMock,
        return_value=cr_meta,
    ):
        node = ReferenceGeneratorNode()
        result = await node.generate(state)

    ref = result["references"][0]
    assert ref.authors == sample_papers[0].authors


@pytest.mark.asyncio
async def test_reference_generator_uses_paper_doi_for_crossref_lookup(base_state):
    paper = PaperMetadata(
        paper_id="user:1",
        title="Provided DOI Paper",
        authors=["Author A"],
        abstract="Abstract.",
        doi="10.1000/provided",
        source="user_library",
    )
    sections = [_make_section("See [1].")]
    state = cast(AgentState, {**base_state, "papers": [paper], "sections": sections})
    cr_meta = CrossRefMetadata(
        doi="10.1000/provided",
        year=2024,
        venue="Journal",
        authors=["Author A"],
    )

    with patch(
        "seele_scholar_agent.nodes.reference_generator.fetch_metadata",
        new_callable=AsyncMock,
        return_value=cr_meta,
    ) as fetch_metadata:
        node = ReferenceGeneratorNode()
        result = await node.generate(state)

    fetch_metadata.assert_awaited_once_with("10.1000/provided")
    ref = result["references"][0]
    assert ref.metadata_verified is True
    assert ref.verification_source == "crossref"


# ---------------------------------------------------------------------------
# REF-09: CrossRef API fails (returns None) → fallback to local year extraction
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_reference_generator_crossref_failure_falls_back_to_local(base_state):
    paper = PaperMetadata(
        paper_id="arxiv:2301.00001",
        title="Test Paper",
        authors=["Author A"],
        abstract="Published in 2021.",
        url="https://doi.org/10.1234/test",
        source="arxiv",
    )
    sections = [_make_section("See [1].")]
    state = cast(AgentState, {**base_state, "papers": [paper], "sections": sections})

    with patch(
        "seele_scholar_agent.nodes.reference_generator.fetch_metadata",
        new_callable=AsyncMock,
        return_value=None,
    ):
        node = ReferenceGeneratorNode()
        result = await node.generate(state)

    ref = result["references"][0]
    assert ref.year == 2021
    assert ref.venue is None
    assert ref.metadata_verified is False
    assert ref.verification_source == "local"


# ---------------------------------------------------------------------------
# REF-10: Paper with DOI URL → doi field populated from URL extraction fallback
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_reference_generator_doi_extracted_from_url_on_crossref_failure(base_state):
    paper = PaperMetadata(
        paper_id="oa:W001",
        title="OpenAlex Paper",
        authors=["Researcher"],
        abstract="Some abstract.",
        url="https://doi.org/10.1038/nature12373",
        source="openalex",
    )
    sections = [_make_section("See [1].")]
    state = cast(AgentState, {**base_state, "papers": [paper], "sections": sections})

    with patch(
        "seele_scholar_agent.nodes.reference_generator.fetch_metadata",
        new_callable=AsyncMock,
        return_value=None,
    ):
        node = ReferenceGeneratorNode()
        result = await node.generate(state)

    ref = result["references"][0]
    assert ref.doi == "10.1038/nature12373"
    assert ref.metadata_verified is True
    assert ref.verification_source == "openalex"


# ---------------------------------------------------------------------------
# REF-11: Paper with no URL → doi is None when CrossRef also returns None
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_reference_generator_no_url_doi_is_none(base_state):
    paper = PaperMetadata(
        paper_id="s2:xyz",
        title="No URL Paper",
        authors=["Unknown"],
        abstract="Abstract without URL.",
        source="semantic_scholar",
    )
    sections = [_make_section("See [1].")]
    state = cast(AgentState, {**base_state, "papers": [paper], "sections": sections})

    with patch(
        "seele_scholar_agent.nodes.reference_generator.fetch_metadata",
        new_callable=AsyncMock,
        return_value=None,
    ):
        node = ReferenceGeneratorNode()
        result = await node.generate(state)

    ref = result["references"][0]
    assert ref.doi is None


# ---------------------------------------------------------------------------
# REF-12: extract_doi_from_url — DOI URL, ArXiv URL, and non-DOI URL
# ---------------------------------------------------------------------------


def test_extract_doi_from_doi_org_url():
    assert extract_doi_from_url("https://doi.org/10.1038/nature12373") == "10.1038/nature12373"


def test_extract_doi_from_dx_doi_org_url():
    assert extract_doi_from_url("https://dx.doi.org/10.1000/xyz123") == "10.1000/xyz123"


def test_extract_doi_from_arxiv_url():
    doi = extract_doi_from_url("https://arxiv.org/abs/1706.03762")
    assert doi == "10.48550/arXiv.1706.03762"


def test_extract_doi_from_non_doi_url_returns_none():
    assert extract_doi_from_url("https://arxiv.org/pdf/1706.03762") is None


def test_extract_doi_from_empty_string_returns_none():
    assert extract_doi_from_url("") is None
