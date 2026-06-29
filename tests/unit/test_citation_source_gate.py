import pytest
from seele_scholar_agent.citation import CitationSourceGateNode
from seele_scholar_agent.state import PaperMetadata


@pytest.mark.asyncio
async def test_citation_source_gate_builds_stable_sources(base_state):
    papers = [
        PaperMetadata(
            paper_id="oa:1",
            title="Citable Paper",
            authors=["Author A"],
            abstract="Abstract.",
            doi="10.1234/test",
            year=2024,
            source="openalex",
        )
    ]

    result = await CitationSourceGateNode().build(
        {**base_state, "papers": papers, "search_queries": ["test query"]}
    )

    sources = result["citation_sources"]
    assert len(sources) == 1
    assert sources[0].citation_id == 1
    assert sources[0].doi == "10.1234/test"
    assert sources[0].source_quality.citable is True
    assert result["retrieval_diagnostics"][0].query == "test query"


@pytest.mark.asyncio
async def test_citation_source_gate_reports_uncitable_candidates(base_state):
    papers = [
        PaperMetadata(
            paper_id="bad",
            title="",
            authors=[],
            abstract="Abstract.",
            source="openalex",
        )
    ]

    result = await CitationSourceGateNode().build({**base_state, "papers": papers})

    assert result["citation_sources"] == []
    assert result["quality_issues"][0].code == "INSUFFICIENT_CITABLE_SOURCES"
    assert result["quality_issues"][0].blocking is False
