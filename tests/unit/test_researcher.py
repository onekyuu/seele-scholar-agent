from typing import cast
from unittest.mock import patch

import httpx
import pytest
from seele_scholar_agent.nodes.researcher import (
    ArxivRetriever,
    OpenAlexRetriever,
    ResearcherNode,
    SemanticScholarRetriever,
    _dedupe_and_rank_papers,
)
from seele_scholar_agent.state import AgentState, PaperMetadata

from tests.fixtures.mock_responses import (
    ARXIV_EMPTY_FEED_XML,
    ARXIV_MISSING_TITLE_XML,
    ARXIV_SINGLE_ENTRY_XML,
    ARXIV_TWO_ENTRIES_XML,
    OPENALEX_NULL_ABSTRACT,
    OPENALEX_SINGLE_RESULT,
    SEMANTIC_SCHOLAR_TWO_PAPERS,
)


def _make_paper(
    paper_id: str,
    source: str = "openalex",
    relevance_score: float = 0.5,
    title: str | None = None,
    authors: list[str] | None = None,
    abstract: str = "",
    doi: str | None = None,
    year: int | None = None,
) -> PaperMetadata:
    return PaperMetadata(
        paper_id=paper_id,
        title=title or paper_id.title(),
        authors=authors or [],
        abstract=abstract,
        source=source,  # type: ignore[arg-type]
        relevance_score=relevance_score,
        doi=doi,
        year=year,
    )


@pytest.mark.asyncio
async def test_arxiv_retriever_parse_xml(respx_mock):
    respx_mock.get(url__startswith="https://export.arxiv.org").mock(
        return_value=httpx.Response(200, text=ARXIV_TWO_ENTRIES_XML)
    )
    retriever = ArxivRetriever(top_k=10)
    papers = await retriever.search("test query")
    assert len(papers) == 2
    assert papers[0].source == "arxiv"
    assert "First Test Paper" in [p.title for p in papers]
    assert "Second Test Paper" in [p.title for p in papers]


@pytest.mark.asyncio
async def test_arxiv_retriever_uses_encoded_all_query(respx_mock):
    respx_mock.get(url__startswith="https://export.arxiv.org").mock(
        return_value=httpx.Response(200, text=ARXIV_EMPTY_FEED_XML)
    )
    retriever = ArxivRetriever()

    await retriever.search("large language model")

    called_url = str(respx_mock.calls.last.request.url)
    assert "search_query=all%3Alarge+language+model" in called_url


def test_arxiv_retriever_parses_atom_authors():
    xml = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <id>https://arxiv.org/abs/2401.00001</id>
    <title>Atom Author Paper</title>
    <author><name>Alice Author</name></author>
    <author><name>Bob Researcher</name></author>
    <published>2024-01-01T00:00:00Z</published>
    <summary>Paper abstract.</summary>
  </entry>
</feed>"""
    retriever = ArxivRetriever()

    papers = retriever._parse_response(xml)

    assert papers[0].authors == ["Alice Author", "Bob Researcher"]
    assert papers[0].year == 2024


@pytest.mark.asyncio
async def test_arxiv_retriever_empty_feed(respx_mock):
    respx_mock.get(url__startswith="https://export.arxiv.org").mock(
        return_value=httpx.Response(200, text=ARXIV_EMPTY_FEED_XML)
    )
    retriever = ArxivRetriever()
    papers = await retriever.search("nothing")
    assert papers == []


@pytest.mark.asyncio
async def test_arxiv_retriever_rate_limit_retry(respx_mock):
    respx_mock.get(url__startswith="https://export.arxiv.org").mock(
        side_effect=[
            httpx.Response(429, headers={"retry-after": "0"}),
            httpx.Response(200, text=ARXIV_SINGLE_ENTRY_XML),
        ]
    )
    retriever = ArxivRetriever()
    papers = await retriever.search("test")
    assert len(papers) == 1
    assert papers[0].title == "Test Paper Title"


@pytest.mark.asyncio
async def test_arxiv_retriever_http_500_returns_empty(respx_mock):
    respx_mock.get(url__startswith="https://export.arxiv.org").mock(
        return_value=httpx.Response(500)
    )
    retriever = ArxivRetriever()
    papers = await retriever.search("test")
    assert papers == []


@pytest.mark.asyncio
async def test_arxiv_retriever_missing_title_tag(respx_mock):
    respx_mock.get(url__startswith="https://export.arxiv.org").mock(
        return_value=httpx.Response(200, text=ARXIV_MISSING_TITLE_XML)
    )
    retriever = ArxivRetriever()
    papers = await retriever.search("test")
    assert len(papers) == 1
    assert papers[0].title == ""


@pytest.mark.asyncio
async def test_openalex_retriever_normal(respx_mock):
    respx_mock.get(url__startswith="https://api.openalex.org").mock(
        return_value=httpx.Response(200, json=OPENALEX_SINGLE_RESULT)
    )
    retriever = OpenAlexRetriever()
    papers = await retriever.search("transformer")
    assert len(papers) == 1
    assert papers[0].source == "openalex"
    assert papers[0].title == "OpenAlex Test Paper"
    assert "Jane Doe" in papers[0].authors
    assert "This is the abstract" in papers[0].abstract


@pytest.mark.asyncio
async def test_openalex_retriever_null_abstract(respx_mock):
    respx_mock.get(url__startswith="https://api.openalex.org").mock(
        return_value=httpx.Response(200, json=OPENALEX_NULL_ABSTRACT)
    )
    retriever = OpenAlexRetriever()
    papers = await retriever.search("test")
    assert len(papers) == 1
    assert papers[0].abstract == ""


@pytest.mark.asyncio
async def test_openalex_rate_limit_above_wait_budget_skips_without_sleep(respx_mock):
    route = respx_mock.get(url__startswith="https://api.openalex.org").mock(
        return_value=httpx.Response(429, headers={"retry-after": "38503"})
    )
    retriever = OpenAlexRetriever()

    with patch("seele_scholar_agent.nodes.researcher.asyncio.sleep") as sleep_mock:
        papers = await retriever.search("test")

    assert papers == []
    assert route.call_count == 1
    sleep_mock.assert_not_called()


def test_openalex_calculate_relevance_high_citation():
    retriever = OpenAlexRetriever()
    score = retriever._calculate_relevance({"cited_by_count": 5000, "publication_year": 2023})
    assert score == 1.0


def test_openalex_calculate_relevance_old_paper_zero_citation():
    retriever = OpenAlexRetriever()
    score = retriever._calculate_relevance({"cited_by_count": 0, "publication_year": 2015})
    assert score == pytest.approx(0.18, abs=0.01)


@pytest.mark.asyncio
async def test_semantic_scholar_retriever_normal(respx_mock):
    respx_mock.get(url__startswith="https://api.semanticscholar.org").mock(
        return_value=httpx.Response(200, json=SEMANTIC_SCHOLAR_TWO_PAPERS)
    )
    retriever = SemanticScholarRetriever()
    papers = await retriever.search("test")
    assert len(papers) == 2
    assert all(p.source == "semantic_scholar" for p in papers)
    assert papers[0].paper_id == "s2:paper001"
    assert papers[0].pdf_url == "https://pdf.semanticscholar.org/001.pdf"


@pytest.mark.asyncio
async def test_semantic_scholar_retriever_http_403(respx_mock):
    respx_mock.get(url__startswith="https://api.semanticscholar.org").mock(
        return_value=httpx.Response(403)
    )
    retriever = SemanticScholarRetriever()
    papers = await retriever.search("test")
    assert papers == []


@pytest.mark.asyncio
async def test_semantic_scholar_rate_limit_above_wait_budget_skips_without_sleep(respx_mock):
    route = respx_mock.get(url__startswith="https://api.semanticscholar.org").mock(
        return_value=httpx.Response(429, headers={"retry-after": "38503"})
    )
    retriever = SemanticScholarRetriever()

    with patch("seele_scholar_agent.nodes.researcher.asyncio.sleep") as sleep_mock:
        papers = await retriever.search("test")

    assert papers == []
    assert route.call_count == 1
    sleep_mock.assert_not_called()


@pytest.mark.asyncio
async def test_researcher_node_deduplication(state_with_papers, respx_mock):
    duplicate = PaperMetadata(
        paper_id="shared:001",
        title="Shared Paper",
        authors=["Author X"],
        abstract="Abstract.",
        source="openalex",
        relevance_score=0.9,
    )

    with (
        patch.object(OpenAlexRetriever, "search", return_value=[duplicate]),
        patch.object(SemanticScholarRetriever, "search", return_value=[duplicate]),
        patch.object(ArxivRetriever, "search", return_value=[]),
    ):
        node = ResearcherNode()
        result = await node.search(
            cast(AgentState, {**state_with_papers, "papers": [], "topic": "test"})
        )

    assert len(result["papers"]) == 1


def test_researcher_dedupes_by_doi_title_and_author_year():
    doi_a = _make_paper(
        "openalex:1",
        title="Retrieval Augmented Generation for Scholarly Writing",
        authors=["Jane Doe"],
        abstract="Retrieval augmented generation for scholarly writing.",
        doi="10.1234/RAG.001",
        year=2024,
        relevance_score=0.4,
    )
    doi_b = _make_paper(
        "s2:2",
        source="semantic_scholar",
        title="Retrieval-Augmented Generation for Scholarly Writing",
        authors=["J. Doe"],
        abstract="Same paper from another source.",
        doi="https://doi.org/10.1234/rag.001",
        year=2024,
        relevance_score=0.8,
    )
    title_duplicate = _make_paper(
        "arxiv:3",
        source="arxiv",
        title="Retrieval Augmented Generation for Scholarly Writing",
        authors=["Jane Doe"],
        abstract="Preprint copy.",
        year=2024,
        relevance_score=0.7,
    )

    ranked = _dedupe_and_rank_papers(
        [doi_a, doi_b, title_duplicate],
        ["retrieval augmented generation scholarly writing"],
    )

    assert len(ranked) == 1
    assert ranked[0].doi == "10.1234/rag.001"
    assert ranked[0].query_overlap_score > 0


def test_researcher_rerank_uses_query_overlap_and_user_priority():
    low_overlap = _make_paper(
        "low",
        title="Unrelated Optimization",
        abstract="A paper about unrelated systems.",
        relevance_score=0.8,
    )
    high_overlap = _make_paper(
        "high",
        title="Large Language Model Alignment",
        abstract="Large language model alignment and safety evaluation.",
        relevance_score=0.2,
    )
    user_paper = _make_paper(
        "user",
        source="user_library",
        title="Alignment Notes",
        abstract="Large language model alignment notes.",
        relevance_score=0.2,
    )

    ranked = _dedupe_and_rank_papers(
        [low_overlap, high_overlap, user_paper],
        ["large language model alignment"],
    )

    assert ranked[0].paper_id in {"high", "user"}
    assert ranked[0].query_overlap_score > low_overlap.query_overlap_score
    assert any(p.user_priority > 0 for p in ranked if p.source == "user_library")


@pytest.mark.asyncio
async def test_researcher_node_sorted_by_relevance(base_state, respx_mock):
    p_low = PaperMetadata(
        paper_id="low", title="Low", authors=[], abstract="", source="openalex", relevance_score=0.5
    )
    p_high = PaperMetadata(
        paper_id="high",
        title="High",
        authors=[],
        abstract="",
        source="openalex",
        relevance_score=0.9,
    )
    p_mid = PaperMetadata(
        paper_id="mid",
        title="Mid",
        authors=[],
        abstract="",
        source="semantic_scholar",
        relevance_score=0.7,
    )

    with (
        patch.object(OpenAlexRetriever, "search", return_value=[p_low]),
        patch.object(SemanticScholarRetriever, "search", return_value=[p_high]),
        patch.object(ArxivRetriever, "search", return_value=[p_mid]),
    ):
        node = ResearcherNode()
        result = await node.search(cast(AgentState, {**base_state, "topic": "test"}))

    scores = [p.relevance_score for p in result["papers"]]
    assert scores == sorted(scores, reverse=True)


@pytest.mark.asyncio
async def test_researcher_non_english_without_llm_returns_fallback_queries(base_state):
    with (
        patch.object(OpenAlexRetriever, "search", return_value=[]),
        patch.object(SemanticScholarRetriever, "search", return_value=[]),
        patch.object(ArxivRetriever, "search", return_value=[]),
    ):
        node = ResearcherNode(llm=None)
        result = await node.search(cast(AgentState, {**base_state, "topic": "大语言模型 RAG"}))

    assert "大语言模型 RAG" in result["search_queries"]
    assert "RAG" in result["search_queries"]


@pytest.mark.asyncio
async def test_researcher_node_all_sources_fail(base_state, respx_mock):
    respx_mock.get(url__startswith="https://api.openalex.org").mock(
        return_value=httpx.Response(500)
    )
    respx_mock.get(url__startswith="https://api.semanticscholar.org").mock(
        return_value=httpx.Response(500)
    )
    respx_mock.get(url__startswith="https://export.arxiv.org").mock(
        return_value=httpx.Response(500)
    )

    node = ResearcherNode()
    result = await node.search(cast(AgentState, {**base_state, "topic": "test"}))
    assert result["papers"] == []
    assert result["status"] == "planning"


# ---------------------------------------------------------------------------
# R-extra-01: extra_paper_retrievers results are merged into output
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_extra_retriever_results_merged(base_state):
    extra_paper = _make_paper("pubmed:99999", source="user_library", relevance_score=0.75)

    async def pubmed_retriever(query: str) -> list[PaperMetadata]:
        return [extra_paper]

    with (
        patch.object(OpenAlexRetriever, "search", return_value=[]),
        patch.object(SemanticScholarRetriever, "search", return_value=[]),
        patch.object(ArxivRetriever, "search", return_value=[]),
    ):
        node = ResearcherNode(extra_paper_retrievers=[pubmed_retriever])
        result = await node.search(cast(AgentState, {**base_state, "topic": "test"}))

    assert any(p.paper_id == "pubmed:99999" for p in result["papers"])


# ---------------------------------------------------------------------------
# R-extra-02: multiple extra_paper_retrievers all called and results merged
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_multiple_extra_retrievers_all_called(base_state):
    calls: list[str] = []

    async def retriever_a(query: str) -> list[PaperMetadata]:
        calls.append("a")
        return [_make_paper("a:001")]

    async def retriever_b(query: str) -> list[PaperMetadata]:
        calls.append("b")
        return [_make_paper("b:001")]

    with (
        patch.object(OpenAlexRetriever, "search", return_value=[]),
        patch.object(SemanticScholarRetriever, "search", return_value=[]),
        patch.object(ArxivRetriever, "search", return_value=[]),
    ):
        node = ResearcherNode(extra_paper_retrievers=[retriever_a, retriever_b])
        result = await node.search(cast(AgentState, {**base_state, "topic": "test"}))

    assert "a" in calls
    assert "b" in calls
    paper_ids = {p.paper_id for p in result["papers"]}
    assert "a:001" in paper_ids
    assert "b:001" in paper_ids


# ---------------------------------------------------------------------------
# R-extra-03: extra_paper_retrievers duplicate IDs are deduplicated
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_extra_retriever_deduplication_with_builtin(base_state):
    shared = _make_paper("shared:overlap", source="openalex", relevance_score=0.9)

    async def extra_retriever(query: str) -> list[PaperMetadata]:
        return [shared]

    with (
        patch.object(OpenAlexRetriever, "search", return_value=[shared]),
        patch.object(SemanticScholarRetriever, "search", return_value=[]),
        patch.object(ArxivRetriever, "search", return_value=[]),
    ):
        node = ResearcherNode(extra_paper_retrievers=[extra_retriever])
        result = await node.search(cast(AgentState, {**base_state, "topic": "test"}))

    assert len([p for p in result["papers"] if p.paper_id == "shared:overlap"]) == 1


# ---------------------------------------------------------------------------
# R-extra-04: extra_paper_retrievers=None behaves identically to no argument
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_extra_retriever_none_is_noop(base_state):
    builtin_paper = _make_paper("oa:builtin", source="openalex", relevance_score=0.8)

    with (
        patch.object(OpenAlexRetriever, "search", return_value=[builtin_paper]),
        patch.object(SemanticScholarRetriever, "search", return_value=[]),
        patch.object(ArxivRetriever, "search", return_value=[]),
    ):
        node_no_arg = ResearcherNode()
        node_none = ResearcherNode(extra_paper_retrievers=None)
        result_no_arg = await node_no_arg.search(cast(AgentState, {**base_state, "topic": "test"}))
        result_none = await node_none.search(cast(AgentState, {**base_state, "topic": "test"}))

    assert [p.paper_id for p in result_no_arg["papers"]] == [
        p.paper_id for p in result_none["papers"]
    ]


# ---------------------------------------------------------------------------
# R-extra-05: extra_paper_retrievers results are included in sort order
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_extra_retriever_sorted_with_builtins(base_state):
    builtin_low = _make_paper("oa:low", source="openalex", relevance_score=0.3)
    extra_high = _make_paper("extra:high", source="user_library", relevance_score=0.95)

    async def extra_retriever(query: str) -> list[PaperMetadata]:
        return [extra_high]

    with (
        patch.object(OpenAlexRetriever, "search", return_value=[builtin_low]),
        patch.object(SemanticScholarRetriever, "search", return_value=[]),
        patch.object(ArxivRetriever, "search", return_value=[]),
    ):
        node = ResearcherNode(extra_paper_retrievers=[extra_retriever])
        result = await node.search(cast(AgentState, {**base_state, "topic": "test"}))

    assert result["papers"][0].paper_id == "extra:high"
    scores = [p.relevance_score for p in result["papers"]]
    assert scores == sorted(scores, reverse=True)
    assert result["papers"][0].paper_id == "extra:high"


# ---------------------------------------------------------------------------
# R-extra-06: astream emits searching_extra progress event when retrievers present
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_astream_emits_searching_extra_progress(base_state):
    async def extra_retriever(query: str) -> list[PaperMetadata]:
        return []

    with (
        patch.object(OpenAlexRetriever, "search", return_value=[]),
        patch.object(SemanticScholarRetriever, "search", return_value=[]),
        patch.object(ArxivRetriever, "search", return_value=[]),
    ):
        node = ResearcherNode(extra_paper_retrievers=[extra_retriever])
        events = [e async for e in node.astream(cast(AgentState, {**base_state, "topic": "test"}))]

    progress_values = [e.get("progress", "") for e in events if e.get("type") == "progress"]
    assert any(v.startswith("searching_extra:") for v in progress_values)


# ---------------------------------------------------------------------------
# R-extra-07: astream does NOT emit searching_extra when no extra retrievers
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_astream_no_searching_extra_without_retrievers(base_state):
    with (
        patch.object(OpenAlexRetriever, "search", return_value=[]),
        patch.object(SemanticScholarRetriever, "search", return_value=[]),
        patch.object(ArxivRetriever, "search", return_value=[]),
    ):
        node = ResearcherNode()
        events = [e async for e in node.astream(cast(AgentState, {**base_state, "topic": "test"}))]

    progress_values = [e.get("progress", "") for e in events if e.get("type") == "progress"]
    assert not any(v.startswith("searching_extra:") for v in progress_values)


# ---------------------------------------------------------------------------
# R-extra-08: astream final result event includes extra_paper_retrievers output
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_astream_result_includes_extra_papers(base_state):
    extra_paper = _make_paper("pubmed:astream-test", source="user_library", relevance_score=0.6)

    async def extra_retriever(query: str) -> list[PaperMetadata]:
        return [extra_paper]

    with (
        patch.object(OpenAlexRetriever, "search", return_value=[]),
        patch.object(SemanticScholarRetriever, "search", return_value=[]),
        patch.object(ArxivRetriever, "search", return_value=[]),
    ):
        node = ResearcherNode(extra_paper_retrievers=[extra_retriever])
        events = [e async for e in node.astream(cast(AgentState, {**base_state, "topic": "test"}))]

    result_events = [e for e in events if e.get("type") == "result"]
    assert len(result_events) == 1
    papers = result_events[0].get("result", {}).get("papers", [])
    assert any(p.paper_id == "pubmed:astream-test" for p in papers)
