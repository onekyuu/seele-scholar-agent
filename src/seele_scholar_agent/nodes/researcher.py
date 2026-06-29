import asyncio
import re
import unicodedata
from collections.abc import AsyncIterator
from typing import Any
from urllib.parse import urlencode
from xml.etree import ElementTree

from httpx import AsyncClient, HTTPStatusError
from langchain_core.language_models import BaseLanguageModel

from seele_scholar_agent.agent_config import PaperSearchFunc, PromptsConfig
from seele_scholar_agent.state import AgentState, PaperMetadata

from ..config import settings
from ..logging import get_logger
from . import (
    API_MAX_RETRIES,
    API_RETRY_BASE_DELAY,
    ARXIV_RATE_LIMIT_DELAY,
    HTTP_TIMEOUT,
    PAPER_SUMMARY_ABSTRACT_CHARS,
    NodeStreamEvent,
)
from .material_registry import (
    apply_material_registry_priority,
    get_material_registry,
    material_policy_suffix,
)

logger = get_logger(__name__)

# 保留在 state 中的 abstract 最大字符数（节省序列化体积）
_PAPER_STATE_ABSTRACT_CHARS = settings.PAPER_STATE_ABSTRACT_CHARS
_RETRIEVER_MAX_RETRY_AFTER_SECONDS = settings.RETRIEVER_MAX_RETRY_AFTER_SECONDS
_DOI_RE = re.compile(r"10\.\d{4,9}/[-._;()/:A-Z0-9]+", re.IGNORECASE)
_WORD_RE = re.compile(r"[A-Za-z0-9]+")
_NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")
_ARXIV_NS = {"atom": "http://www.w3.org/2005/Atom"}


def _parse_retry_after(value: str | None, *, default: float = 5.0) -> float:
    if not value:
        return default
    try:
        return max(float(value), 0.0)
    except ValueError:
        return default


async def _wait_or_skip_rate_limit(
    source: str,
    retry_after_header: str | None,
    *,
    attempt: int,
) -> bool:
    retry_after = _parse_retry_after(retry_after_header)
    attempt_number = attempt + 1
    if retry_after > _RETRIEVER_MAX_RETRY_AFTER_SECONDS:
        logger.warning(
            f"{source} rate limited beyond wait budget, skipping source",
            retry_after=retry_after,
            max_retry_after=_RETRIEVER_MAX_RETRY_AFTER_SECONDS,
            attempt=attempt_number,
        )
        return False
    if attempt >= API_MAX_RETRIES - 1:
        logger.warning(
            f"{source} rate limited on final attempt, skipping source",
            retry_after=retry_after,
            attempt=attempt_number,
        )
        return False
    logger.warning(
        f"{source} rate limited, retrying",
        retry_after=retry_after,
        attempt=attempt_number,
    )
    await asyncio.sleep(retry_after)
    return True


def _normalize_doi(value: str | None) -> str | None:
    if not value:
        return None
    match = _DOI_RE.search(value)
    if not match:
        return None
    return match.group(0).rstrip(".,;").lower()


def _normalize_title(title: str) -> str:
    normalized = unicodedata.normalize("NFKD", title).lower()
    normalized = _NON_ALNUM_RE.sub(" ", normalized)
    return " ".join(normalized.split())


def _query_tokens(text: str) -> set[str]:
    return {token.lower() for token in _WORD_RE.findall(text) if len(token) > 2}


def _query_overlap_score(query: str, paper: PaperMetadata) -> float:
    query_terms = _query_tokens(query)
    if not query_terms:
        return 0.0
    paper_terms = _query_tokens(f"{paper.title} {paper.abstract}")
    if not paper_terms:
        return 0.0
    return round(len(query_terms & paper_terms) / len(query_terms), 3)


def _first_author_key(authors: list[str]) -> str:
    if not authors:
        return ""
    first = unicodedata.normalize("NFKD", authors[0]).lower()
    tokens = _WORD_RE.findall(first)
    return tokens[-1] if tokens else ""


def _paper_identity_keys(paper: PaperMetadata) -> list[str]:
    keys: list[str] = []
    doi = paper.doi or _normalize_doi(paper.url) or _normalize_doi(paper.pdf_url)
    if doi:
        keys.append(f"doi:{doi}")
    if paper.paper_id:
        keys.append(f"id:{paper.paper_id.lower()}")
    title = _normalize_title(paper.title)
    if title:
        keys.append(f"title:{title}")
    author = _first_author_key(paper.authors)
    if author and paper.year and title:
        keys.append(f"author_year:{author}:{paper.year}:{title[:80]}")
    return keys


def _merge_duplicate_papers(existing: PaperMetadata, incoming: PaperMetadata) -> PaperMetadata:
    if _paper_quality_score(incoming) <= _paper_quality_score(existing):
        return existing
    merged_score = max(existing.relevance_score, incoming.relevance_score)
    merged_overlap = max(existing.query_overlap_score, incoming.query_overlap_score)
    return incoming.model_copy(
        update={
            "relevance_score": merged_score,
            "query_overlap_score": merged_overlap,
            "user_priority": max(existing.user_priority, incoming.user_priority),
        }
    )


def _paper_quality_score(paper: PaperMetadata) -> float:
    metadata_score = 0.0
    if paper.doi or _normalize_doi(paper.url):
        metadata_score += 0.15
    if paper.abstract:
        metadata_score += 0.1
    if paper.authors:
        metadata_score += 0.05
    if paper.year:
        metadata_score += 0.05
    source_score = {"user_library": 0.2, "openalex": 0.08, "semantic_scholar": 0.08, "arxiv": 0.05}
    return (
        paper.relevance_score
        + paper.query_overlap_score * 0.35
        + (paper.embedding_similarity or 0.0) * 0.25
        + paper.user_priority
        + metadata_score
        + source_score.get(paper.source, 0.0)
    )


def _dedupe_and_rank_papers(papers: list[PaperMetadata], queries: list[str]) -> list[PaperMetadata]:
    indexed: dict[str, PaperMetadata] = {}
    key_to_primary: dict[str, str] = {}

    for paper in papers:
        best_overlap = max((_query_overlap_score(query, paper) for query in queries), default=0.0)
        user_priority = max(paper.user_priority, 0.2 if paper.source == "user_library" else 0.0)
        enriched = paper.model_copy(
            update={
                "doi": (
                    _normalize_doi(paper.doi)
                    or _normalize_doi(paper.url)
                    or _normalize_doi(paper.pdf_url)
                ),
                "query_overlap_score": best_overlap,
                "user_priority": user_priority,
            }
        )
        keys = _paper_identity_keys(enriched)
        primary = next((key_to_primary[key] for key in keys if key in key_to_primary), None)
        if primary is None:
            primary = keys[0] if keys else f"object:{len(indexed)}"
            indexed[primary] = enriched
        else:
            indexed[primary] = _merge_duplicate_papers(indexed[primary], enriched)
        for key in keys:
            key_to_primary[key] = primary

    ranked = []
    for paper in indexed.values():
        ranked.append(
            paper.model_copy(
                update={"relevance_score": round(min(_paper_quality_score(paper), 1.0), 3)}
            )
        )
    ranked.sort(key=lambda paper: paper.relevance_score, reverse=True)
    return ranked


def _looks_like_placeholder_query(query: str) -> bool:
    lowered = query.strip().lower()
    if not lowered:
        return True
    return lowered in {"topic", "{topic}", "n/a", "none"}


def _fallback_query_variants(topic: str) -> list[str]:
    normalized = " ".join(topic.split())
    variants = [normalized]
    ascii_terms = " ".join(_WORD_RE.findall(topic))
    if ascii_terms and ascii_terms != normalized:
        variants.append(ascii_terms)
    return list(dict.fromkeys(v for v in variants if v))


def _compress_papers(
    papers: list[PaperMetadata], registry: Any | None = None
) -> tuple[list[PaperMetadata], list[str]]:
    """Strip full abstracts from PaperMetadata (reduces state size) and build compact summaries.

    Returns:
        stripped_papers: PaperMetadata with abstract truncated to _PAPER_STATE_ABSTRACT_CHARS
        paper_summaries: list of compact 1-3 sentence summary strings, one per paper
    """
    stripped: list[PaperMetadata] = []
    summaries: list[str] = []
    for idx, p in enumerate(papers, 1):
        compact_abstract = p.abstract[:_PAPER_STATE_ABSTRACT_CHARS] if p.abstract else ""
        stripped.append(p.model_copy(update={"abstract": compact_abstract}))

        authors_str = ", ".join(p.authors[:2])
        if len(p.authors) > 2:
            authors_str += " et al."

        abstract = (p.abstract or "").strip()
        if len(abstract) > PAPER_SUMMARY_ABSTRACT_CHARS:
            # Try to end at a sentence boundary
            snippet = abstract[:PAPER_SUMMARY_ABSTRACT_CHARS]
            last_period = snippet.rfind(". ")
            if last_period > PAPER_SUMMARY_ABSTRACT_CHARS // 2:
                snippet = snippet[: last_period + 1]
            else:
                snippet += "..."
        else:
            snippet = abstract

        summaries.append(
            f"[{idx}] {p.title} — {authors_str}. {snippet}"
            f"{material_policy_suffix(p, registry)}"
        )

    return stripped, summaries


class ArxivRetriever:
    BASE_URL = "https://export.arxiv.org/api/query"

    def __init__(self, top_k: int = settings.RETRIEVER_TOP_K):
        self.top_k = top_k

    async def search(self, query: str) -> list[PaperMetadata]:
        params = {
            "search_query": f"all:{query}",
            "sortBy": "relevance",
            "sortOrder": "descending",
            "start": 0,
            "max_results": self.top_k,
        }
        search_url = f"{self.BASE_URL}?{urlencode(params)}"

        for attempt in range(API_MAX_RETRIES):
            try:
                async with AsyncClient(timeout=HTTP_TIMEOUT) as client:
                    response = await client.get(search_url)

                    if response.status_code == 429:
                        should_retry = await _wait_or_skip_rate_limit(
                            "ArXiv",
                            response.headers.get("retry-after"),
                            attempt=attempt,
                        )
                        if should_retry:
                            continue
                        return []

                    if response.status_code != 200:
                        logger.error("ArXiv API error", status_code=response.status_code)
                        return []

                    return self._parse_response(response.text)

            except HTTPStatusError as e:
                if attempt < API_MAX_RETRIES - 1:
                    wait_time = API_RETRY_BASE_DELAY * (2**attempt)
                    logger.warning(
                        "ArXiv request failed, retrying",
                        wait_time=wait_time,
                        attempt=attempt + 1,
                        error=str(e),
                    )
                    await asyncio.sleep(wait_time)
                    continue
                logger.error("ArXiv search failed after retries", error=str(e))
                return []
            except Exception as e:
                logger.error("ArXiv search failed", error=str(e))
                return []

        return []

    def _parse_response(self, xml_text: str) -> list[PaperMetadata]:
        papers: list[PaperMetadata] = []
        try:
            root = ElementTree.fromstring(xml_text)
        except ElementTree.ParseError as e:
            logger.warning("failed to parse ArXiv XML", error=str(e))
            return []

        entries = root.findall("atom:entry", _ARXIV_NS)
        for entry in entries[: self.top_k]:
            try:
                paper_id = self._entry_text(entry, "id")
                title = self._entry_text(entry, "title").replace("\n", " ")
                authors = [
                    name.strip()
                    for name in [
                        author.findtext("atom:name", default="", namespaces=_ARXIV_NS)
                        for author in entry.findall("atom:author", _ARXIV_NS)
                    ]
                    if name.strip()
                ]
                if not authors:
                    authors = [
                        a.strip()
                        for a in self._entry_text(entry, "authors").split(",")
                        if a.strip()
                    ]
                abstract = self._entry_text(entry, "summary").replace("\n", " ")
                published = self._entry_text(entry, "published")
                year = self._extract_year(published)
                url = paper_id
                pdf_url = paper_id.replace("abs", "pdf") + ".pdf" if paper_id else ""

                papers.append(
                    PaperMetadata(
                        paper_id=paper_id,
                        title=title,
                        authors=authors,
                        abstract=abstract,
                        url=url,
                        pdf_url=pdf_url,
                        source="arxiv",
                        relevance_score=0.8,
                        year=year,
                    )
                )
            except Exception as e:
                logger.warning("failed to parse ArXiv entry", error=str(e))
                continue

        return papers

    def _entry_text(self, entry: ElementTree.Element, tag: str) -> str:
        return entry.findtext(f"atom:{tag}", default="", namespaces=_ARXIV_NS).strip()

    def _extract_year(self, value: str) -> int | None:
        match = re.search(r"\b(19|20)\d{2}\b", value)
        return int(match.group(0)) if match else None


class OpenAlexRetriever:
    BASE_URL = "https://api.openalex.org/works"

    def __init__(self, top_k: int = settings.RETRIEVER_TOP_K, email: str | None = None):
        self.top_k = top_k
        self.email = email or "research@example.com"

    def _reconstruct_abstract(self, inverted_index: dict[str, list[int]] | None) -> str:
        if not inverted_index:
            return ""
        max_idx = max(idx for indices in inverted_index.values() for idx in indices)
        words = [""] * (max_idx + 1)
        for word, indices in inverted_index.items():
            for idx in indices:
                words[idx] = word
        return " ".join(words).strip()

    def _calculate_relevance(self, paper: dict[str, Any]) -> float:
        raw_citation_count = paper.get("cited_by_count", 0)
        raw_year = paper.get("publication_year", 2020)
        citation_count = raw_citation_count if isinstance(raw_citation_count, int | float) else 0
        year = raw_year if isinstance(raw_year, int) else 2020
        citation_score = min(citation_count / 1000, 1.0)
        year_weight = 1.0 if year >= 2020 else 0.6
        return round(citation_score * 0.7 + year_weight * 0.3, 2)

    async def search(self, query: str) -> list[PaperMetadata]:
        params: dict[str, str | int] = {
            "search": query,
            "per-page": self.top_k,
            "mailto": self.email,
            "select": (
                "id,doi,title,publication_year,authorships,"
                "abstract_inverted_index,cited_by_count,primary_location"
            ),
        }

        for attempt in range(API_MAX_RETRIES):
            try:
                async with AsyncClient(timeout=HTTP_TIMEOUT) as client:
                    logger.info("OpenAlex searching", attempt=attempt + 1)
                    response = await client.get(self.BASE_URL, params=params)

                    if response.status_code == 429:
                        should_retry = await _wait_or_skip_rate_limit(
                            "OpenAlex",
                            response.headers.get("retry-after"),
                            attempt=attempt,
                        )
                        if should_retry:
                            continue
                        return []

                    if response.status_code != 200:
                        logger.error("OpenAlex API error", status_code=response.status_code)
                        return []

                    data = response.json()
                    papers = []

                    for item in data.get("results", []):
                        authorships = item.get("authorships", [])
                        authors = [
                            author.get("author", {}).get("display_name", "")
                            for author in authorships
                            if author.get("author", {}).get("display_name")
                        ]

                        abstract = self._reconstruct_abstract(item.get("abstract_inverted_index"))
                        relevance = self._calculate_relevance(item)
                        doi = _normalize_doi(item.get("doi"))
                        primary_location = item.get("primary_location") or {}
                        venue = (primary_location.get("source") or {}).get("display_name")

                        papers.append(
                            PaperMetadata(
                                paper_id=item.get("id", ""),
                                title=item.get("title", ""),
                                authors=authors,
                                abstract=abstract,
                                url=item.get("doi", ""),
                                pdf_url="",
                                source="openalex",
                                relevance_score=relevance,
                                doi=doi,
                                year=item.get("publication_year"),
                                venue=venue,
                            )
                        )

                    return papers

            except Exception as e:
                if attempt < API_MAX_RETRIES - 1:
                    wait_time = API_RETRY_BASE_DELAY * (2**attempt)
                    logger.warning(
                        "OpenAlex request failed, retrying",
                        wait_time=wait_time,
                        attempt=attempt + 1,
                        error=str(e),
                    )
                    await asyncio.sleep(wait_time)
                    continue
                logger.error("OpenAlex search failed after retries", error=str(e))
                return []

        return []


class SemanticScholarRetriever:
    BASE_URL = "https://api.semanticscholar.org/graph/v1"

    def __init__(self, api_key: str | None = None, top_k: int = settings.RETRIEVER_TOP_K):
        self.api_key = api_key
        self.top_k = top_k
        self.headers: dict[str, str] = {"x-api-key": api_key} if api_key else {}

    async def search(self, query: str, fields: list[str] | None = None) -> list[PaperMetadata]:
        if fields is None:
            fields = [
                "paperId",
                "title",
                "authors",
                "abstract",
                "url",
                "openAccessPdf",
                "year",
                "venue",
                "citationCount",
                "externalIds",
            ]

        url = f"{self.BASE_URL}/paper/search"
        params: dict[str, str | int] = {
            "query": query,
            "limit": self.top_k,
            "fields": ",".join(fields),
        }

        for attempt in range(API_MAX_RETRIES):
            try:
                async with AsyncClient(timeout=HTTP_TIMEOUT) as client:
                    response = await client.get(url, params=params, headers=self.headers)

                    if response.status_code == 429:
                        should_retry = await _wait_or_skip_rate_limit(
                            "Semantic Scholar",
                            response.headers.get("retry-after"),
                            attempt=attempt,
                        )
                        if should_retry:
                            continue
                        return []

                    if response.status_code != 200:
                        logger.error("Semantic Scholar API error", status_code=response.status_code)
                        return []

                    data = response.json()

                papers = []
                for r in data.get("data", [])[: self.top_k]:
                    relevance = self._calculate_relevance(r)
                    external_ids = r.get("externalIds") or {}
                    doi = _normalize_doi(external_ids.get("DOI")) or _normalize_doi(r.get("url"))
                    open_access_pdf = r.get("openAccessPdf") or {}
                    papers.append(
                        PaperMetadata(
                            paper_id=r["paperId"],
                            title=r.get("title", ""),
                            authors=[a.get("name", "") for a in r.get("authors", [])],
                            abstract=r.get("abstract", ""),
                            url=r.get("url", ""),
                            pdf_url=open_access_pdf.get("url") or r.get("pdfUrl", ""),
                            source="semantic_scholar",
                            relevance_score=relevance,
                            doi=doi,
                            year=r.get("year"),
                            venue=r.get("venue") or None,
                        )
                    )
                return papers

            except Exception as e:
                if attempt < API_MAX_RETRIES - 1:
                    wait_time = API_RETRY_BASE_DELAY * (2**attempt)
                    logger.warning(
                        "Semantic Scholar request failed, retrying",
                        wait_time=wait_time,
                        attempt=attempt + 1,
                        error=str(e),
                    )
                    await asyncio.sleep(wait_time)
                    continue
                logger.error("Semantic Scholar search failed after retries", error=str(e))
                return []

        return []

    def _calculate_relevance(self, paper: dict[str, Any]) -> float:
        raw_citation_count = paper.get("citationCount", 0)
        raw_year = paper.get("year", 2020)
        citation_count = raw_citation_count if isinstance(raw_citation_count, int | float) else 0
        year = raw_year if isinstance(raw_year, int) else 2020
        citation_score = min(citation_count / 1000, 1.0)
        year_weight = 1.0 if year >= 2020 else 0.6
        return round(citation_score * 0.7 + year_weight * 0.3, 2)


class ResearcherNode:
    def __init__(
        self,
        llm: BaseLanguageModel[Any] | None = None,
        prompts: PromptsConfig | None = None,
        semantic_scholar_key: str | None = None,
        openalex_email: str | None = None,
        extra_paper_retrievers: list[PaperSearchFunc] | None = None,
    ):
        self.llm = llm
        self.prompts = prompts
        self.arxiv = ArxivRetriever()
        self.openalex = OpenAlexRetriever(email=openalex_email)
        self.ss = SemanticScholarRetriever(api_key=semantic_scholar_key)
        self.extra_paper_retrievers: list[PaperSearchFunc] = extra_paper_retrievers or []

    def _needs_translation(self, topic: str) -> bool:
        for ch in topic:
            cp = ord(ch)
            if 0x4E00 <= cp <= 0x9FFF:
                return True
            if 0x3040 <= cp <= 0x30FF:
                return True
            if 0xAC00 <= cp <= 0xD7A3:
                return True
            if cp > 0x024F and unicodedata.category(ch) not in ("Po", "Pd", "Ps", "Pe", "Zs"):
                return True
        return False

    async def _translate_topic(self, topic: str) -> list[str]:
        if not self._needs_translation(topic):
            logger.info("topic is already English, skipping translation", topic=topic)
            return [topic]

        fallback_queries = _fallback_query_variants(topic)
        if self.llm is None:
            logger.warning(
                "no LLM injected into ResearcherNode; skipping translation, "
                "search quality may be reduced for non-English topics",
                topic=topic,
            )
            return fallback_queries

        try:
            from langchain_core.messages import HumanMessage, SystemMessage

            sys_prompt = self.prompts.topic_translation_system_prompt if self.prompts else ""
            user_prompt = self.prompts.topic_translation_user_prompt if self.prompts else "{topic}"
            messages = [
                SystemMessage(content=sys_prompt),
                HumanMessage(content=user_prompt.format(topic=topic)),
            ]
            response = await self.llm.ainvoke(messages)
            raw = response.content if hasattr(response, "content") else str(response)
            queries = [
                line.strip(" -0123456789.")
                for line in raw.strip().splitlines()
                if line.strip()
            ]
            queries = [q for q in queries if not _looks_like_placeholder_query(q)]
            if queries:
                queries = list(dict.fromkeys([*queries, *fallback_queries]))[:5]
                logger.info(
                    "topic translated to English queries",
                    original=topic,
                    queries=queries,
                )
                return queries
            logger.warning("LLM returned empty translation, falling back to original", topic=topic)
        except Exception as e:
            logger.error(
                "topic translation failed, falling back to original", topic=topic, error=str(e)
            )

        return fallback_queries

    async def search(self, state: AgentState) -> dict[str, Any]:
        topic = state["topic"]

        queries = await self._translate_topic(topic)
        all_papers = await self._search_queries(queries)
        all_papers = apply_material_registry_priority(all_papers, get_material_registry(state))
        ranked_papers = _dedupe_and_rank_papers(all_papers, queries)

        stripped_papers, paper_summaries = _compress_papers(
            ranked_papers, get_material_registry(state)
        )

        logger.info(
            "research complete",
            topic=topic,
            queries=queries,
            total_papers=len(stripped_papers),
        )

        return {
            "papers": stripped_papers,
            "paper_summaries": paper_summaries,
            "search_queries": queries,
            "status": "planning",
        }

    async def astream(self, state: AgentState) -> AsyncIterator[NodeStreamEvent]:
        topic = state["topic"]

        yield NodeStreamEvent(type="progress", progress="translating")
        queries = await self._translate_topic(topic)

        all_papers: list[PaperMetadata] = []
        for query in queries:
            yield NodeStreamEvent(type="progress", progress=f"searching_openalex_ss:{query}")
            if self.extra_paper_retrievers:
                yield NodeStreamEvent(type="progress", progress=f"searching_extra:{query}")
            query_papers = await self._search_single_query(query)
            all_papers.extend(query_papers)

        all_papers = apply_material_registry_priority(all_papers, get_material_registry(state))
        ranked_papers = _dedupe_and_rank_papers(all_papers, queries)
        stripped_papers, paper_summaries = _compress_papers(
            ranked_papers, get_material_registry(state)
        )

        yield NodeStreamEvent(
            type="result",
            result={
                "papers": stripped_papers,
                "paper_summaries": paper_summaries,
                "search_queries": queries,
                "status": "planning",
            },
        )

    async def _search_queries(self, queries: list[str]) -> list[PaperMetadata]:
        all_papers: list[PaperMetadata] = []
        for query in queries:
            all_papers.extend(await self._search_single_query(query))
        return all_papers

    async def _search_single_query(self, query: str) -> list[PaperMetadata]:
        logger.info("searching all sources", query=query)

        openalex_papers, ss_papers = await asyncio.gather(
            self.openalex.search(query),
            self.ss.search(query),
        )

        logger.info("searching ArXiv (with rate-limit delay)", query=query)
        await asyncio.sleep(ARXIV_RATE_LIMIT_DELAY)
        arxiv_papers = await self.arxiv.search(query)

        extra_results: list[list[PaperMetadata]] = []
        if self.extra_paper_retrievers:
            extra_results = list(
                await asyncio.gather(
                    *[retriever(query) for retriever in self.extra_paper_retrievers]
                )
            )

        return [
            *openalex_papers,
            *ss_papers,
            *arxiv_papers,
            *[p for batch in extra_results for p in batch],
        ]
