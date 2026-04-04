import asyncio
import unicodedata
from collections.abc import AsyncIterator
from typing import Any

from httpx import AsyncClient, HTTPStatusError
from langchain_core.language_models import BaseLanguageModel

from seele_scholar_agent.agent_config import PaperSearchFunc, PromptsConfig
from seele_scholar_agent.state import AgentState, PaperMetadata

from ..logging import get_logger
from . import (
    API_MAX_RETRIES,
    API_RETRY_BASE_DELAY,
    ARXIV_RATE_LIMIT_DELAY,
    HTTP_TIMEOUT,
    PAPER_SUMMARY_ABSTRACT_CHARS,
    NodeStreamEvent,
)

logger = get_logger(__name__)

# 保留在 state 中的 abstract 最大字符数（节省序列化体积）
_PAPER_STATE_ABSTRACT_CHARS = 100


def _compress_papers(papers: list[PaperMetadata]) -> tuple[list[PaperMetadata], list[str]]:
    """Strip full abstracts from PaperMetadata (reduces state size) and build compact summaries.

    Returns:
        stripped_papers: PaperMetadata with abstract truncated to _PAPER_STATE_ABSTRACT_CHARS
        paper_summaries: list of compact 1-3 sentence summary strings, one per paper
    """
    stripped: list[PaperMetadata] = []
    summaries: list[str] = []
    for idx, p in enumerate(papers, 1):
        # Keep only a short stub of abstract in state — downstream nodes use paper_summaries
        compact_abstract = p.abstract[:_PAPER_STATE_ABSTRACT_CHARS] if p.abstract else ""
        stripped.append(p.model_copy(update={"abstract": compact_abstract}))

        # Build compact summary: title, 2 authors, ~2 sentences of abstract
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

        summaries.append(f"[{idx}] {p.title} — {authors_str}. {snippet}")

    return stripped, summaries


class ArxivRetriever:
    BASE_URL = "https://export.arxiv.org/api/query"

    def __init__(self, top_k: int = 10):
        self.top_k = top_k

    async def search(self, query: str) -> list[PaperMetadata]:
        search_url = f"{self.BASE_URL}?search_query={query}&sortBy=relevance&sortOrder=descending&start=0&max_results={self.top_k}"

        for attempt in range(API_MAX_RETRIES):
            try:
                async with AsyncClient(timeout=HTTP_TIMEOUT) as client:
                    response = await client.get(search_url)

                    if response.status_code == 429:
                        retry_after = int(response.headers.get("retry-after", 5))
                        logger.warning(
                            "ArXiv rate limited, retrying",
                            retry_after=retry_after,
                            attempt=attempt + 1,
                        )
                        await asyncio.sleep(retry_after)
                        continue

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
        papers = []
        entries = xml_text.split("<entry>")[1:]

        for entry in entries[: self.top_k]:
            try:
                paper_id = self._extract_tag(entry, "id")
                title = self._extract_tag(entry, "title").replace("\n", " ")
                authors = [
                    a.strip() for a in self._extract_tag(entry, "authors").split(",") if a.strip()
                ]
                abstract = self._extract_tag(entry, "summary").replace("\n", " ")
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
                    )
                )
            except Exception as e:
                logger.warning("failed to parse ArXiv entry", error=str(e))
                continue

        return papers

    def _extract_tag(self, xml: str, tag: str) -> str:
        start = xml.find(f"<{tag}>")
        end = xml.find(f"</{tag}>")
        if start == -1 or end == -1:
            return ""
        return xml[start + len(tag) + 2 : end]


class OpenAlexRetriever:
    BASE_URL = "https://api.openalex.org/works"

    def __init__(self, top_k: int = 10, email: str | None = None):
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
        citation_count = paper.get("cited_by_count", 0)
        year = paper.get("publication_year", 2020)
        citation_score = min(citation_count / 1000, 1.0)
        year_weight = 1.0 if year >= 2020 else 0.6
        return round(citation_score * 0.7 + year_weight * 0.3, 2)

    async def search(self, query: str) -> list[PaperMetadata]:
        params = {
            "search": query,
            "per-page": self.top_k,
            "mailto": self.email,
            "select": "id,doi,title,publication_year,authorships,abstract_inverted_index,cited_by_count",
        }

        for attempt in range(API_MAX_RETRIES):
            try:
                async with AsyncClient(timeout=HTTP_TIMEOUT) as client:
                    logger.info("OpenAlex searching", attempt=attempt + 1)
                    response = await client.get(self.BASE_URL, params=params)

                    if response.status_code == 429:
                        retry_after = int(response.headers.get("retry-after", 5))
                        logger.warning(
                            "OpenAlex rate limited, retrying",
                            retry_after=retry_after,
                            attempt=attempt + 1,
                        )
                        await asyncio.sleep(retry_after)
                        continue

                    if response.status_code != 200:
                        logger.error("OpenAlex API error", status_code=response.status_code)
                        return []

                    data = response.json()
                    papers = []

                    for item in data.get("results", []):
                        authorships = item.get("authorships", [])
                        first_author = "Unknown"
                        if authorships:
                            first_author_dict = authorships[0].get("author", {})
                            if first_author_dict:
                                first_author = first_author_dict.get("display_name", "Unknown")

                        abstract = self._reconstruct_abstract(item.get("abstract_inverted_index"))
                        relevance = self._calculate_relevance(item)

                        papers.append(
                            PaperMetadata(
                                paper_id=item.get("id", ""),
                                title=item.get("title", ""),
                                authors=[first_author] if first_author != "Unknown" else [],
                                abstract=abstract,
                                url=item.get("doi", ""),
                                pdf_url="",
                                source="openalex",
                                relevance_score=relevance,
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

    def __init__(self, api_key: str | None = None, top_k: int = 10):
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
                "pdfUrl",
                "year",
                "citationCount",
            ]

        url = f"{self.BASE_URL}/paper/search"
        params = {"query": query, "limit": self.top_k, "fields": ",".join(fields)}

        for attempt in range(API_MAX_RETRIES):
            try:
                async with AsyncClient(timeout=HTTP_TIMEOUT) as client:
                    response = await client.get(url, params=params, headers=self.headers)

                    if response.status_code == 429:
                        retry_after = int(response.headers.get("retry-after", 5))
                        logger.warning(
                            "Semantic Scholar rate limited, retrying",
                            retry_after=retry_after,
                            attempt=attempt + 1,
                        )
                        await asyncio.sleep(retry_after)
                        continue

                    if response.status_code != 200:
                        logger.error("Semantic Scholar API error", status_code=response.status_code)
                        return []

                    data = response.json()

                papers = []
                for r in data.get("data", [])[: self.top_k]:
                    relevance = self._calculate_relevance(r)
                    papers.append(
                        PaperMetadata(
                            paper_id=r["paperId"],
                            title=r.get("title", ""),
                            authors=[a.get("name", "") for a in r.get("authors", [])],
                            abstract=r.get("abstract", ""),
                            url=r.get("url", ""),
                            pdf_url=r.get("pdfUrl", ""),
                            source="semantic_scholar",
                            relevance_score=relevance,
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
        citation_count = paper.get("citationCount", 0)
        year = paper.get("year", 2020)
        citation_score = min(citation_count / 1000, 1.0)
        year_weight = 1.0 if year >= 2020 else 0.6
        return round(citation_score * 0.7 + year_weight * 0.3, 2)


class ResearcherNode:
    def __init__(
        self,
        llm: BaseLanguageModel | None = None,
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

        if self.llm is None:
            logger.warning(
                "no LLM injected into ResearcherNode; skipping translation, "
                "search quality may be reduced for non-English topics",
                topic=topic,
            )
            return [topic]

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
            queries = [line.strip() for line in raw.strip().splitlines() if line.strip()]
            if queries:
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

        return [topic]

    async def search(self, state: AgentState) -> dict[str, Any]:
        topic = state["topic"]

        queries = await self._translate_topic(topic)

        all_papers: list[PaperMetadata] = []
        seen_ids: set[str] = set()

        for query in queries:
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

            for paper in [
                *openalex_papers,
                *ss_papers,
                *arxiv_papers,
                *[p for batch in extra_results for p in batch],
            ]:
                if paper.paper_id not in seen_ids:
                    seen_ids.add(paper.paper_id)
                    all_papers.append(paper)

        all_papers.sort(key=lambda x: x.relevance_score, reverse=True)

        # Compress: strip full abstracts from state, build compact summaries for LLM context
        stripped_papers, paper_summaries = _compress_papers(all_papers)

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
        seen_ids: set[str] = set()

        for query in queries:
            yield NodeStreamEvent(type="progress", progress=f"searching_openalex_ss:{query}")
            openalex_papers, ss_papers = await asyncio.gather(
                self.openalex.search(query),
                self.ss.search(query),
            )

            yield NodeStreamEvent(type="progress", progress=f"searching_arxiv:{query}")
            await asyncio.sleep(ARXIV_RATE_LIMIT_DELAY)
            arxiv_papers = await self.arxiv.search(query)

            extra_results: list[list[PaperMetadata]] = []
            if self.extra_paper_retrievers:
                yield NodeStreamEvent(type="progress", progress=f"searching_extra:{query}")
                extra_results = list(
                    await asyncio.gather(
                        *[retriever(query) for retriever in self.extra_paper_retrievers]
                    )
                )

            for paper in [
                *openalex_papers,
                *ss_papers,
                *arxiv_papers,
                *[p for batch in extra_results for p in batch],
            ]:
                if paper.paper_id not in seen_ids:
                    seen_ids.add(paper.paper_id)
                    all_papers.append(paper)

        all_papers.sort(key=lambda x: x.relevance_score, reverse=True)

        stripped_papers, paper_summaries = _compress_papers(all_papers)

        yield NodeStreamEvent(
            type="result",
            result={
                "papers": stripped_papers,
                "paper_summaries": paper_summaries,
                "search_queries": queries,
                "status": "planning",
            },
        )
