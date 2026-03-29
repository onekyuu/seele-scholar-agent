import asyncio
from typing import Any

from httpx import AsyncClient, HTTPStatusError

from seele_scholar_agent.state import AgentState, PaperMetadata

from ..logging import get_logger

logger = get_logger(__name__)


class ArxivRetriever:
    BASE_URL = "https://export.arxiv.org/api/query"

    def __init__(self, top_k: int = 10):
        self.top_k = top_k

    async def search(self, query: str) -> list[PaperMetadata]:
        search_url = f"{self.BASE_URL}?search_query={query}&sortBy=relevance&sortOrder=descending&start=0&max_results={self.top_k}"

        for attempt in range(3):
            try:
                async with AsyncClient(timeout=30.0) as client:
                    response = await client.get(search_url)

                    if response.status_code == 429:
                        retry_after = int(response.headers.get("retry-after", 5))
                        logger.warning(
                            f"ArXiv rate limited, waiting {retry_after}s before retry..."
                        )
                        await asyncio.sleep(retry_after)
                        continue

                    if response.status_code != 200:
                        logger.error(f"ArXiv API error: {response.status_code}")
                        return []

                    return self._parse_response(response.text)

            except HTTPStatusError as e:
                if attempt < 2:
                    wait_time = 2**attempt * 3
                    logger.warning(f"ArXiv request failed, retrying in {wait_time}s... ({e}")
                    await asyncio.sleep(wait_time)
                    continue
                logger.error(f"ArXiv search failed after 3 attempts: {e}")
                return []
            except Exception as e:
                logger.error(f"ArXiv search failed: {e}")
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
                logger.warning(f"Failed to parse ArXiv entry: {e}")
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

    def _calculate_relevance(self, paper: dict) -> float:
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

        try:
            async with AsyncClient(timeout=30.0) as client:
                logger.info("OpenAlex searching...")
                response = await client.get(self.BASE_URL, params=params)

                if response.status_code != 200:
                    logger.error(f"OpenAlex API error: {response.status_code}")
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
            logger.error(f"OpenAlex search failed: {e}")
            return []


class SemanticScholarRetriever:
    BASE_URL = "https://api.semanticscholar.org/graph/v1"

    def __init__(self, api_key: str | None = None, top_k: int = 10):
        self.api_key = api_key
        self.top_k = top_k
        self.headers = {"x-api-key": api_key} if api_key else {}

    async def search(self, query: str, fields: list[str] | None = None):
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

        try:
            async with AsyncClient() as client:
                response = await client.get(url, params=params, headers=self.headers, timeout=30.0)
                if response.status_code != 200:
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
            logger.error("Semantic Scholar search failed", error=str(e))
            return []

    def _calculate_relevance(self, paper: dict) -> float:
        citation_count = paper.get("citationCount", 0)
        year = paper.get("year", 2020)
        citation_score = min(citation_count / 1000, 1.0)
        year_weight = 1.0 if year >= 2020 else 0.6
        return round(citation_score * 0.7 + year_weight * 0.3, 2)


class ResearcherNode:
    def __init__(
        self,
        qdrant_client=None,
        embedding_model=None,
        semantic_scholar_key: str | None = None,
        openalex_email: str | None = None,
    ):
        self.arxiv = ArxivRetriever()
        self.openalex = OpenAlexRetriever(email=openalex_email)
        self.ss = SemanticScholarRetriever(api_key=semantic_scholar_key)
        self.qdrant = qdrant_client
        self.embedding = embedding_model

    async def search(self, state: AgentState) -> dict[str, Any]:
        topic = state["topic"]
        all_papers = []

        logger.info(f"Searching OpenAlex for topic: {topic}")
        openalex_papers = await self.openalex.search(topic)
        all_papers.extend(openalex_papers)

        logger.info(f"Searching Semantic Scholar for topic: {topic}")
        ss_papers = await self.ss.search(topic)
        all_papers.extend(ss_papers)

        logger.info(f"Searching ArXiv for topic: {topic} (waiting 3s for rate limit)")
        await asyncio.sleep(3)
        arxiv_papers = await self.arxiv.search(topic)
        all_papers.extend(arxiv_papers)

        seen_ids = set()
        unique_papers = [
            p for p in all_papers if p.paper_id not in seen_ids and not seen_ids.add(p.paper_id)
        ]
        unique_papers.sort(key=lambda x: x.relevance_score, reverse=True)

        return {"papers": unique_papers, "status": "planning"}
