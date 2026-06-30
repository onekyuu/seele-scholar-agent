import asyncio
import re
from collections.abc import AsyncIterator
from typing import Any, Literal

from ..citation import CitationSource
from ..logging import get_logger
from ..profiles import get_document_profile
from ..state import AgentState, PaperMetadata, QualityIssue, ReferenceEntry
from ..tools.crossref import CrossRefMetadata, extract_doi_from_url, fetch_metadata
from . import CITATION_PATTERN, NodeStreamEvent

logger = get_logger(__name__)

_AUTHOR_YEAR_RE = re.compile(r"(\d{4})")
_CROSSREF_CONCURRENCY = 5


def _extract_year_from_paper(paper: PaperMetadata) -> int | None:
    for field in (paper.url or "", paper.abstract or ""):
        m = _AUTHOR_YEAR_RE.search(field)
        if m:
            year = int(m.group(1))
            if 1900 <= year <= 2100:
                return year
    return None


def _format_authors(authors: list[str], max_authors: int = 3) -> str:
    if not authors:
        return "Unknown"
    if len(authors) <= max_authors:
        return ", ".join(authors)
    return ", ".join(authors[:max_authors]) + " et al."


def _format_reference(entry: ReferenceEntry) -> str:
    authors_str = _format_authors(entry.authors)
    year_str = f" ({entry.year})" if entry.year else ""
    venue_str = f". {entry.venue}" if entry.venue else ""
    url_str = f". {entry.url}" if entry.url else ""
    return f"[{entry.number}] {authors_str}{year_str}. {entry.title}{venue_str}{url_str}"


def _collect_cited_numbers(sections_content: list[str]) -> set[int]:
    cited: set[int] = set()
    for content in sections_content:
        for m in CITATION_PATTERN.finditer(content):
            cited.add(int(m.group(1)))
    return cited


async def _enrich_from_crossref(
    paper: PaperMetadata,
    semaphore: asyncio.Semaphore,
) -> CrossRefMetadata | None:
    doi = paper.doi
    if paper.url:
        doi = doi or extract_doi_from_url(paper.url)
    if not doi and paper.pdf_url:
        doi = extract_doi_from_url(paper.pdf_url)
    if not doi:
        return None

    async with semaphore:
        return await fetch_metadata(doi)


class ReferenceGeneratorNode:
    async def generate(self, state: AgentState) -> dict[str, Any]:
        papers = state.get("papers", [])
        citation_sources = _citation_sources_from_state(state)
        sections = state.get("sections", [])

        if not papers and not citation_sources:
            logger.warning("no papers available for reference generation")
            return {"references": [], "status": "completed"}

        cited_numbers = _collect_cited_numbers([s.content for s in sections if s.content])
        if not cited_numbers:
            logger.warning("no inline citations found; refusing to generate full references")
            document_profile = get_document_profile(state)
            if document_profile.allow_empty_references:
                quality_issue = document_profile.empty_reference_issue()
                return {
                    "references": [],
                    "quality_issues": [quality_issue] if quality_issue is not None else [],
                    "status": "completed",
                }
            quality_issue = QualityIssue(
                code="NO_INLINE_CITATIONS",
                message=(
                    "No inline citations were found in the generated sections; "
                    "reference generation was skipped."
                ),
                severity="blocking",
                location="references",
                blocking=True,
            )
            return {
                "references": [],
                "quality_issues": [quality_issue],
                "status": "completed",
            }

        target_sources = _target_citation_sources(cited_numbers, citation_sources)
        if citation_sources:
            return await self._generate_from_citation_sources(target_sources)

        target_papers = _target_papers(cited_numbers, papers)

        semaphore = asyncio.Semaphore(_CROSSREF_CONCURRENCY)
        crossref_results = await asyncio.gather(
            *[_enrich_from_crossref(paper, semaphore) for _, paper in target_papers]
        )

        entries: list[ReferenceEntry] = []
        for (num, paper), cr in zip(target_papers, crossref_results, strict=True):
            verification_source: Literal["crossref", "openalex", "local", "none"]
            if cr is not None:
                year = cr.year
                venue = cr.venue
                authors = cr.authors if cr.authors else paper.authors
                doi = cr.doi or None
                metadata_verified = True
                verification_source = "crossref"
            else:
                year = paper.year or _extract_year_from_paper(paper)
                venue = paper.venue
                authors = paper.authors
                doi = paper.doi or (extract_doi_from_url(paper.url) if paper.url else None)
                metadata_verified = bool(doi and paper.source == "openalex")
                verification_source = "openalex" if metadata_verified else "local"

            entry = ReferenceEntry(
                number=num,
                paper_id=paper.paper_id,
                title=paper.title,
                authors=authors,
                year=year,
                venue=venue,
                url=paper.url,
                doi=doi,
                metadata_verified=metadata_verified,
                verification_source=verification_source,
                formatted="",
            )
            entry = entry.model_copy(update={"formatted": _format_reference(entry)})
            entries.append(entry)

        logger.info("references generated", count=len(entries))
        return {"references": entries, "status": "completed"}

    async def _generate_from_citation_sources(
        self, target_sources: list[tuple[int, CitationSource]]
    ) -> dict[str, Any]:
        semaphore = asyncio.Semaphore(_CROSSREF_CONCURRENCY)
        crossref_results = await asyncio.gather(
            *[_enrich_from_crossref(source.paper, semaphore) for _, source in target_sources]
        )

        entries: list[ReferenceEntry] = []
        for (num, source), cr in zip(target_sources, crossref_results, strict=True):
            paper = source.paper
            verification_source: Literal["crossref", "openalex", "local", "none"]
            if cr is not None:
                year = cr.year
                venue = cr.venue
                authors = cr.authors if cr.authors else paper.authors
                doi = cr.doi or source.doi
                metadata_verified = True
                verification_source = "crossref"
            else:
                year = paper.year or _extract_year_from_paper(paper)
                venue = paper.venue
                authors = paper.authors
                doi = source.doi or paper.doi
                metadata_verified = source.source_quality.metadata_verified
                verification_source = _reference_verification_source(source)

            entry = ReferenceEntry(
                number=num,
                paper_id=paper.paper_id,
                title=paper.title,
                authors=authors,
                year=year,
                venue=venue,
                url=source.stable_url or paper.url,
                doi=doi,
                metadata_verified=metadata_verified,
                verification_source=verification_source,
                formatted="",
            )
            entry = entry.model_copy(update={"formatted": _format_reference(entry)})
            entries.append(entry)

        logger.info("references generated from citation sources", count=len(entries))
        return {"references": entries, "status": "completed"}

    async def astream(self, state: AgentState) -> AsyncIterator[NodeStreamEvent]:
        yield NodeStreamEvent(type="progress", progress="generating_references")
        result = await self.generate(state)
        yield NodeStreamEvent(type="result", result=result)


def _citation_sources_from_state(state: AgentState) -> list[CitationSource]:
    sources: list[CitationSource] = []
    for item in state.get("citation_sources", []) or []:
        if isinstance(item, CitationSource):
            sources.append(item)
        elif isinstance(item, dict):
            sources.append(CitationSource(**item))
    return sources


def _target_citation_sources(
    cited_numbers: set[int], citation_sources: list[CitationSource]
) -> list[tuple[int, CitationSource]]:
    if not citation_sources:
        return []
    by_id = {source.citation_id: source for source in citation_sources}
    target_sources: list[tuple[int, CitationSource]] = []
    for num in sorted(cited_numbers):
        source = by_id.get(num)
        if source is None:
            logger.warning("citation source id out of range", number=num, total=len(by_id))
            continue
        target_sources.append((num, source))
    return target_sources


def _target_papers(
    cited_numbers: set[int], papers: list[PaperMetadata]
) -> list[tuple[int, PaperMetadata]]:
    target_papers: list[tuple[int, PaperMetadata]] = []
    for num in sorted(cited_numbers):
        idx = num - 1
        if idx < 0 or idx >= len(papers):
            logger.warning("citation number out of range", number=num, total=len(papers))
            continue
        target_papers.append((num, papers[idx]))
    return target_papers


def _reference_verification_source(
    source: CitationSource,
) -> Literal["openalex", "local", "none"]:
    verification_source = source.source_quality.verification_source
    if verification_source == "openalex":
        return "openalex"
    if verification_source in {"arxiv", "semantic_scholar", "local"}:
        return "local"
    return "none"
