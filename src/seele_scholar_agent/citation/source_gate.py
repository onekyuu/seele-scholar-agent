from typing import Any, Literal

from ..logging import get_logger
from ..state import AgentState, PaperMetadata, QualityIssue
from .models import CitationSource, RetrievalDiagnostic, SourceQuality

logger = get_logger(__name__)


class CitationSourceGateNode:
    """Normalize retrieved papers into stable citation sources."""

    async def build(self, state: AgentState) -> dict[str, Any]:
        papers = list(state.get("papers", []) or [])
        queries = list(state.get("search_queries", []) or [])
        citation_sources: list[CitationSource] = []
        diagnostics = _retrieval_diagnostics(papers, queries)
        rejected: list[dict[str, Any]] = []

        for paper in papers:
            source_quality = _source_quality(paper)
            if not source_quality.citable:
                rejected.append(
                    {
                        "paper_id": paper.paper_id,
                        "title": paper.title,
                        "missing_fields": source_quality.missing_fields,
                    }
                )
                continue
            citation_sources.append(
                CitationSource(
                    citation_id=len(citation_sources) + 1,
                    paper=paper,
                    stable_url=_stable_url(paper),
                    doi=paper.doi,
                    open_access_pdf_url=paper.pdf_url or None,
                    source_quality=source_quality,
                )
            )

        result: dict[str, Any] = {
            "citation_sources": citation_sources,
            "retrieval_diagnostics": diagnostics,
            "citation_source_diagnostics": {
                "candidate_count": len(papers),
                "citable_count": len(citation_sources),
                "rejected": rejected,
            },
        }
        if papers and not citation_sources:
            result["quality_issues"] = [
                QualityIssue(
                    code="INSUFFICIENT_CITABLE_SOURCES",
                    message="Retrieved papers did not include any citable sources.",
                    severity="warning",
                    location="citation_sources",
                    blocking=False,
                    details={"candidate_count": len(papers), "rejected": rejected},
                )
            ]

        logger.info(
            "citation source gate completed",
            candidate_count=len(papers),
            citable_count=len(citation_sources),
        )
        return result


def _source_quality(paper: PaperMetadata) -> SourceQuality:
    missing_fields: list[str] = []
    warnings: list[str] = []
    if not paper.title:
        missing_fields.append("title")
    if not paper.authors:
        missing_fields.append("authors")
    if paper.year is None:
        missing_fields.append("year")
    if not (paper.doi or _stable_url(paper)):
        missing_fields.append("stable_identifier")

    verification_source = _verification_source(paper)
    metadata_verified = verification_source in {"openalex", "semantic_scholar", "arxiv"}
    if paper.source == "arxiv" and not paper.doi:
        warnings.append("arxiv source without DOI; using arXiv URL as stable identifier")

    return SourceQuality(
        citable=not missing_fields,
        metadata_verified=metadata_verified,
        verification_source=verification_source,
        missing_fields=missing_fields,
        warnings=warnings,
    )


def _stable_url(paper: PaperMetadata) -> str | None:
    if paper.doi:
        return f"https://doi.org/{paper.doi}"
    if paper.url:
        return paper.url
    if paper.pdf_url:
        return paper.pdf_url
    if paper.paper_id.startswith("arxiv:"):
        return f"https://arxiv.org/abs/{paper.paper_id.removeprefix('arxiv:')}"
    if paper.source == "arxiv" and "arxiv.org" in paper.paper_id:
        return paper.paper_id
    return None


def _verification_source(
    paper: PaperMetadata,
) -> Literal["crossref", "openalex", "semantic_scholar", "arxiv", "local", "none"]:
    if paper.doi and paper.source == "openalex":
        return "openalex"
    if paper.doi and paper.source == "semantic_scholar":
        return "semantic_scholar"
    if paper.source == "arxiv" and _stable_url(paper):
        return "arxiv"
    if paper.source == "user_library":
        return "local"
    return "none"


def _retrieval_diagnostics(
    papers: list[PaperMetadata], queries: list[str]
) -> list[RetrievalDiagnostic]:
    if not queries:
        queries = [""]
    diagnostics: list[RetrievalDiagnostic] = []
    for source in ("openalex", "semantic_scholar", "arxiv", "user_library"):
        count = sum(1 for paper in papers if paper.source == source)
        for query in queries:
            diagnostics.append(
                RetrievalDiagnostic(
                    source=source,
                    query=query,
                    status="ok" if count else "empty",
                    result_count=count,
                )
            )
    return diagnostics
