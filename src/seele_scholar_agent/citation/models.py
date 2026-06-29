from typing import Literal

from pydantic import BaseModel, Field

from ..state import PaperMetadata


class RetrievalDiagnostic(BaseModel):
    source: Literal["openalex", "semantic_scholar", "arxiv", "crossref", "user_library"]
    query: str
    status: Literal["ok", "empty", "rate_limited", "error", "disabled"]
    http_status: int | None = None
    result_count: int = 0
    message: str = ""


class SourceQuality(BaseModel):
    citable: bool = False
    metadata_verified: bool = False
    verification_source: Literal[
        "crossref",
        "openalex",
        "semantic_scholar",
        "arxiv",
        "local",
        "none",
    ] = "none"
    missing_fields: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class CitationSource(BaseModel):
    citation_id: int
    paper: PaperMetadata
    stable_url: str | None = None
    doi: str | None = None
    open_access_pdf_url: str | None = None
    source_quality: SourceQuality
