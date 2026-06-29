from typing import Any

from pydantic import BaseModel, Field

from ..budget import SectionBudget
from ..state import EvidencePacket, PaperMetadata, SectionStyleGuidance


class SectionBrief(BaseModel):
    section_id: str
    title: str
    order: int
    purpose: str = ""
    content_summary: str = ""


class OutlineContext(BaseModel):
    title: str
    abstract: str = ""
    keywords: list[str] = Field(default_factory=list)
    paper_type: str = "auto"
    structure_pattern: str = "auto"
    rationale: str = ""
    sections: list[SectionBrief] = Field(default_factory=list)


class SectionWritingSpec(BaseModel):
    section_id: str
    title: str
    order: int
    description: str = ""
    purpose: str = ""
    content_summary: str = ""
    key_points: list[str] = Field(default_factory=list)
    target_claims: list[str] = Field(default_factory=list)
    key_sources: list[str] = Field(default_factory=list)
    citation_plan: list[str] = Field(default_factory=list)
    evidence_gaps: list[str] = Field(default_factory=list)
    transition_to_next: str = ""
    suggested_figures: list[str] = Field(default_factory=list)
    section_style: SectionStyleGuidance = Field(default_factory=SectionStyleGuidance)
    budget: SectionBudget | None = None


class WriterInput(BaseModel):
    topic: str
    language: str
    outline_context: OutlineContext
    current_section: SectionWritingSpec
    previous_section_summaries: list[str] = Field(default_factory=list)
    citation_sources: list[Any] = Field(default_factory=list)
    papers: list[PaperMetadata] = Field(default_factory=list)
    evidence_packets: list[EvidencePacket] = Field(default_factory=list)
    review_comments: list[str] = Field(default_factory=list)
    style_context: str = ""
