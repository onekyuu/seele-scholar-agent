import operator
from datetime import datetime
from typing import Annotated, Any, Literal, NotRequired

from pydantic import BaseModel, Field
from typing_extensions import TypedDict


class PaperMetadata(BaseModel):
    paper_id: str
    title: str
    authors: list[str]
    abstract: str
    url: str | None = None
    pdf_url: str | None = None
    relevance_score: float = 0.0
    source: Literal["arxiv", "semantic_scholar", "openalex", "user_library"] = "openalex"


class ProposedTopic(BaseModel):
    title: str
    description: str
    trend_analysis: str
    difficulty_level: Literal["easy", "medium", "hard"]


class DocumentChunk(BaseModel):
    chunk_id: str
    content: str
    source: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class SectionEvidencePlan(BaseModel):
    section_title: str
    target_claims: list[str] = Field(default_factory=list)
    key_sources: list[str] = Field(default_factory=list)
    evidence_gaps: list[str] = Field(default_factory=list)
    citation_plan: list[str] = Field(default_factory=list)


class SectionOutline(BaseModel):
    title: str
    description: str = ""
    order: int
    key_points: list[str] = Field(default_factory=list)
    suggested_figures: list[str] = Field(default_factory=list)
    purpose: str = ""
    content_summary: str = ""
    target_words: int | None = None
    target_claims: list[str] = Field(default_factory=list)
    key_sources: list[str] = Field(default_factory=list)
    evidence_gaps: list[str] = Field(default_factory=list)
    citation_plan: list[str] = Field(default_factory=list)
    transition_to_next: str = ""


class OutlineStructure(BaseModel):
    title: str
    abstract: str
    sections: list[SectionOutline] = Field(default_factory=list)
    keywords: list[str] = Field(default_factory=list)
    paper_type: str = "auto"
    structure_pattern: str = "auto"
    rationale: str = ""
    evidence_map: list[SectionEvidencePlan] = Field(default_factory=list)


class SectionDraft(BaseModel):
    section_id: str
    title: str
    description: str = ""
    content: str = ""
    order_index: int
    status: Literal["pending", "writing", "review", "approved", "auto_generated"] = "pending"
    revision_count: int = 0
    review_comments: list[str] = Field(default_factory=list)


class ReviewIssue(BaseModel):
    type: Literal[
        "factual_error",
        "missing_citation",
        "weak_argument",
        "format_issue",
        "citation_mismatch",
        "other",
    ]
    description: str
    suggestion: str
    location: str | None = None


class ReviewResult(BaseModel):
    approved: bool
    score: int = Field(ge=1, le=10)
    issues: list[ReviewIssue] = []
    summary: str


class ReferenceEntry(BaseModel):
    number: int
    paper_id: str
    title: str
    authors: list[str]
    year: int | None = None
    venue: str | None = None
    url: str | None = None
    doi: str | None = None
    formatted: str


class ConsistencyIssue(BaseModel):
    issue_type: Literal["terminology", "citation", "logic", "other"]
    description: str
    sections_involved: list[str]
    suggestion: str


class QualityIssue(BaseModel):
    code: str
    message: str
    severity: Literal["info", "warning", "error", "blocking"] = "error"
    location: str | None = None
    blocking: bool = False
    details: dict[str, Any] = Field(default_factory=dict)


class AgentState(TypedDict):
    thread_id: str
    topic: str
    paper_type: NotRequired[str]
    structure_pattern: NotRequired[str]
    target_word_count: NotRequired[int]
    broad_papers: list[PaperMetadata]
    proposed_topics: list[ProposedTopic]
    language: Literal["zh", "en", "ja"]
    created_at: datetime
    tenant_id: str | None

    papers: Annotated[list[PaperMetadata], operator.add]
    search_queries: Annotated[list[str], operator.add]
    outline: OutlineStructure | None
    outline_approved: bool
    sections: list[SectionDraft]
    current_section_index: int
    sections_completed: Annotated[list[str], operator.add]
    review_history: Annotated[list[dict[str, Any]], operator.add]
    current_review: ReviewResult | None
    rag_context: Annotated[list[DocumentChunk], operator.add]

    # Writer 写完每章后生成的摘要，按章节索引定位（positional list，非累加）
    # section_summaries[i] 对应 sections[i] 的内容摘要，约 150-200 tokens per item
    section_summaries: list[str]

    # Researcher 生成的精简文献摘要列表（每条 1-3 句话），与 papers 列表等长且一一对应
    # 用于 Writer/Planner 的上下文构建，避免把完整 abstract 塞入 LLM prompt
    paper_summaries: list[str]

    status: Literal[
        "idle",
        "researching",
        "planning",
        "writing",
        "reviewing",
        "finalizing",
        "checking_consistency",
        "waiting_human",
        "completed",
        "failed",
    ]
    error_message: str | None
    max_revisions: int
    revision_count: int

    references: list[ReferenceEntry]
    consistency_issues: list[ConsistencyIssue]
    consistency_checked: bool
    quality_issues: Annotated[list[QualityIssue], operator.add]
