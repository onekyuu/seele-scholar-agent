"""Sample AgentState snapshots for different workflow stages."""

from datetime import datetime

from seele_scholar_agent.state import (
    AgentState,
    OutlineStructure,
    PaperMetadata,
    SectionDraft,
    SectionOutline,
)


def make_base_state(topic: str = "Large Language Models", language: str = "zh") -> AgentState:
    return AgentState(
        thread_id="test-thread-001",
        topic=topic,
        language=language,  # type: ignore[arg-type]
        created_at=datetime.now(),
        tenant_id=None,
        broad_papers=[],
        proposed_topics=[],
        papers=[],
        search_queries=[],
        outline=None,
        outline_approved=False,
        sections=[],
        current_section_index=0,
        sections_completed=[],
        review_history=[],
        current_review=None,
        rag_context=[],
        status="idle",
        error_message=None,
        max_revisions=3,
        revision_count=0,
        references=[],
        consistency_issues=[],
        consistency_checked=False,
    )


SAMPLE_PAPERS = [
    PaperMetadata(
        paper_id="arxiv:2301.00001",
        title="Attention Is All You Need",
        authors=["Vaswani", "Shazeer"],
        abstract="We propose the Transformer architecture.",
        source="arxiv",
        relevance_score=0.95,
    ),
    PaperMetadata(
        paper_id="s2:abc123",
        title="BERT: Bidirectional Transformers",
        authors=["Devlin", "Chang"],
        abstract="We introduce BERT.",
        source="semantic_scholar",
        relevance_score=0.88,
    ),
]

SAMPLE_OUTLINE = OutlineStructure(
    title="Large Language Models: A Survey",
    abstract="This paper surveys LLMs.",
    sections=[
        SectionOutline(title="Introduction", description="Background", order=1, key_points=[]),
        SectionOutline(title="Related Work", description="Prior art", order=2, key_points=[]),
        SectionOutline(title="Conclusion", description="Summary", order=3, key_points=[]),
    ],
    keywords=["LLM", "NLP"],
)

SAMPLE_SECTIONS = [
    SectionDraft(
        section_id="section_0", title="Introduction", description="Background", order_index=1
    ),
    SectionDraft(
        section_id="section_1", title="Related Work", description="Prior art", order_index=2
    ),
    SectionDraft(section_id="section_2", title="Conclusion", description="Summary", order_index=3),
]
