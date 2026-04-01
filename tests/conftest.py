"""Global fixtures for all tests."""

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain_core.messages import AIMessage
from langchain_openai import ChatOpenAI

from seele_scholar_agent.agent_config import PromptsConfig
from seele_scholar_agent.nodes.prompts import (
    PLANNER_SYSTEM_PROMPT,
    PLANNER_USER_PROMPT,
    REVIEWER_SYSTEM_PROMPT,
    REVIEWER_USER_PROMPT,
    TOPIC_PROPOSER_SYSTEM_PROMPT,
    TOPIC_PROPOSER_USER_PROMPT,
    WRITER_SYSTEM_PROMPT,
    WRITER_USER_PROMPT,
)
from seele_scholar_agent.state import (
    AgentState,
    OutlineStructure,
    PaperMetadata,
    SectionDraft,
    SectionOutline,
)


@pytest.fixture(scope="session")
def mock_prompts() -> PromptsConfig:
    """Minimal valid PromptsConfig for testing."""
    return PromptsConfig(
        planner_system_prompt=PLANNER_SYSTEM_PROMPT,
        planner_user_prompt=PLANNER_USER_PROMPT,
        writer_system_prompt=WRITER_SYSTEM_PROMPT,
        writer_user_prompt=WRITER_USER_PROMPT,
        reviewer_system_prompt=REVIEWER_SYSTEM_PROMPT,
        reviewer_user_prompt=REVIEWER_USER_PROMPT,
        topic_proposer_system_prompt=TOPIC_PROPOSER_SYSTEM_PROMPT,
        topic_proposer_user_prompt=TOPIC_PROPOSER_USER_PROMPT,
    )


@pytest.fixture
def mock_llm() -> ChatOpenAI:
    """LLM stub that returns an empty JSON object by default."""
    llm = MagicMock(spec=ChatOpenAI)
    llm.ainvoke = AsyncMock(return_value=AIMessage(content="{}"))
    return llm


@pytest.fixture
def sample_papers() -> list[PaperMetadata]:
    return [
        PaperMetadata(
            paper_id="arxiv:2301.00001",
            title="Attention Is All You Need",
            authors=["Vaswani", "Shazeer"],
            abstract="We propose the Transformer architecture...",
            url="https://arxiv.org/abs/1706.03762",
            source="arxiv",
            relevance_score=0.95,
        ),
        PaperMetadata(
            paper_id="s2:abc123",
            title="BERT: Pre-training of Deep Bidirectional Transformers",
            authors=["Devlin", "Chang"],
            abstract="We introduce BERT...",
            source="semantic_scholar",
            relevance_score=0.88,
        ),
        PaperMetadata(
            paper_id="oa:W2100001",
            title="GPT-3: Language Models are Few-Shot Learners",
            authors=["Brown"],
            abstract="We show that scaling language models...",
            source="openalex",
            relevance_score=0.82,
        ),
    ]


@pytest.fixture
def base_state() -> AgentState:
    return AgentState(
        thread_id="test-thread-001",
        topic="Large Language Models",
        language="zh",
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
    )


@pytest.fixture
def state_with_papers(base_state: AgentState, sample_papers: list[PaperMetadata]) -> AgentState:
    return {**base_state, "papers": sample_papers, "status": "planning"}


@pytest.fixture
def sample_outline() -> OutlineStructure:
    return OutlineStructure(
        title="Large Language Models: A Survey",
        abstract="This paper surveys recent advances in LLMs...",
        sections=[
            SectionOutline(
                title="Introduction",
                description="Background",
                order=1,
                key_points=["motivation"],
            ),
            SectionOutline(
                title="Related Work",
                description="Prior art",
                order=2,
                key_points=["transformers"],
            ),
            SectionOutline(
                title="Conclusion",
                description="Summary",
                order=3,
                key_points=["future work"],
            ),
        ],
        keywords=["LLM", "Transformer", "NLP"],
    )


@pytest.fixture
def state_with_outline(
    state_with_papers: AgentState, sample_outline: OutlineStructure
) -> AgentState:
    sections = [
        SectionDraft(
            section_id=f"section_{i}",
            title=s.title,
            description=s.description,
            order_index=s.order,
        )
        for i, s in enumerate(sample_outline.sections)
    ]
    return {
        **state_with_papers,
        "outline": sample_outline,
        "sections": sections,
        "current_section_index": 0,
        "status": "writing",
    }


@pytest.fixture
def state_with_written_section(state_with_outline: AgentState) -> AgentState:
    sections = list(state_with_outline["sections"])
    sections[0] = sections[0].model_copy(
        update={"content": "This is the introduction content.", "status": "review"}
    )
    return {**state_with_outline, "sections": sections, "status": "reviewing"}
