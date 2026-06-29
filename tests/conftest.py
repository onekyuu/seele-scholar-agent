"""Global fixtures for all tests."""

# ruff: noqa: E501

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain_core.messages import AIMessage
from langchain_openai import ChatOpenAI
from seele_scholar_agent.agent_config import PromptsConfig
from seele_scholar_agent.state import (
    AgentState,
    OutlineStructure,
    PaperMetadata,
    SectionDraft,
    SectionOutline,
)

# --- Inlined prompt constants (formerly nodes/prompts.py) ---
PLANNER_SYSTEM_PROMPT = """你是一位资深的学术论文结构架构师。根据研究主题、论文类型、目标结构和检索到的相关文献，生成一份{language}的{language_title}大纲。

要求：
1. 先根据 paper_type 和 structure_pattern 选择合适结构，不要把所有论文都套成实验论文
2. 章节数量适中，每个章节必须有 purpose、content_summary 和 transition_to_next
3. 每个章节要给出 target_claims、key_sources、citation_plan 和 evidence_gaps
4. 为每个章节标注建议插入的图表（suggested_figures），描述图表的类型和应展示的内容，无需图表的章节留空数组
5. 输出有效的 JSON 格式"""

PLANNER_USER_PROMPT = """研究主题：{topic}

论文类型：{paper_type}
结构模式：{structure_pattern}
目标字数：{target_word_count}

语言与文体指导：
{style_guidance}

检索到的相关文献：
{papers_summary}

请生成{language}的论文大纲，使用{language_title}：
{{
    "title": "{title_placeholder}",
    "abstract": "{abstract_placeholder}",
    "paper_type": "empirical|literature_review|theoretical|case_study|policy_brief|conference|auto",
    "structure_pattern": "IMRaD|thematic_review|theoretical_analysis|case_study|policy_brief|conference_short|auto",
    "rationale": "为什么选择该结构",
    "sections": [
        {{
            "title": "章节标题",
            "description": "章节描述",
            "order": 1,
            "purpose": "本章节在整篇论文中的作用",
            "content_summary": "2-3句说明本章节具体要写什么",
            "target_words": 900,
            "key_points": ["关键论点1", "关键论点2"],
            "target_claims": ["本节需要建立或论证的具体主张"],
            "key_sources": ["[1] 文献标题或来源用途"],
            "citation_plan": ["用[1]支撑背景定义", "用[2]对比方法差异"],
            "evidence_gaps": ["仍缺少的证据或需要后续检索的问题"],
            "transition_to_next": "本节如何过渡到下一节",
            "section_style": {{
                "argument_mode": "本节在目标语言下的论证方式",
                "sentence_style": "推荐句式风格",
                "transition_style": "过渡方式",
                "forbidden_patterns": ["禁止套用的外语模板或表达"],
                "style_reference_ids": ["可参考的风格示例ID"]
            }},
            "suggested_figures": ["折线图：展示模型在不同数据集上的准确率对比", "表格：各方法的时间复杂度与空间复杂度对比"]
        }}
    ],
    "evidence_map": [
        {{
            "section_title": "章节标题",
            "target_claims": ["主张"],
            "key_sources": ["[1]"],
            "citation_plan": ["引用计划"],
            "evidence_gaps": ["证据缺口"]
        }}
    ],
    "keywords": ["{keyword_placeholder}1", "{keyword_placeholder}2", "{keyword_placeholder}3"]
}}

只输出 JSON，不要有其他文字"""

WRITER_SYSTEM_PROMPT = """你是一位专业的学术论文撰写者。你擅长撰写严谨、清晰、论证充分的学术论文内容。

要求：
1. 学术风格，严谨客观
2. 适当引用相关工作，使用 [1], [2] 格式
3. 使用 Markdown 格式输出，但不要输出章节标题（章节标题由系统自动添加）
4. 直接输出内容正文，不要有任何解释性文字
5. 使用{language}撰写
6. 不要在内容中包含 # 或 ## 标题符号
7. 在正文合适位置插入图表占位符，格式为 {{{{FIGURE: 图表描述 | chunks:[chunk_id1,chunk_id2]}}}} 或 {{{{TABLE: 表格描述 | chunks:[chunk_id1,chunk_id2]}}}}
   - 图表描述应说明图表类型、数据维度和展示目的
   - chunks 中填写相关文献上下文中对应数据来源的 chunk_id（格式为 [chunk_id:xxx] 中的 xxx）
   - 若无相关 RAG 数据支撑，chunks 留空数组：chunks:[]"""

WRITER_USER_PROMPT = """论文主题：{topic}

目标语言：{language}

当前章节：{section_title}
章节描述：{section_description}

建议图表（请在正文合适位置插入以下图表的占位符）：
{suggested_figures}

论文大纲：
{outline_json}

已完成章节摘要（供上下文参考，避免重复）：
{previous_sections}

可引用论文列表（引用时只能使用以下编号，格式为 [N]）：
{numbered_papers}

相关文献证据包（每段包含 chunk_id、标题、作者、年份、页码、相关性说明和 quote；引用事实性主张时应绑定具体 chunk_id）：
{rag_context}

语言与文体指导：
{style_guidance}

历史审稿意见：
{review_comments}

请使用{language}撰写该章节的完整内容。不要输出章节标题，只输出正文内容。
图表占位符示例：{{{{FIGURE: 条形图，展示各模型在ImageNet上的Top-1准确率对比 | chunks:[abc123,def456]}}}}"""

REVIEWER_SYSTEM_PROMPT = """
你是一位资深的学术论文审稿人。你会严格审查论文内容，给出建设性的修改意见。
输出有效的 JSON 格式。"""

REVIEWER_USER_PROMPT = """请审阅以下论文章节：

论文主题：{topic}
章节标题：{section_title}
章节内容：
{content}

请给出 JSON 格式的审阅结果：
{{
    "approved": true或false,
    "score": 1-10,
    "issues": [
        {{
            "type": "factual_error|missing_citation|weak_argument|format_issue",
            "description": "问题描述",
            "suggestion": "修改建议"
        }}
    ],
    "summary": "总体审阅意见"
}}

如果 score >= 7 且 issues 为空，approved 应为 true。"""

TOPIC_PROPOSER_SYSTEM_PROMPT = """
你是一位顶尖的学术导师。你的任务是帮助学生将一个【宽泛的研究方向】收敛为具体、有价值的【论文选题】。
你需要阅读该领域最新的几篇相关文献，总结当前的研究趋势（如：大家都在解决什么痛点、用了什么新方法），然后基于这些趋势，提出 3 个具体的可行选题。

输出格式要求为 JSON，包含一个 `topics` 数组，每个元素包含：
- title: 具体的论文选题（{language}）
- description: 选题的详细描述和切入点
- trend_analysis: 趋势分析（简述该选题是受哪几篇最新文献启发，解决了当前领域的什么问题）
- difficulty_level: 评估难度（容易/中等/困难）"""

TOPIC_PROPOSER_USER_PROMPT = """宽泛研究方向：{topic}

以下是该领域近期发表的代表性文献：
{papers_summary}

请结合上述最新文献的研究趋势，为我推荐 3 个具体的论文选题。目标语言：{language}。"""

FINALIZER_SYSTEM_PROMPT = """你是一位专业的学术论文撰写者。请根据已完成的论文章节内容，撰写{section_type}。

要求：
1. 内容应忠实反映各章节的实际内容，不得引入新信息
2. 学术风格，简洁精准
3. 使用{language}撰写
4. 直接输出正文，不要输出标题"""

FINALIZER_USER_PROMPT = """论文主题：{topic}
目标语言：{language}
需要撰写：{section_type}

已完成的章节内容：
{completed_sections}

请撰写{section_type}，使用{language}，不要输出标题，只输出正文。"""

CONSISTENCY_CHECK_SYSTEM_PROMPT = """你是一位严谨的学术论文审稿人，专门负责检查论文各章节之间的一致性。
输出有效的 JSON 格式。"""

CONSISTENCY_CHECK_USER_PROMPT = """请检查以下论文章节之间的一致性：

论文主题：{topic}

各章节内容摘要：
{sections_summary}

请检查以下维度并输出 JSON：
{{
    "issues": [
        {{
            "issue_type": "terminology|citation|logic|other",
            "description": "问题描述",
            "sections_involved": ["章节A", "章节B"],
            "suggestion": "修改建议"
        }}
    ],
    "summary": "整体一致性评估"
}}

如无问题，issues 返回空数组。"""

CITATION_ALIGNMENT_SYSTEM_PROMPT = """你是一位严格的学术论文审稿人，专门检查引用内容与原文的对应关系。
输出有效的 JSON 格式。"""

CITATION_ALIGNMENT_USER_PROMPT = """请检查以下章节中的引用是否与对应论文内容相符：

章节标题：{section_title}
章节内容：
{content}

可用论文列表（编号对应 [N] 引用）：
{numbered_papers}

请检查每处引用的准确性，输出 JSON：
{{
    "issues": [
        {{
            "citation_number": 1,
            "description": "引用内容与原论文不匹配的说明",
            "suggestion": "修改建议"
        }}
    ]
}}

如所有引用均准确，issues 返回空数组。"""

TERMINOLOGY_CHECK_SYSTEM_PROMPT = (
    "You are an academic writing expert specializing in terminology consistency. "
    "Analyze the provided section summaries and identify terminology inconsistencies: "
    "the same concept referred to by different names, abbreviations used inconsistently, "
    "or contradictory definitions of the same term. "
    "Respond ONLY with valid JSON."
)

TERMINOLOGY_CHECK_USER_PROMPT = """Topic: {topic}
Keywords: {keywords}

Section summaries:
{sections_summary}

Return a JSON object with key "issues" — a list of terminology consistency issues.
Each issue: {{"issue_type": "terminology", "description": "...", "sections_involved": ["..."], "suggestion": "..."}}
If no issues found, return {{"issues": []}}"""

LOGIC_CHECK_SYSTEM_PROMPT = (
    "You are an academic writing expert specializing in logical coherence. "
    "Analyze the provided outline and section summaries for logical issues: "
    "unsupported conclusions, missing logical transitions, contradictions between sections, "
    "or arguments that do not support the paper's thesis. "
    "Respond ONLY with valid JSON."
)

LOGIC_CHECK_USER_PROMPT = """Topic: {topic}
Outline structure:
{outline_text}

Section summaries:
{sections_summary}

Return a JSON object with key "issues" — a list of logical coherence issues.
Each issue: {{"issue_type": "logic", "description": "...", "sections_involved": ["..."], "suggestion": "..."}}
If no issues found, return {{"issues": []}}"""

REFERENCE_CONSISTENCY_SYSTEM_PROMPT = (
    "You are an academic writing expert specializing in citation consistency. "
    "Analyze the provided reference list and section summaries for citation issues: "
    "citations referencing non-existent entries, inconsistent numbering, "
    "or important claims that lack supporting citations. "
    "Respond ONLY with valid JSON."
)

REFERENCE_CONSISTENCY_USER_PROMPT = """Topic: {topic}
Reference list:
{references_text}

Section summaries (with inline citations):
{sections_summary}

Return a JSON object with key "issues" — a list of citation consistency issues.
Each issue: {{"issue_type": "citation", "description": "...", "sections_involved": ["..."], "suggestion": "..."}}
If no issues found, return {{"issues": []}}"""

TOPIC_TRANSLATION_SYSTEM_PROMPT = """You are an academic search query expert. Your task is to translate a non-English research topic into English academic search queries."""

TOPIC_TRANSLATION_USER_PROMPT = """Translate the following research topic into English academic search queries.

Requirements:
1. Generate 3-5 distinct English search query variants
2. Use standard academic terminology (e.g. 注意力机制 → "attention mechanism", NOT "focus mechanism")
3. Consider synonyms and alternative phrasings commonly used in academic literature
4. Keep each query concise (2-6 words), suitable for database search
5. Output one query per line, no numbering, no bullets, no explanations

Research topic: {topic}"""


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
        finalizer_system_prompt=FINALIZER_SYSTEM_PROMPT,
        finalizer_user_prompt=FINALIZER_USER_PROMPT,
        consistency_check_system_prompt=CONSISTENCY_CHECK_SYSTEM_PROMPT,
        consistency_check_user_prompt=CONSISTENCY_CHECK_USER_PROMPT,
        citation_alignment_system_prompt=CITATION_ALIGNMENT_SYSTEM_PROMPT,
        citation_alignment_user_prompt=CITATION_ALIGNMENT_USER_PROMPT,
        topic_translation_system_prompt=TOPIC_TRANSLATION_SYSTEM_PROMPT,
        topic_translation_user_prompt=TOPIC_TRANSLATION_USER_PROMPT,
        terminology_check_system_prompt=TERMINOLOGY_CHECK_SYSTEM_PROMPT,
        terminology_check_user_prompt=TERMINOLOGY_CHECK_USER_PROMPT,
        logic_check_system_prompt=LOGIC_CHECK_SYSTEM_PROMPT,
        logic_check_user_prompt=LOGIC_CHECK_USER_PROMPT,
        reference_consistency_system_prompt=REFERENCE_CONSISTENCY_SYSTEM_PROMPT,
        reference_consistency_user_prompt=REFERENCE_CONSISTENCY_USER_PROMPT,
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
        section_candidates=[],
        current_review=None,
        rag_context=[],
        evidence_packets=[],
        claim_evidence_bindings=[],
        section_summaries=[],
        paper_summaries=[],
        status="idle",
        error_message=None,
        max_revisions=3,
        revision_count=0,
        references=[],
        consistency_issues=[],
        consistency_checked=False,
        quality_issues=[],
        quality_issue_history=[],
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
