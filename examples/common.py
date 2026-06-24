from datetime import datetime
from os import getenv
from uuid import uuid4

from langchain_openai import ChatOpenAI
from seele_scholar_agent.agent_config import PromptsConfig
from seele_scholar_agent.config import settings
from seele_scholar_agent.state import AgentState


def build_model() -> ChatOpenAI:
    api_key = getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Set OPENAI_API_KEY before running this example.")

    return ChatOpenAI(
        model=getenv("OPENAI_MODEL", "gpt-4o-mini"),
        api_key=api_key,
        base_url=getenv("OPENAI_BASE_URL") or None,
        temperature=float(getenv("OPENAI_TEMPERATURE", "0.7")),
    )


def build_prompts() -> PromptsConfig:
    return PromptsConfig(
        planner_system_prompt=(
            "You are an academic structure architect. Select an appropriate "
            "document type and structure pattern, then generate a structured "
            "{language} outline as valid JSON."
        ),
        planner_user_prompt="""Topic: {topic}
Paper type: {paper_type}
Structure pattern: {structure_pattern}
Target word count: {target_word_count}

Style guidance:
{style_guidance}

Papers:
{papers_summary}

Return JSON:
{{
  "title": "{title_placeholder}",
  "abstract": "{abstract_placeholder}",
  "paper_type": "empirical|literature_review|theoretical|case_study|policy_brief|conference|research_proposal|auto",
  "structure_pattern": "IMRaD|thematic_review|theoretical_analysis|case_study|research_proposal|auto",
  "rationale": "Why this structure fits the topic and literature",
  "sections": [
    {{
      "title": "Section title",
      "description": "Section goal",
      "order": 1,
      "purpose": "Role of this section in the paper",
      "content_summary": "Two to three sentences describing what this section will cover",
      "target_words": 900,
      "key_points": ["point"],
      "target_claims": ["claim this section should establish"],
      "key_sources": ["[1] source title or intended use"],
      "citation_plan": ["Use [1] for background definitions"],
      "evidence_gaps": ["Evidence still needed"],
      "transition_to_next": "How this section leads into the next one",
      "section_style": {{
        "argument_mode": "How this section should argue in the target locale",
        "sentence_style": "Recommended sentence style",
        "transition_style": "Recommended transition style",
        "forbidden_patterns": ["Patterns to avoid"],
        "style_reference_ids": ["style reference id"]
      }},
      "suggested_figures": []
    }}
  ],
  "evidence_map": [
    {{
      "section_title": "Section title",
      "target_claims": ["claim"],
      "key_sources": ["[1]"],
      "citation_plan": ["citation purpose"],
      "evidence_gaps": ["gap"]
    }}
  ],
  "keywords": ["{keyword_placeholder}"]
}}""",
        writer_system_prompt=(
            "You are an academic writer. Write rigorous {language} prose or, when "
            "requested, concrete research-proposal prose. Use [N] "
            "citations only from the provided paper list. Do not output section headings."
        ),
        writer_user_prompt="""Topic: {topic}
Language: {language}
Section: {section_title}
Description: {section_description}

Suggested figures:
{suggested_figures}

Outline:
{outline_json}

Previous sections:
{previous_sections}

Papers:
{numbered_papers}

Evidence packets:
{rag_context}

Style guidance:
{style_guidance}

Review comments:
{review_comments}

Write the complete section body.""",
        reviewer_system_prompt=(
            "You are a strict academic reviewer. Respond only with valid JSON."
        ),
        reviewer_user_prompt="""Review this section.

Topic: {topic}
Section: {section_title}
Content:
{content}

Return JSON:
{{
  "approved": true,
  "score": 8,
  "issues": [],
  "summary": "Short review summary"
}}""",
        topic_proposer_system_prompt=(
            "You are an academic mentor. Propose concrete paper topics from a broad "
            "research direction. Respond only with valid JSON."
        ),
        topic_proposer_user_prompt="""Broad topic: {topic}

Recent papers:
{papers_summary}

Target language: {language}

Return JSON with a "topics" array. Each item must have title, description,
trend_analysis, and difficulty_level ("easy", "medium", or "hard").""",
        finalizer_system_prompt=(
            "You write paper abstracts or conclusions from completed sections. "
            "Do not introduce new information."
        ),
        finalizer_user_prompt="""Topic: {topic}
Language: {language}
Section type: {section_type}

Completed sections:
{completed_sections}

Write the {section_type} body only.""",
        consistency_check_system_prompt=(
            "You check cross-section consistency in academic writing. Respond only "
            "with valid JSON."
        ),
        consistency_check_user_prompt="""Topic: {topic}

Section summaries:
{sections_summary}

Return JSON with an "issues" array. Use issue_type terminology, citation,
logic, or other.""",
        citation_alignment_system_prompt=(
            "You verify whether inline citations match the cited papers. Respond "
            "only with valid JSON."
        ),
        citation_alignment_user_prompt="""Section: {section_title}
Content:
{content}

Numbered papers:
{numbered_papers}

Return JSON:
{{"issues": []}}""",
        topic_translation_system_prompt=(
            "You are an academic search query expert. Translate non-English topics "
            "into concise English academic search queries."
        ),
        topic_translation_user_prompt="""Topic: {topic}

Return 3-5 English search queries, one per line.""",
        terminology_check_system_prompt=(
            "You check terminology consistency across academic paper sections. "
            "Respond only with valid JSON."
        ),
        terminology_check_user_prompt="""Topic: {topic}
Keywords: {keywords}

Section summaries:
{sections_summary}

Return JSON:
{{"issues": []}}""",
        logic_check_system_prompt=(
            "You check logical coherence across academic paper sections. Respond "
            "only with valid JSON."
        ),
        logic_check_user_prompt="""Topic: {topic}
Outline:
{outline_text}

Section summaries:
{sections_summary}

Return JSON:
{{"issues": []}}""",
        reference_consistency_system_prompt=(
            "You check citation consistency against the generated reference list. "
            "Respond only with valid JSON."
        ),
        reference_consistency_user_prompt="""Topic: {topic}
References:
{references_text}

Section summaries:
{sections_summary}

Return JSON:
{{"issues": []}}""",
    )


def build_initial_state(topic: str, language: str = "zh") -> AgentState:
    return AgentState(
        thread_id=str(uuid4()),
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
        evidence_packets=[],
        claim_evidence_bindings=[],
        section_summaries=[],
        paper_summaries=[],
        status="idle",
        error_message=None,
        max_revisions=settings.MAX_REVISIONS,
        revision_count=0,
        references=[],
        consistency_issues=[],
        consistency_checked=False,
        quality_issues=[],
    )
