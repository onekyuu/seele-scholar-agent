from collections.abc import AsyncIterator
from typing import Any

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from ..agent_config import PromptsConfig
from ..draft.models import coerce_draft_integration_state
from ..i18n import t, t_list
from ..logging import get_logger
from ..profiles import DocumentProfile, get_document_profile
from ..state import (
    AgentState,
    OutlineStructure,
    PaperMetadata,
    SectionDraft,
    SectionEvidencePlan,
    SectionOutline,
    SectionStyleGuidance,
)
from ..style_packs import build_planner_style_context
from . import NodeStreamEvent, _stream_llm_text, invoke_with_retry
from .material_registry import (
    annotate_paper_summaries,
    get_material_registry,
    material_policy_suffix,
)

logger = get_logger(__name__)

_PAPER_ABSTRACT_SNIPPET_CHARS = 220


class PlannerNode:
    def __init__(self, llm: ChatOpenAI, prompts: PromptsConfig):
        self.llm = llm
        self.prompts = prompts
        self.prompt = ChatPromptTemplate.from_messages(
            [
                ("system", self.prompts.planner_system_prompt),
                ("user", self.prompts.planner_user_prompt),
            ]
        )
        self.parser = JsonOutputParser()
        self.chain = self.prompt | self.llm | self.parser
        self.stream_chain = self.prompt | self.llm

    async def plan(self, state: AgentState) -> dict[str, Any]:
        topic = state["topic"]
        lang = state.get("language", "zh")
        papers = state.get("papers", [])
        paper_summaries: list[str] = state.get("paper_summaries") or []
        papers_summary = self._build_papers_summary(papers, paper_summaries, lang, state)
        profile = get_document_profile(state)
        paper_type = self._effective_paper_type(state, profile)
        structure_pattern = self._effective_structure_pattern(state, profile)
        target_word_count = str(
            profile.target_word_count(state) or state.get("target_word_count", "auto")
        )
        style_context = self._build_planner_context(
            state, paper_type, structure_pattern, target_word_count, profile
        )

        logger.info(
            "generating outline",
            topic=topic,
            language=lang,
            paper_count=len(papers),
        )
        try:
            result = await invoke_with_retry(
                self.chain,
                {
                    "topic": topic,
                    "papers_summary": papers_summary,
                    "language": t(lang, "language_name"),
                    "language_title": t(lang, "language_title"),
                    "title_placeholder": t(lang, "language_title"),
                    "abstract_placeholder": t(lang, "language_abstract"),
                    "keyword_placeholder": t(lang, "language_keywords"),
                    "paper_type": paper_type,
                    "structure_pattern": structure_pattern,
                    "target_word_count": target_word_count,
                    "style_guidance": style_context,
                },
            )
        except Exception as e:
            logger.error("LLM planning failed after retries", error=str(e))
            result = profile.default_outline(topic, lang) or self._default_outline(topic, lang)

        outline = self._build_outline(result, topic, paper_type, structure_pattern)
        outline = profile.normalize_outline(outline, topic)

        sections = [
            SectionDraft(
                section_id=f"section_{i}",
                title=s.title,
                description=s.description,
                order_index=s.order,
            )
            for i, s in enumerate(sorted(outline.sections, key=lambda x: x.order))
        ]

        logger.info("outline generated", topic=topic, section_count=len(outline.sections))

        return {
            "outline": outline,
            "sections": sections,
            "current_section_index": 0,
            "status": "waiting_human",
        }

    async def astream(self, state: AgentState) -> AsyncIterator[NodeStreamEvent]:
        topic = state["topic"]
        lang = state.get("language", "zh")
        papers = state.get("papers", [])
        paper_summaries: list[str] = state.get("paper_summaries") or []
        profile = get_document_profile(state)
        paper_type = self._effective_paper_type(state, profile)
        structure_pattern = self._effective_structure_pattern(state, profile)
        target_word_count = str(
            profile.target_word_count(state) or state.get("target_word_count", "auto")
        )
        style_context = self._build_planner_context(
            state, paper_type, structure_pattern, target_word_count, profile
        )

        input_data = {
            "topic": topic,
            "papers_summary": self._build_papers_summary(papers, paper_summaries, lang, state),
            "language": t(lang, "language_name"),
            "language_title": t(lang, "language_title"),
            "title_placeholder": t(lang, "language_title"),
            "abstract_placeholder": t(lang, "language_abstract"),
            "keyword_placeholder": t(lang, "language_keywords"),
            "paper_type": paper_type,
            "structure_pattern": structure_pattern,
            "target_word_count": target_word_count,
            "style_guidance": style_context,
        }

        yield NodeStreamEvent(type="progress", progress="generating_outline")

        full_text = ""
        async for event in _stream_llm_text(self.stream_chain, input_data):
            full_text += event.get("token", "")
            yield event

        try:
            result = self.parser.parse(full_text)
        except Exception as e:
            logger.error("LLM planning stream parse failed", error=str(e))
            result = profile.default_outline(topic, lang) or self._default_outline(topic, lang)

        outline = self._build_outline(result, topic, paper_type, structure_pattern)
        outline = profile.normalize_outline(outline, topic)

        sections = [
            SectionDraft(
                section_id=f"section_{i}",
                title=s.title,
                description=s.description,
                order_index=s.order,
            )
            for i, s in enumerate(sorted(outline.sections, key=lambda x: x.order))
        ]

        yield NodeStreamEvent(
            type="result",
            result={
                "outline": outline,
                "sections": sections,
                "current_section_index": 0,
                "status": "waiting_human",
            },
        )

    def _default_outline(self, topic: str, lang: str = "zh") -> dict[str, Any]:
        sections_titles = t_list(lang, "default_sections")
        section_descs = t_list(lang, "default_section_descs")
        title = t(lang, "default_paper_title", topic=topic)
        sections = [
            {
                "title": sections_titles[i],
                "description": section_descs[i],
                "order": i + 1,
                "key_points": [],
                "purpose": section_descs[i],
                "content_summary": section_descs[i],
                "target_claims": [],
                "key_sources": [],
                "evidence_gaps": ["LLM planning failed; evidence mapping needs review"],
                "citation_plan": [],
                "transition_to_next": "",
                "section_style": {},
            }
            for i in range(len(sections_titles))
        ]
        return {
            "title": title,
            "sections": sections,
            "keywords": [topic],
            "paper_type": "auto",
            "structure_pattern": "auto",
            "rationale": "Fallback outline generated after planner failure.",
        }

    def _effective_paper_type(self, state: AgentState, profile: DocumentProfile) -> str:
        requested = str(state.get("paper_type") or "auto")
        return profile.effective_paper_type(requested)

    def _effective_structure_pattern(
        self, state: AgentState, profile: DocumentProfile
    ) -> str:
        requested = str(state.get("structure_pattern") or "auto")
        return profile.effective_structure_pattern(requested)

    def _build_planner_context(
        self,
        state: AgentState,
        paper_type: str,
        structure_pattern: str,
        target_word_count: str,
        profile: DocumentProfile,
    ) -> str:
        context = build_planner_style_context(state, paper_type, structure_pattern)
        draft_context = self._build_draft_planner_context(state)
        if draft_context:
            context += "\n\n" + draft_context
        profile_context = profile.planner_context_suffix(target_word_count)
        if profile_context:
            context += "\n\n" + profile_context
        return context

    def _build_draft_planner_context(self, state: AgentState) -> str:
        draft_state = coerce_draft_integration_state(state.get("draft_integration"))
        if draft_state is None:
            return ""
        decision = draft_state.outline_decision
        lines = [
            "Draft integration context:",
            f"- User intent: {draft_state.existing_content.user_intent}",
            f"- Preserve mode: {draft_state.existing_content.preserve_policy.mode}",
        ]
        if decision is not None:
            lines.append(f"- Outline adaptation: {decision.action}")
            lines.extend(f"  reason: {reason}" for reason in decision.reasons)
        if draft_state.uncovered_requirements:
            lines.append(
                "- Unmapped draft segments: " + ", ".join(draft_state.uncovered_requirements)
            )
        if draft_state.conflicts:
            lines.append("- Draft conflicts:")
            lines.extend(f"  - {conflict}" for conflict in draft_state.conflicts)
        lines.append("- Draft segments:")
        for segment in draft_state.existing_content.segments[:8]:
            heading = f"{segment.detected_heading}: " if segment.detected_heading else ""
            snippet = segment.text[:220]
            if len(segment.text) > 220:
                snippet += "..."
            lines.append(f"  - [{segment.segment_id}] {segment.inferred_role}: {heading}{snippet}")
        return "\n".join(lines)

    def _build_papers_summary(
        self,
        papers: list[PaperMetadata],
        paper_summaries: list[str],
        lang: str,
        state: AgentState | None = None,
    ) -> str:
        if paper_summaries:
            registry = get_material_registry(state) if state is not None else None
            return "\n".join(annotate_paper_summaries(paper_summaries, papers, registry)[:15])
        if not papers:
            return t(lang, "no_papers_found")

        lines: list[str] = []
        for i, paper in enumerate(papers[:15], 1):
            authors = ", ".join(paper.authors[:3]) or "Unknown"
            if len(paper.authors) > 3:
                authors += " et al."
            abstract = paper.abstract.strip()
            if len(abstract) > _PAPER_ABSTRACT_SNIPPET_CHARS:
                abstract = abstract[:_PAPER_ABSTRACT_SNIPPET_CHARS] + "..."
            parts = [
                f"[{i}] {paper.title}",
                f"authors: {authors}",
                f"source: {paper.source}",
                f"relevance: {paper.relevance_score:.2f}",
            ]
            if abstract:
                parts.append(f"summary: {abstract}")
            registry = get_material_registry(state) if state is not None else None
            lines.append("; ".join(parts) + material_policy_suffix(paper, registry))
        return "\n".join(lines)

    def _build_outline(
        self,
        result: dict[str, Any],
        topic: str,
        requested_paper_type: str,
        requested_structure_pattern: str,
    ) -> OutlineStructure:
        raw_sections = result.get("sections", [])
        sections = [
            self._build_section_outline(section_data, i)
            for i, section_data in enumerate(raw_sections, 1)
            if isinstance(section_data, dict)
        ]
        evidence_map = self._build_evidence_map(result.get("evidence_map"), sections)

        return OutlineStructure(
            title=result.get("title", f"Research on {topic}"),
            abstract=result.get("abstract", ""),
            sections=sections,
            keywords=result.get("keywords", []),
            paper_type=result.get("paper_type") or requested_paper_type,
            structure_pattern=result.get("structure_pattern") or requested_structure_pattern,
            rationale=result.get("rationale", ""),
            evidence_map=evidence_map,
        )

    def _build_section_outline(self, raw: dict[str, Any], fallback_order: int) -> SectionOutline:
        return SectionOutline(
            title=raw.get("title", f"Section {fallback_order}"),
            description=raw.get("description", ""),
            order=raw.get("order", fallback_order),
            key_points=self._as_str_list(raw.get("key_points")),
            suggested_figures=self._as_str_list(raw.get("suggested_figures")),
            purpose=raw.get("purpose", ""),
            content_summary=raw.get("content_summary", ""),
            target_words=self._as_int_or_none(raw.get("target_words")),
            target_claims=self._as_str_list(raw.get("target_claims")),
            key_sources=self._as_str_list(raw.get("key_sources")),
            evidence_gaps=self._as_str_list(raw.get("evidence_gaps")),
            citation_plan=self._as_str_list(raw.get("citation_plan")),
            transition_to_next=raw.get("transition_to_next", ""),
            section_style=self._build_section_style(raw.get("section_style")),
        )

    def _build_section_style(self, raw: Any) -> SectionStyleGuidance:
        if not isinstance(raw, dict):
            return SectionStyleGuidance()
        return SectionStyleGuidance(
            argument_mode=str(raw.get("argument_mode") or ""),
            sentence_style=str(raw.get("sentence_style") or ""),
            transition_style=str(raw.get("transition_style") or ""),
            forbidden_patterns=self._as_str_list(raw.get("forbidden_patterns")),
            style_reference_ids=self._as_str_list(raw.get("style_reference_ids")),
        )

    def _build_evidence_map(
        self, raw: Any, sections: list[SectionOutline]
    ) -> list[SectionEvidencePlan]:
        if isinstance(raw, list):
            plans: list[SectionEvidencePlan] = []
            for item in raw:
                if not isinstance(item, dict):
                    continue
                plans.append(
                    SectionEvidencePlan(
                        section_title=str(item.get("section_title") or item.get("title") or ""),
                        target_claims=self._as_str_list(item.get("target_claims")),
                        key_sources=self._as_str_list(item.get("key_sources")),
                        evidence_gaps=self._as_str_list(item.get("evidence_gaps")),
                        citation_plan=self._as_str_list(item.get("citation_plan")),
                    )
                )
            if plans:
                return plans

        return [
            SectionEvidencePlan(
                section_title=section.title,
                target_claims=section.target_claims,
                key_sources=section.key_sources,
                evidence_gaps=section.evidence_gaps,
                citation_plan=section.citation_plan,
            )
            for section in sections
        ]

    def _as_str_list(self, value: Any) -> list[str]:
        if not isinstance(value, list):
            return []
        return [str(item) for item in value if item is not None]

    def _as_int_or_none(self, value: Any) -> int | None:
        if value is None:
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None
