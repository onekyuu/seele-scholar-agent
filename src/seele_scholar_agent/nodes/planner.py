from collections.abc import AsyncIterator
from typing import Any

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from ..agent_config import PromptsConfig
from ..i18n import t, t_list
from ..logging import get_logger
from ..state import (
    AgentState,
    OutlineStructure,
    PaperMetadata,
    SectionDraft,
    SectionEvidencePlan,
    SectionOutline,
)
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
        paper_type = str(state.get("paper_type", "auto"))
        structure_pattern = str(state.get("structure_pattern", "auto"))
        target_word_count = str(state.get("target_word_count", "auto"))

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
                },
            )
        except Exception as e:
            logger.error("LLM planning failed after retries", error=str(e))
            result = self._default_outline(topic, lang)

        outline = self._build_outline(result, topic, paper_type, structure_pattern)

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
        paper_type = str(state.get("paper_type", "auto"))
        structure_pattern = str(state.get("structure_pattern", "auto"))

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
            "target_word_count": str(state.get("target_word_count", "auto")),
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
            result = self._default_outline(topic, lang)

        outline = self._build_outline(result, topic, paper_type, structure_pattern)

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
