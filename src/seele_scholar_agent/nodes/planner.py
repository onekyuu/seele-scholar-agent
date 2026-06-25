from collections.abc import AsyncIterator
from typing import Any

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from ..agent_config import PromptsConfig
from ..document_profile import get_target_word_count, is_research_proposal
from ..i18n import t, t_list
from ..logging import get_logger
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
        proposal_profile = is_research_proposal(state)
        paper_type = self._effective_paper_type(state)
        structure_pattern = self._effective_structure_pattern(state)
        target_word_count = str(
            get_target_word_count(state) or state.get("target_word_count", "auto")
        )
        style_context = self._build_planner_context(
            state, paper_type, structure_pattern, target_word_count
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
            result = (
                self._default_proposal_outline(topic)
                if proposal_profile
                else self._default_outline(topic, lang)
            )

        outline = self._build_outline(result, topic, paper_type, structure_pattern)
        if proposal_profile:
            outline = self._normalize_proposal_outline(outline, topic)

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
        proposal_profile = is_research_proposal(state)
        paper_type = self._effective_paper_type(state)
        structure_pattern = self._effective_structure_pattern(state)
        target_word_count = str(
            get_target_word_count(state) or state.get("target_word_count", "auto")
        )
        style_context = self._build_planner_context(
            state, paper_type, structure_pattern, target_word_count
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
            result = (
                self._default_proposal_outline(topic)
                if proposal_profile
                else self._default_outline(topic, lang)
            )

        outline = self._build_outline(result, topic, paper_type, structure_pattern)
        if proposal_profile:
            outline = self._normalize_proposal_outline(outline, topic)

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

    def _default_proposal_outline(self, topic: str) -> dict[str, Any]:
        sections = [
            {
                "title": "研究背景",
                "description": (
                    "申請審査に必要な背景、問題意識、研究テーマとの接続を"
                    "約350-450字で簡潔に述べる。"
                ),
                "order": 1,
                "purpose": "研究計画書の問題意識と研究動機を明確にする。",
                "content_summary": "背景、問題意識、申請者固有の関心を簡潔に説明する。",
                "target_words": 400,
                "key_points": ["研究背景", "問題意識", "研究テーマとの接続"],
                "target_claims": [],
                "key_sources": [],
                "citation_plan": [],
                "evidence_gaps": [],
                "transition_to_next": "この問題意識を受けて研究目的を定義する。",
                "section_style": {},
                "suggested_figures": [],
            },
            {
                "title": "先行研究と課題",
                "description": (
                    "主要な先行研究の位置づけと残された課題を約400-500字で述べる。"
                    "論文の関連研究章のように網羅せず、申請審査に必要な情報密度に絞る。"
                ),
                "order": 2,
                "purpose": "本研究が取り組む課題の位置づけを示す。",
                "content_summary": "先行研究の要点、限界、未解決課題を申請書向けに整理する。",
                "target_words": 450,
                "key_points": ["先行研究", "残された課題", "本研究の位置づけ"],
                "target_claims": [],
                "key_sources": [],
                "citation_plan": [],
                "evidence_gaps": [],
                "transition_to_next": "課題を受けて研究目的へ接続する。",
                "section_style": {},
                "suggested_figures": [],
            },
            {
                "title": "研究目的",
                "description": (
                    "本研究で明らかにしたいこと、研究対象、問いを約350-450字で"
                    "具体化する。期待成果は概要に留め、方法章へ接続する。"
                ),
                "order": 3,
                "purpose": "本研究が何を明らかにするかを示す。",
                "content_summary": "研究目的、研究質問、対象範囲を簡潔に述べる。",
                "target_words": 400,
                "key_points": ["研究目的", "研究質問", "対象範囲"],
                "target_claims": [],
                "key_sources": [],
                "citation_plan": [],
                "evidence_gaps": [],
                "transition_to_next": "目的達成のための方法と計画へ接続する。",
                "section_style": {},
                "suggested_figures": [],
            },
            {
                "title": "研究方法・計画",
                "description": (
                    "使用する資料・ツール・方法、分析や検証の進め方、修士段階での"
                    "実行可能性を約500-600字で述べる。複合タイトルの「計画」は"
                    "概要級でよく、論文の方法章のような実験 protocol、詳細な変数設計、"
                    "統計検定までは要求しない。"
                ),
                "order": 4,
                "purpose": "研究計画の方法と修士段階での実行可能性を示す。",
                "content_summary": "資料、方法、検証方針、概略スケジュールを申請書向けに述べる。",
                "target_words": 550,
                "key_points": ["資料・ツール", "分析・検証方針", "修士段階での実行可能性"],
                "target_claims": [],
                "key_sources": [],
                "citation_plan": [],
                "evidence_gaps": [],
                "transition_to_next": "方法と計画を踏まえて期待される成果へ接続する。",
                "section_style": {},
                "suggested_figures": [],
            },
            {
                "title": "期待される成果",
                "description": (
                    "期待される成果、申請上の価値、将来展望を約250-350字で述べる。"
                    "1-2個の具体的貢献でよく、論文の contribution section のような"
                    "網羅的展開は不要。"
                ),
                "order": 5,
                "purpose": "研究の意義と進学後の展望を締めくくる。",
                "content_summary": "成果、申請価値、将来展望を簡潔にまとめる。",
                "target_words": 300,
                "key_points": ["期待成果", "申請価値"],
                "target_claims": [],
                "key_sources": [],
                "citation_plan": [],
                "evidence_gaps": [],
                "transition_to_next": "",
                "section_style": {},
                "suggested_figures": [],
            },
        ]
        return {
            "title": f"{topic}に関する研究計画書",
            "abstract": "",
            "sections": sections,
            "keywords": [topic],
            "paper_type": "research_proposal",
            "structure_pattern": "research_proposal",
            "rationale": "Fallback research proposal outline for Japanese graduate admission.",
            "evidence_map": [],
        }

    def _effective_paper_type(self, state: AgentState) -> str:
        requested = str(state.get("paper_type") or "auto")
        if is_research_proposal(state) and requested == "auto":
            return "research_proposal"
        return requested

    def _effective_structure_pattern(self, state: AgentState) -> str:
        requested = str(state.get("structure_pattern") or "auto")
        if is_research_proposal(state) and requested == "auto":
            return "research_proposal"
        return requested

    def _build_planner_context(
        self,
        state: AgentState,
        paper_type: str,
        structure_pattern: str,
        target_word_count: str,
    ) -> str:
        context = build_planner_style_context(state, paper_type, structure_pattern)
        if not is_research_proposal(state):
            return context
        proposal_lines = [
            "",
            "Research proposal requirements:",
            "- Treat this as a Japanese graduate-school research proposal, not a paper.",
            "- Plan a lightweight 4-5 chapter structure for a complete 2000-3000 "
            "Japanese-character document.",
            "- Prefer separate, single-task titles such as 研究背景, 先行研究と課題, "
            "研究目的, 研究方法・計画, 期待される成果.",
            "- Avoid heavy dual-task titles. If a compound title is used, its description "
            "must say the second part only needs overview-level coverage, not a full "
            "paper-style subsection.",
            "- Include motivation, prior-work gap, research purpose/questions, method, "
            "feasibility/plan, and expected outcomes at application-review density.",
            "- Only explicit schedule/timeline sections must cover 1年次前期, 1年次後期, "
            "2年次前期, and 2年次後期.",
            "- Do not require citations for the applicant's own plan, intended work, "
            "timeline, deliverables, or future evaluation.",
            "- Use citations only for prior-work/background claims.",
            f"- Total target length: {target_word_count}. Allocate target_words per section.",
        ]
        return context + "\n" + "\n".join(proposal_lines)

    def _normalize_proposal_outline(
        self, outline: OutlineStructure, topic: str
    ) -> OutlineStructure:
        if not outline.sections:
            return self._build_outline(
                self._default_proposal_outline(topic),
                topic,
                "research_proposal",
                "research_proposal",
            )

        sections = [
            self._with_proposal_description_guard(section)
            for section in sorted(outline.sections, key=lambda section: section.order)
        ]
        evidence_map = [
            SectionEvidencePlan(
                section_title=section.title,
                target_claims=section.target_claims,
                key_sources=section.key_sources,
                evidence_gaps=section.evidence_gaps,
                citation_plan=section.citation_plan,
            )
            for section in sections
        ]
        return outline.model_copy(
            update={
                "paper_type": "research_proposal",
                "structure_pattern": "research_proposal",
                "sections": sections,
                "evidence_map": evidence_map,
            }
        )

    def _with_proposal_description_guard(self, section: SectionOutline) -> SectionOutline:
        compound_markers = ("・", "/", "／", "&", " and ", "、")
        title_lower = section.title.casefold()
        if not any(marker in title_lower for marker in compound_markers):
            return section
        guard = (
            " 複合タイトルの場合、第二要素は申請審査に必要な概要級のカバーでよく、"
            "論文本文の独立小節のような完全展開は要求しない。"
        )
        if "概要級" in section.description or "overview-level" in section.description:
            return section
        return section.model_copy(update={"description": section.description + guard})

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
