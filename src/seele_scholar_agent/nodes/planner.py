from collections.abc import AsyncIterator
from typing import Any

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from ..agent_config import PromptsConfig
from ..i18n import t, t_list
from ..logging import get_logger
from ..state import AgentState, OutlineStructure, SectionDraft, SectionOutline
from . import NodeStreamEvent, _stream_llm_text, invoke_with_retry

logger = get_logger(__name__)


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

        papers_summary = "\n\n".join(
            [f"- **{p.title}** ({', '.join(p.authors[:3])})" for p in papers[:15]]
        ) or t(lang, "no_papers_found")

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
                },
            )
        except Exception as e:
            logger.error("LLM planning failed after retries", error=str(e))
            result = self._default_outline(topic, lang)

        outline = OutlineStructure(
            title=result.get("title", f"Research on {topic}"),
            abstract=result.get("abstract", ""),
            sections=[
                SectionOutline(
                    title=s.get("title", f"Section {i}"),
                    description=s.get("description", ""),
                    order=s.get("order", i),
                    key_points=s.get("key_points", []),
                )
                for i, s in enumerate(result.get("sections", []), 1)
            ],
            keywords=result.get("keywords", []),
        )

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

        papers_summary = "\n\n".join(
            [f"- **{p.title}** ({', '.join(p.authors[:3])})" for p in papers[:15]]
        ) or t(lang, "no_papers_found")

        input_data = {
            "topic": topic,
            "papers_summary": papers_summary,
            "language": t(lang, "language_name"),
            "language_title": t(lang, "language_title"),
            "title_placeholder": t(lang, "language_title"),
            "abstract_placeholder": t(lang, "language_abstract"),
            "keyword_placeholder": t(lang, "language_keywords"),
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

        outline = OutlineStructure(
            title=result.get("title", f"Research on {topic}"),
            abstract=result.get("abstract", ""),
            sections=[
                SectionOutline(
                    title=s.get("title", f"Section {i}"),
                    description=s.get("description", ""),
                    order=s.get("order", i),
                    key_points=s.get("key_points", []),
                )
                for i, s in enumerate(result.get("sections", []), 1)
            ],
            keywords=result.get("keywords", []),
        )

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
            }
            for i in range(len(sections_titles))
        ]
        return {"title": title, "sections": sections, "keywords": [topic]}
