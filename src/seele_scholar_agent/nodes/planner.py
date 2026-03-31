from typing import Any

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from ..agent_config import PromptsConfig
from ..logging import get_logger
from ..state import AgentState, OutlineStructure, SectionDraft, SectionOutline

logger = get_logger(__name__)


class PlannerNode:
    def __init__(self, model: ChatOpenAI, prompts: PromptsConfig):
        self.model = model
        self.prompts = prompts
        self.prompt = ChatPromptTemplate.from_messages(
            [
                ("system", self.prompts.planner_system_prompt),
                ("user", self.prompts.planner_user_prompt),
            ]
        )
        self.parser = JsonOutputParser()
        self.chain = self.prompt | self.model | self.parser

    async def plan(self, state: AgentState) -> dict[str, Any]:
        topic = state["topic"]
        lang = state.get("language", "zh")
        papers = state.get("papers", [])

        papers_summary = (
            "\n\n".join([f"- **{p.title}** ({', '.join(p.authors[:3])})" for p in papers[:15]])
            or "无相关文献"
        )

        logger.info(
            "Generating outline with planner",
            topic=topic,
            language=lang,
            paper_count=len(papers),
        )
        try:
            result = await self.chain.ainvoke(
                {
                    "topic": topic,
                    "papers_summary": papers_summary,
                    "language": self.prompts.language_names.get(lang, "中文"),
                    "language_title": self.prompts.language_titles.get(lang, "论文标题"),
                    "title_placeholder": self.prompts.language_titles.get(lang, "论文标题"),
                    "abstract_placeholder": self.prompts.language_abstract.get(lang, "摘要"),
                    "keyword_placeholder": self.prompts.language_keywords.get(lang, "关键词"),
                }
            )
        except Exception as e:
            logger.error(f"LLM planning failed: {e}")
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

        logger.info(f"Generated outline with {len(outline.sections)} sections", topic=topic)

        return {
            "outline": outline,
            "sections": sections,
            "current_section_index": 0,
            "status": "waiting_human",
        }

    def _default_outline(self, topic: str, lang: str = "zh") -> dict:
        """LLM 调用失败时的兜底大纲"""
        titles = {
            "zh": {
                "title": f"关于 {topic} 的研究",
                "sections": ["引言", "相关工作", "方法", "实验", "结论"],
            },
            "en": {
                "title": f"Research on {topic}",
                "sections": [
                    "Introduction",
                    "Related Work",
                    "Methodology",
                    "Experiment",
                    "Conclusion",
                ],
            },
            "ja": {
                "title": f"{topic}に関する研究",
                "sections": ["序論", "関連研究", "方法", "実験", "結論"],
            },
        }
        t = titles.get(lang, titles["zh"])
        sections = [
            {"title": t["sections"][0], "description": "研究背景", "order": 1, "key_points": []},
            {"title": t["sections"][1], "description": "文献综述", "order": 2, "key_points": []},
            {"title": t["sections"][2], "description": "提出方法", "order": 3, "key_points": []},
            {"title": t["sections"][3], "description": "实验结果", "order": 4, "key_points": []},
            {"title": t["sections"][4], "description": "总结", "order": 5, "key_points": []},
        ]
        return {"title": t["title"], "sections": sections, "keywords": [topic]}
