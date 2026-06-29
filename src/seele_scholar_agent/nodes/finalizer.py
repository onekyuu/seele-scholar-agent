from collections.abc import AsyncIterator
from typing import Any

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from ..agent_config import PromptsConfig
from ..document_profile import is_research_proposal
from ..i18n import t
from ..logging import get_logger
from ..state import AgentState, SectionDraft
from . import NodeStreamEvent, _stream_llm_text, invoke_with_retry

logger = get_logger(__name__)

_ABSTRACT_TITLES = {"abstract", "摘要", "要旨"}
_CONCLUSION_TITLES = {"conclusion", "conclusions", "结论", "総括", "结语"}


def _is_abstract_section(title: str) -> bool:
    return title.strip().lower() in _ABSTRACT_TITLES


def _is_conclusion_section(title: str) -> bool:
    return title.strip().lower() in _CONCLUSION_TITLES


def _build_completed_sections_summary(sections: list[SectionDraft], max_chars: int = 800) -> str:
    parts = []
    for s in sections:
        if s.content and s.status in (
            "approved",
            "accepted_with_issues",
            "review",
            "auto_generated",
        ):
            snippet = s.content[:max_chars]
            if len(s.content) > max_chars:
                snippet += "..."
            parts.append(f"## {s.title}\n{snippet}")
    return "\n\n".join(parts) if parts else "无"


class FinalizerNode:
    def __init__(self, llm: ChatOpenAI, prompts: PromptsConfig):
        self.llm = llm
        self.prompts = prompts
        self.prompt = ChatPromptTemplate.from_messages(
            [
                ("system", prompts.finalizer_system_prompt),
                ("user", prompts.finalizer_user_prompt),
            ]
        )
        self.chain = self.prompt | self.llm

    async def finalize(self, state: AgentState) -> dict[str, Any]:
        sections = state.get("sections", [])
        lang = state.get("language", "zh")
        topic = state["topic"]
        if is_research_proposal(state):
            return {"status": "completed"}

        completed_summary = _build_completed_sections_summary(sections)
        updated_sections = list(sections)
        modified = False

        for i, section in enumerate(updated_sections):
            if section.status != "pending":
                continue

            if _is_abstract_section(section.title):
                section_type = t(lang, "language_abstract")
            elif _is_conclusion_section(section.title):
                if lang == "zh":
                    section_type = "结论"
                elif lang == "en":
                    section_type = "Conclusion"
                else:
                    section_type = "結論"
            else:
                continue

            logger.info("finalizing section", title=section.title, lang=lang)

            try:
                result = await invoke_with_retry(
                    self.chain,
                    {
                        "topic": topic,
                        "language": t(lang, "language_name"),
                        "section_type": section_type,
                        "completed_sections": completed_summary,
                    },
                )
                content = result.content if hasattr(result, "content") else str(result)
                if isinstance(content, list):
                    content = "\n".join(str(c) for c in content)
                updated_sections[i] = section.model_copy(
                    update={"content": content.strip(), "status": "auto_generated"}
                )
                modified = True
            except Exception as e:
                logger.error("finalizer failed", section=section.title, error=str(e))

        if not modified:
            return {"status": "completed"}

        return {"sections": updated_sections, "status": "completed"}

    async def astream(self, state: AgentState) -> AsyncIterator[NodeStreamEvent]:
        sections = state.get("sections", [])
        lang = state.get("language", "zh")
        topic = state["topic"]
        if is_research_proposal(state):
            yield NodeStreamEvent(type="result", result={"status": "completed"})
            return

        completed_summary = _build_completed_sections_summary(sections)
        updated_sections = list(sections)
        modified = False

        for i, section in enumerate(updated_sections):
            if section.status != "pending":
                continue
            if _is_abstract_section(section.title):
                section_type = t(lang, "language_abstract")
            elif _is_conclusion_section(section.title):
                if lang == "zh":
                    section_type = "结论"
                elif lang == "en":
                    section_type = "Conclusion"
                else:
                    section_type = "結論"
            else:
                continue

            yield NodeStreamEvent(type="progress", progress=f"finalizing:{section.title}")

            input_data = {
                "topic": topic,
                "language": t(lang, "language_name"),
                "section_type": section_type,
                "completed_sections": completed_summary,
            }

            full_text = ""
            try:
                async for event in _stream_llm_text(self.chain, input_data):
                    full_text += event.get("token", "")
                    yield event
                updated_sections[i] = section.model_copy(
                    update={"content": full_text.strip(), "status": "auto_generated"}
                )
                modified = True
            except Exception as e:
                logger.error("finalizer stream failed", section=section.title, error=str(e))

        final: dict[str, Any] = {"status": "completed"}
        if modified:
            final["sections"] = updated_sections

        yield NodeStreamEvent(type="result", result=final)
