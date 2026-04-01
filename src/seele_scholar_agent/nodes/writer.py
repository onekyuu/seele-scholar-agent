from typing import Any

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from ..agent_config import PromptsConfig, RAGRetrieverFunc
from ..i18n import t
from ..logging import get_logger
from ..state import AgentState
from . import invoke_with_retry

logger = get_logger(__name__)


class WriterNode:
    def __init__(
        self, llm: ChatOpenAI, prompts: PromptsConfig, rag_retriever: RAGRetrieverFunc | None = None
    ):
        self.llm = llm
        self.prompts = prompts
        self.rag_retriever = rag_retriever
        self.prompt = ChatPromptTemplate.from_messages(
            [("system", prompts.writer_system_prompt), ("user", prompts.writer_user_prompt)]
        )
        self.chain = self.prompt | self.llm

    async def write(self, state: AgentState) -> dict[str, Any]:
        sections = state["sections"]
        current_index = state["current_section_index"]
        lang = state.get("language", "zh")

        if current_index >= len(sections):
            return {
                "status": "completed",
                "sections_completed": state.get("sections_completed", []),
            }

        section = sections[current_index]

        if section.status == "approved":
            return await self._move_to_next(state)

        logger.info("writing section", title=section.title, language=lang)

        if self.rag_retriever:
            search_query = f"{state['topic']} {section.title} {section.description}"
            rag_chunks = await self.rag_retriever(search_query)
            rag_context = self._build_rag_context(rag_chunks)
        else:
            rag_context = self._build_rag_context(state.get("rag_context"))

        outline_json = self._build_outline_json(state.get("outline"))
        review_comments = self._build_review_comments(section)

        try:
            result = await invoke_with_retry(
                self.chain,
                {
                    "topic": state["topic"],
                    "language": t(lang, "language_name"),
                    "section_title": section.title,
                    "section_description": section.description,
                    "outline_json": outline_json,
                    "rag_context": rag_context,
                    "review_comments": review_comments,
                },
            )

            content = result.content if hasattr(result, "content") else str(result)
            if isinstance(content, list):
                content = "\n".join(str(c) for c in content)
            content = self._clean_content(content)
        except Exception as e:
            logger.error("writing failed after retries", error=str(e))
            updated_sections = sections.copy()
            updated_sections[current_index] = section.model_copy(update={"status": "pending"})
            return {
                "sections": updated_sections,
                "status": "failed",
                "error_message": f"Writing section '{section.title}' failed: {e}",
            }

        updated_sections = sections.copy()
        updated_sections[current_index] = section.model_copy(
            update={
                "content": content,
                "status": "review",
                "revision_count": section.revision_count,
            }
        )

        return {"sections": updated_sections, "status": "reviewing"}

    def _build_outline_json(self, outline: Any) -> str:
        if not outline:
            return ""
        lines = [f"Title: {outline.title}", ""]
        for s in outline.sections:
            lines.append(f"- {s.title}: {s.description}")
        return "\n".join(lines)

    def _build_rag_context(self, rag_context: Any) -> str:
        if not rag_context:
            return "无"
        return "\n\n".join([c.content for c in rag_context[:5]])

    def _build_review_comments(self, section: Any) -> str:
        if not section.review_comments:
            return "无"
        return "\n".join([f"- {c}" for c in section.review_comments])

    async def _move_to_next(self, state: AgentState) -> dict[str, Any]:
        sections = state["sections"]
        index = state["current_section_index"]
        completed = state.get("sections_completed", [])
        completed.append(sections[index].title)

        if index + 1 >= len(sections):
            return {"sections_completed": completed, "status": "completed"}

        return {
            "sections_completed": completed,
            "current_section_index": index + 1,
            "status": "writing",
        }

    def _clean_content(self, content: str) -> str:
        lines = content.split("\n")
        clean = []
        in_code_block = False

        for line in lines:
            if line.strip().startswith("```"):
                in_code_block = not in_code_block
                continue
            if in_code_block:
                clean.append(line)
                continue
            if line.strip():
                clean.append(line)

        result = "\n".join(clean).strip()
        result = result.replace("]", "")
        return result
