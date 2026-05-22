from collections.abc import AsyncIterator
from typing import Any

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from ..agent_config import PromptsConfig, RAGRetrieverFunc
from ..i18n import t
from ..logging import get_logger
from ..state import AgentState, PaperMetadata, SectionDraft
from . import (
    PREVIOUS_SECTION_MAX_CHARS,
    SECTION_SUMMARY_MAX_CHARS,
    NodeStreamEvent,
    _stream_llm_text,
    invoke_with_retry,
)

logger = get_logger(__name__)


def _generate_section_summary(title: str, content: str) -> str:
    """Generate a compact summary (~150 tokens) of a written section for use as prior context.

    Uses a heuristic paragraph extraction — no extra LLM call needed.
    """
    if not content:
        return f"[{title}]\n(empty)"

    paragraphs = [p.strip() for p in content.split("\n\n") if p.strip()]
    if not paragraphs:
        snippet = content[:SECTION_SUMMARY_MAX_CHARS]
        if len(content) > SECTION_SUMMARY_MAX_CHARS:
            snippet += "..."
        return f"[{title}]\n{snippet}"

    parts: list[str] = []
    total_chars = 0
    for para in paragraphs:
        remaining = SECTION_SUMMARY_MAX_CHARS - total_chars
        if remaining <= 0:
            break
        if len(para) <= remaining:
            parts.append(para)
            total_chars += len(para)
        else:
            # End at a sentence boundary when possible
            snippet = para[:remaining]
            last_period = snippet.rfind(". ")
            if last_period > remaining // 2:
                snippet = snippet[: last_period + 1]
            else:
                snippet += "..."
            parts.append(snippet)
            break

    return f"[{title}]\n" + "\n\n".join(parts)


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

        section_summaries: list[str] = list(state.get("section_summaries") or [])
        previous_sections = self._build_previous_sections_context(
            sections, current_index, section_summaries
        )

        paper_summaries: list[str] = state.get("paper_summaries") or []
        numbered_papers = (
            self._build_numbered_papers_from_summaries(paper_summaries)
            if paper_summaries
            else self._build_numbered_papers(state.get("papers", []))
        )

        try:
            result = await invoke_with_retry(
                self.chain,
                {
                    "topic": state["topic"],
                    "language": t(lang, "language_name"),
                    "section_title": section.title,
                    "section_description": section.description,
                    "suggested_figures": self._build_suggested_figures(section, state),
                    "outline_json": outline_json,
                    "previous_sections": previous_sections,
                    "numbered_papers": numbered_papers,
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

        updated_summaries = list(state.get("section_summaries") or [])
        while len(updated_summaries) <= current_index:
            updated_summaries.append("")
        updated_summaries[current_index] = _generate_section_summary(section.title, content)

        return {
            "sections": updated_sections,
            "section_summaries": updated_summaries,
            "status": "reviewing",
        }

    async def astream(self, state: AgentState) -> AsyncIterator[NodeStreamEvent]:
        sections = state["sections"]
        current_index = state["current_section_index"]
        lang = state.get("language", "zh")

        if current_index >= len(sections):
            yield NodeStreamEvent(
                type="result",
                result={
                    "status": "completed",
                    "sections_completed": state.get("sections_completed", []),
                },
            )
            return

        section = sections[current_index]

        if section.status == "approved":
            result = await self._move_to_next(state)
            yield NodeStreamEvent(type="result", result=result)
            return

        yield NodeStreamEvent(type="progress", progress=f"writing:{section.title}")

        if self.rag_retriever:
            search_query = f"{state['topic']} {section.title} {section.description}"
            rag_chunks = await self.rag_retriever(search_query)
            rag_context = self._build_rag_context(rag_chunks)
        else:
            rag_context = self._build_rag_context(state.get("rag_context"))

        _section_summaries: list[str] = list(state.get("section_summaries") or [])
        _paper_summaries: list[str] = state.get("paper_summaries") or []

        input_data = {
            "topic": state["topic"],
            "language": t(lang, "language_name"),
            "section_title": section.title,
            "section_description": section.description,
            "suggested_figures": self._build_suggested_figures(section, state),
            "outline_json": self._build_outline_json(state.get("outline")),
            "previous_sections": self._build_previous_sections_context(
                sections, current_index, _section_summaries
            ),
            "numbered_papers": (
                self._build_numbered_papers_from_summaries(_paper_summaries)
                if _paper_summaries
                else self._build_numbered_papers(state.get("papers", []))
            ),
            "rag_context": rag_context,
            "review_comments": self._build_review_comments(section),
        }

        full_text = ""
        async for event in _stream_llm_text(self.chain, input_data):
            full_text += event.get("token", "")
            yield event

        content = self._clean_content(full_text)

        updated_sections = sections.copy()
        updated_sections[current_index] = section.model_copy(
            update={
                "content": content,
                "status": "review",
                "revision_count": section.revision_count,
            }
        )

        _updated_summaries = list(state.get("section_summaries") or [])
        while len(_updated_summaries) <= current_index:
            _updated_summaries.append("")
        _updated_summaries[current_index] = _generate_section_summary(section.title, content)

        yield NodeStreamEvent(
            type="result",
            result={
                "sections": updated_sections,
                "section_summaries": _updated_summaries,
                "status": "reviewing",
            },
        )

    def _build_suggested_figures(self, section: SectionDraft, state: AgentState) -> str:
        outline = state.get("outline")
        if not outline:
            return "无"
        for sec_outline in outline.sections:
            if sec_outline.title == section.title and sec_outline.suggested_figures:
                lines = [f"- {fig}" for fig in sec_outline.suggested_figures]
                return "\n".join(lines)
        return "无"

    def _build_outline_json(self, outline: Any) -> str:
        if not outline:
            return ""
        lines = [
            f"Title: {outline.title}",
            f"Paper type: {getattr(outline, 'paper_type', 'auto')}",
            f"Structure pattern: {getattr(outline, 'structure_pattern', 'auto')}",
            "",
        ]
        for s in outline.sections:
            lines.append(f"- {s.title}: {s.description}")
            purpose = getattr(s, "purpose", "")
            if purpose:
                lines.append(f"  Purpose: {purpose}")
            content_summary = getattr(s, "content_summary", "")
            if content_summary:
                lines.append(f"  Content summary: {content_summary}")
            target_claims = getattr(s, "target_claims", [])
            if target_claims:
                lines.append(f"  Target claims: {'; '.join(target_claims)}")
            key_sources = getattr(s, "key_sources", [])
            if key_sources:
                lines.append(f"  Key sources: {'; '.join(key_sources)}")
            evidence_gaps = getattr(s, "evidence_gaps", [])
            if evidence_gaps:
                lines.append(f"  Evidence gaps: {'; '.join(evidence_gaps)}")
            transition = getattr(s, "transition_to_next", "")
            if transition:
                lines.append(f"  Transition: {transition}")
        return "\n".join(lines)

    def _build_rag_context(self, rag_context: Any) -> str:
        if not rag_context:
            return "无"
        parts = []
        for c in rag_context[:5]:
            parts.append(f"[chunk_id:{c.chunk_id}]\n{c.content}")
        return "\n\n".join(parts)

    def _build_review_comments(self, section: Any) -> str:
        if not section.review_comments:
            return "无"
        return "\n".join([f"- {c}" for c in section.review_comments])

    def _build_previous_sections_context(
        self,
        sections: list[SectionDraft],
        current_index: int,
        section_summaries: list[str] | None = None,
    ) -> str:
        """Build context string for sections written before the current one.

        Prefers ``section_summaries`` (pre-generated compact summaries, ~150 tokens each)
        over full section content to keep the prompt lean.
        """
        if section_summaries is not None:
            prev = [s for s in section_summaries[:current_index] if s]
            if not prev:
                return "无"
            return "\n\n---\n\n".join(prev)

        # Fallback: build from section content (legacy path for states without summaries)
        completed = [
            s for s in sections[:current_index] if s.content and s.status in ("approved", "review")
        ]
        if not completed:
            return "无"
        parts = []
        for s in completed:
            snippet = s.content[:PREVIOUS_SECTION_MAX_CHARS]
            if len(s.content) > PREVIOUS_SECTION_MAX_CHARS:
                snippet += "..."
            parts.append(f"[{s.title}]\n{snippet}")
        return "\n\n".join(parts)

    def _build_numbered_papers(self, papers: list[PaperMetadata]) -> str:
        if not papers:
            return "无"
        lines = []
        for i, p in enumerate(papers, 1):
            authors_str = ", ".join(p.authors[:3])
            if len(p.authors) > 3:
                authors_str += " et al."
            abstract_snippet = p.abstract[:150] + "..." if len(p.abstract) > 150 else p.abstract
            lines.append(f"[{i}] {p.title} — {authors_str}. {abstract_snippet}")
        return "\n".join(lines)

    def _build_numbered_papers_from_summaries(self, paper_summaries: list[str]) -> str:
        """Use pre-built compact paper summaries from ResearcherNode (no abstract duplication)."""
        if not paper_summaries:
            return "无"
        return "\n".join(paper_summaries)

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

        return "\n".join(clean).strip()
