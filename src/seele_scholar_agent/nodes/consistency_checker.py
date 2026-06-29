import asyncio
from collections.abc import AsyncIterator
from typing import Any

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from ..agent_config import PromptsConfig
from ..logging import get_logger
from ..state import AgentState, ConsistencyIssue, SectionDraft
from . import NodeStreamEvent, invoke_with_retry

logger = get_logger(__name__)

_SUMMARY_MAX_CHARS = 600


def _build_sections_summary_from_state(
    sections: list[SectionDraft],
    section_summaries: list[str],
) -> str:
    """Build a combined sections summary string for LLM context.

    Prefers pre-generated ``section_summaries`` from WriterNode (compact, ~150 tokens each).
    Falls back to truncated section content for backward compatibility.
    """
    parts: list[str] = []
    for i, section in enumerate(sections):
        if section.status not in ("approved", "accepted_with_issues", "auto_generated"):
            continue
        if i < len(section_summaries) and section_summaries[i]:
            parts.append(section_summaries[i])
        elif section.content:
            snippet = section.content[:_SUMMARY_MAX_CHARS]
            if len(section.content) > _SUMMARY_MAX_CHARS:
                snippet += "..."
            parts.append(f"[{section.title}]\n{snippet}")
    return "\n\n---\n\n".join(parts) if parts else "无"


class ConsistencyCheckerNode:
    def __init__(self, llm: ChatOpenAI, prompts: PromptsConfig):
        self.llm = llm
        self.prompts = prompts
        self.parser = JsonOutputParser()

    async def check(self, state: AgentState) -> dict[str, Any]:
        sections = state.get("sections", [])
        approved = [
            s
            for s in sections
            if s.status in ("approved", "accepted_with_issues", "auto_generated")
        ]

        if len(approved) < 2:
            logger.info("not enough approved sections for consistency check")
            return {"consistency_checked": True, "consistency_issues": []}

        section_summaries: list[str] = list(state.get("section_summaries") or [])
        sections_summary = _build_sections_summary_from_state(sections, section_summaries)

        outline = state.get("outline")
        references = state.get("references", [])
        topic = state["topic"]

        results = await asyncio.gather(
            self._check_terminology(topic, outline, sections_summary),
            self._check_logic(topic, outline, sections_summary),
            self._check_citations(topic, references, sections_summary),
            return_exceptions=True,
        )

        all_issues: list[ConsistencyIssue] = []
        check_names = ("terminology", "logic", "citation")
        for name, result in zip(check_names, results, strict=True):
            if isinstance(result, BaseException):
                logger.error("consistency sub-check failed", check=name, error=str(result))
            elif isinstance(result, list):
                all_issues.extend(result)

        logger.info("consistency check completed", issues_found=len(all_issues))
        return {"consistency_checked": True, "consistency_issues": all_issues}

    async def astream(self, state: AgentState) -> AsyncIterator[NodeStreamEvent]:
        sections = state.get("sections", [])
        approved = [
            s
            for s in sections
            if s.status in ("approved", "accepted_with_issues", "auto_generated")
        ]

        if len(approved) < 2:
            yield NodeStreamEvent(
                type="result",
                result={"consistency_checked": True, "consistency_issues": []},
            )
            return

        yield NodeStreamEvent(type="progress", progress="checking_terminology")
        yield NodeStreamEvent(type="progress", progress="checking_logic")
        yield NodeStreamEvent(type="progress", progress="checking_citations")

        result = await self.check(state)
        yield NodeStreamEvent(type="result", result=result)

    def _build_outline_context(self, outline: Any) -> str:
        if not outline:
            return "无"
        lines = [f"Title: {outline.title}"]
        keywords = getattr(outline, "keywords", [])
        if keywords:
            lines.append(f"Keywords: {', '.join(keywords)}")
        lines.append("")
        for s in getattr(outline, "sections", []):
            lines.append(f"- {s.title}: {getattr(s, 'description', '')}")
        return "\n".join(lines)

    def _build_references_context(self, references: list[Any]) -> str:
        if not references:
            return "无"
        lines: list[str] = []
        for ref in references:
            formatted = getattr(ref, "formatted", None)
            lines.append(formatted if formatted else str(ref))
        return "\n".join(lines)

    async def _run_sub_check(
        self,
        system_prompt: str,
        user_prompt: str,
        input_data: dict[str, Any],
        fallback_issue_type: str,
    ) -> list[ConsistencyIssue]:
        prompt = ChatPromptTemplate.from_messages(
            [("system", system_prompt), ("user", user_prompt)]
        )
        chain = prompt | self.llm | self.parser
        try:
            result = await invoke_with_retry(chain, input_data)
            raw_issues = result.get("issues", []) if isinstance(result, dict) else []
            issues: list[ConsistencyIssue] = []
            for item in raw_issues:
                try:
                    issues.append(
                        ConsistencyIssue(
                            issue_type=item.get("issue_type", fallback_issue_type),
                            description=item.get("description", ""),
                            sections_involved=item.get("sections_involved", []),
                            suggestion=item.get("suggestion", ""),
                        )
                    )
                except Exception as e:
                    logger.warning("skipping malformed issue", error=str(e), item=item)
            return issues
        except Exception as e:
            logger.error("sub-check invocation failed", check=fallback_issue_type, error=str(e))
            return []

    async def _check_terminology(
        self, topic: str, outline: Any, sections_summary: str
    ) -> list[ConsistencyIssue]:
        """Terminology consistency: only needs keywords + section summaries."""
        keywords = ", ".join(getattr(outline, "keywords", [])) if outline else ""
        return await self._run_sub_check(
            self.prompts.terminology_check_system_prompt,
            self.prompts.terminology_check_user_prompt,
            {"topic": topic, "keywords": keywords, "sections_summary": sections_summary},
            "terminology",
        )

    async def _check_logic(
        self, topic: str, outline: Any, sections_summary: str
    ) -> list[ConsistencyIssue]:
        """Logic coherence: needs outline structure + section summaries."""
        outline_text = self._build_outline_context(outline)
        return await self._run_sub_check(
            self.prompts.logic_check_system_prompt,
            self.prompts.logic_check_user_prompt,
            {"topic": topic, "outline_text": outline_text, "sections_summary": sections_summary},
            "logic",
        )

    async def _check_citations(
        self, topic: str, references: list[Any], sections_summary: str
    ) -> list[ConsistencyIssue]:
        """Citation consistency: only needs reference list + section summaries."""
        references_text = self._build_references_context(references)
        return await self._run_sub_check(
            self.prompts.reference_consistency_system_prompt,
            self.prompts.reference_consistency_user_prompt,
            {
                "topic": topic,
                "references_text": references_text,
                "sections_summary": sections_summary,
            },
            "citation",
        )
