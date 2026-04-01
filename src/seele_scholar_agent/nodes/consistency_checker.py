from typing import Any

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from ..agent_config import PromptsConfig
from ..logging import get_logger
from ..state import AgentState, ConsistencyIssue, SectionDraft
from . import invoke_with_retry

logger = get_logger(__name__)

_SUMMARY_MAX_CHARS = 600


def _build_sections_summary(sections: list[SectionDraft]) -> str:
    parts = []
    for s in sections:
        if s.content and s.status in ("approved", "auto_generated"):
            snippet = s.content[:_SUMMARY_MAX_CHARS]
            if len(s.content) > _SUMMARY_MAX_CHARS:
                snippet += "..."
            parts.append(f"[{s.title}]\n{snippet}")
    return "\n\n".join(parts) if parts else "无"


class ConsistencyCheckerNode:
    def __init__(self, llm: ChatOpenAI, prompts: PromptsConfig):
        self.llm = llm
        self.prompts = prompts
        self.prompt = ChatPromptTemplate.from_messages(
            [
                ("system", prompts.consistency_check_system_prompt),
                ("user", prompts.consistency_check_user_prompt),
            ]
        )
        self.parser = JsonOutputParser()
        self.chain = self.prompt | self.llm | self.parser

    async def check(self, state: AgentState) -> dict[str, Any]:
        sections = state.get("sections", [])
        approved = [s for s in sections if s.status in ("approved", "auto_generated")]

        if len(approved) < 2:
            logger.info("not enough approved sections for consistency check")
            return {"consistency_checked": True, "consistency_issues": []}

        sections_summary = _build_sections_summary(approved)

        try:
            result = await invoke_with_retry(
                self.chain,
                {
                    "topic": state["topic"],
                    "sections_summary": sections_summary,
                },
            )
            raw_issues = result.get("issues", [])
            issues = [
                ConsistencyIssue(
                    issue_type=item.get("issue_type", "other"),
                    description=item.get("description", ""),
                    sections_involved=item.get("sections_involved", []),
                    suggestion=item.get("suggestion", ""),
                )
                for item in raw_issues
            ]
        except Exception as e:
            logger.error("consistency check failed", error=str(e))
            issues = []

        logger.info("consistency check completed", issues_found=len(issues))
        return {"consistency_checked": True, "consistency_issues": issues}
