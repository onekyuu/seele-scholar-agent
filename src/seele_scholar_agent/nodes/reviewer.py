from datetime import UTC, datetime
from typing import Any

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from ..config import settings
from ..i18n import t
from ..logging import get_logger
from ..state import AgentState, ReviewIssue, ReviewResult
from ..agent_config import PromptsConfig

logger = get_logger(__name__)


class ReviewerNode:
    def __init__(self, llm: ChatOpenAI, prompts: PromptsConfig):
        self.llm = llm
        self.prompts = prompts
        self.prompt = ChatPromptTemplate.from_messages(
            [
                ("system", self.prompts.reviewer_system_prompt),
                ("user", self.prompts.reviewer_user_prompt),
            ]
        )
        self.parser = JsonOutputParser()
        self.chain = self.prompt | self.llm | self.parser

    async def review(self, state: AgentState) -> dict[str, Any]:
        sections = state["sections"]
        index = state["current_section_index"]
        section = sections[index]
        lang = state.get("language", "zh")

        logger.info(f"Reviewing section: {section.title}")

        try:
            result = await self.chain.ainvoke(
                {
                    "topic": state["topic"],
                    "section_title": section.title,
                    "content": section.content,
                }
            )
            review = ReviewResult(
                approved=result.get("approved", False),
                score=result.get("score", 5),
                issues=[ReviewIssue(**i) for i in result.get("issues", [])],
                summary=result.get("summary", ""),
            )
        except Exception as e:
            logger.error(f"Review failed: {e}")
            review = ReviewResult(
                approved=False,
                score=5,
                issues=[
                    ReviewIssue(
                        type="other", description=str(e), suggestion=t(lang, "review_error_retry")
                    )
                ],
                summary=t(lang, "review_error_summary"),
            )

        record = {
            "section": section.title,
            "score": review.score,
            "approved": review.approved,
            "timestamp": datetime.now(UTC).isoformat(),
        }

        if review.approved:
            return await self._handle_approved(state, section, review, record)
        return await self._handle_rejected(state, section, review, record)

    async def _handle_approved(self, state, section, review, record):
        sections = state["sections"]
        index = state["current_section_index"]
        updated = sections.copy()
        updated[index] = section.model_copy(update={"status": "approved"})

        completed = state.get("section_completed", [])
        completed.append(section.title)

        if index + 1 >= len(sections):
            return {
                "sections": updated,
                "sections_completed": completed,
                "review_history": [record],
                "current_review": review.model_dump(),
                "status": "completed",
            }

        return {
            "sections": updated,
            "sections_completed": completed,
            "current_section_index": index + 1,
            "review_history": [record],
            "current_review": review.model_dump(),
            "status": "writing",
        }

    async def _handle_rejected(self, state, section, review, record):
        sections = state["sections"]
        index = state["current_section_index"]
        revision_count = state.get("revision_count", 0)
        max_revisions = state.get("max_revisions", settings.MAX_REVISIONS)
        lang = state.get("language", "zh")

        if revision_count >= max_revisions:
            logger.warning("Max revisions reached, forcing approval")
            updated = sections.copy()
            updated[index] = section.model_copy(update={"status": "approved"})
            return {"sections": updated, "review_history": [record], "status": "completed"}

        comments = [
            t(lang, "review_round", round=section.revision_count, score=review.score),
            t(lang, "review_opinion", summary=review.summary),
        ]
        for i, issue in enumerate(review.issues, 1):
            comments.append(
                t(lang, "review_issue", i=i, type=issue.type, description=issue.description)
            )
            comments.append(t(lang, "review_suggestion", suggestion=issue.suggestion))

        updated = sections.copy()
        updated[index] = section.model_copy(
            update={"status": "writing", "review_comments": section.review_comments + comments}
        )

        return {
            "sections": updated,
            "review_history": [record],
            "current_review": review.model_dump(),
            "revision_count": revision_count + 1,
            "status": "writing",
        }
