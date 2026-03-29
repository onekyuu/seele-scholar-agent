from datetime import UTC, datetime
from typing import Any

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from ..config import settings
from ..logging import get_logger
from ..state import AgentState, ReviewIssue, ReviewResult
from .prompts import REVIEWER_SYSTEM_PROMPT, REVIEWER_USER_PROMPT

logger = get_logger(__name__)


class ReviewerNode:
    def __init__(self, model: ChatOpenAI):
        self.model = model
        self.prompt = ChatPromptTemplate.from_messages(
            [
                ("system", REVIEWER_SYSTEM_PROMPT),
                ("user", REVIEWER_USER_PROMPT),
            ]
        )
        self.parser = JsonOutputParser()
        self.chain = self.prompt | self.model | self.parser

    async def review(self, state: AgentState) -> dict[str, Any]:
        sections = state["sections"]
        index = state["current_section_index"]
        section = sections[index]

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
                issues=[ReviewIssue(type="other", description=str(e), suggestion="请重试")],
                summary="审稿过程发生错误",
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

        if revision_count >= max_revisions:
            logger.warning("Max revisions reached, forcing approval")
            updated = sections.copy()
            updated[index] = section.model_copy(update={"status": "approved"})
            return {"sections": updated, "review_history": [record], "status": "completed"}

        comments = [
            f"【第 {section.revision_count} 轮审稿】评分：{review.score}/10",
            f"意见: {review.summary}",
        ]
        for i, issue in enumerate(review.issues, 1):
            comments.append(f"问题 {i}: [{issue.type}] {issue.description}")
            comments.append(f"建议: {issue.suggestion}")

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
