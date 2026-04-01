from collections.abc import AsyncIterator
from datetime import UTC, datetime
from typing import Any

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from ..agent_config import PromptsConfig
from ..config import settings
from ..i18n import t
from ..logging import get_logger
from ..state import AgentState, PaperMetadata, ReviewIssue, ReviewResult
from . import CITATION_PATTERN, NodeStreamEvent, _stream_llm_text, invoke_with_retry

logger = get_logger(__name__)


def _build_numbered_papers_summary(papers: list[PaperMetadata]) -> str:
    if not papers:
        return "无"
    lines = []
    for i, p in enumerate(papers, 1):
        authors_str = ", ".join(p.authors[:3])
        if len(p.authors) > 3:
            authors_str += " et al."
        abstract_snippet = p.abstract[:120] + "..." if len(p.abstract) > 120 else p.abstract
        lines.append(f"[{i}] {p.title} — {authors_str}. {abstract_snippet}")
    return "\n".join(lines)


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
        self.citation_alignment_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", self.prompts.citation_alignment_system_prompt),
                ("user", self.prompts.citation_alignment_user_prompt),
            ]
        )
        self.parser = JsonOutputParser()
        self.chain = self.prompt | self.llm | self.parser
        self.stream_chain = self.prompt | self.llm
        self.citation_chain = self.citation_alignment_prompt | self.llm | self.parser

    async def review(self, state: AgentState) -> dict[str, Any]:
        sections = state["sections"]
        index = state["current_section_index"]
        lang = state.get("language", "zh")

        if index >= len(sections):
            logger.error("review called with out-of-bounds index", index=index, total=len(sections))
            return {"status": "failed", "error_message": "Review index out of bounds"}

        section = sections[index]

        logger.info("reviewing section", title=section.title)

        try:
            result = await invoke_with_retry(
                self.chain,
                {
                    "topic": state["topic"],
                    "section_title": section.title,
                    "content": section.content,
                },
            )
            review = ReviewResult(
                approved=result.get("approved", False),
                score=result.get("score", 5),
                issues=[ReviewIssue(**i) for i in result.get("issues", [])],
                summary=result.get("summary", ""),
            )
        except Exception as e:
            logger.error("review failed after retries", error=str(e))
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

        citation_issues = self._verify_citations(section.content, state.get("papers", []))
        if citation_issues:
            review.issues.extend(citation_issues)
            if review.approved:
                review = review.model_copy(update={"approved": False})

        papers = state.get("papers", [])
        if papers and section.content:
            alignment_issues = await self._verify_citation_alignment(
                section.title, section.content, papers
            )
            if alignment_issues:
                review.issues.extend(alignment_issues)
                if review.approved:
                    review = review.model_copy(update={"approved": False})

        record = {
            "section": section.title,
            "score": review.score,
            "approved": review.approved,
            "timestamp": datetime.now(UTC).isoformat(),
        }

        if review.approved:
            return await self._handle_approved(state, section, review, record)
        return await self._handle_rejected(state, section, review, record)

    async def astream(self, state: AgentState) -> AsyncIterator[NodeStreamEvent]:
        sections = state["sections"]
        index = state["current_section_index"]
        lang = state.get("language", "zh")

        if index >= len(sections):
            yield NodeStreamEvent(
                type="result",
                result={"status": "failed", "error_message": "Review index out of bounds"},
            )
            return

        section = sections[index]
        yield NodeStreamEvent(type="progress", progress=f"reviewing:{section.title}")

        input_data = {
            "topic": state["topic"],
            "section_title": section.title,
            "content": section.content,
        }

        full_text = ""
        async for event in _stream_llm_text(self.stream_chain, input_data):
            full_text += event.get("token", "")
            yield event

        try:
            result = self.parser.parse(full_text)
            review = ReviewResult(
                approved=result.get("approved", False),
                score=result.get("score", 5),
                issues=[ReviewIssue(**i) for i in result.get("issues", [])],
                summary=result.get("summary", ""),
            )
        except Exception as e:
            logger.error("review stream parse failed", error=str(e))
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

        citation_issues = self._verify_citations(section.content, state.get("papers", []))
        if citation_issues:
            review.issues.extend(citation_issues)
            if review.approved:
                review = review.model_copy(update={"approved": False})

        papers = state.get("papers", [])
        if papers and section.content:
            yield NodeStreamEvent(type="progress", progress="verifying_citations")
            alignment_issues = await self._verify_citation_alignment(
                section.title, section.content, papers
            )
            if alignment_issues:
                review.issues.extend(alignment_issues)
                if review.approved:
                    review = review.model_copy(update={"approved": False})

        record = {
            "section": section.title,
            "score": review.score,
            "approved": review.approved,
            "timestamp": datetime.now(UTC).isoformat(),
        }

        if review.approved:
            final_result = await self._handle_approved(state, section, review, record)
        else:
            final_result = await self._handle_rejected(state, section, review, record)

        yield NodeStreamEvent(type="result", result=final_result)

    def _verify_citations(self, content: str, papers: list[PaperMetadata]) -> list[ReviewIssue]:
        cited_numbers = {int(m) for m in CITATION_PATTERN.findall(content)}
        if not cited_numbers:
            return []

        total_papers = len(papers)
        issues: list[ReviewIssue] = []
        for num in sorted(cited_numbers):
            if num < 1 or num > total_papers:
                issues.append(
                    ReviewIssue(
                        type="missing_citation",
                        description=f"Citation [{num}] does not correspond to any paper in the reference list (total: {total_papers})",
                        suggestion=f"Remove [{num}] or replace with a valid citation [1]-[{total_papers}]",
                    )
                )
        return issues

    async def _verify_citation_alignment(
        self, section_title: str, content: str, papers: list[PaperMetadata]
    ) -> list[ReviewIssue]:
        cited_numbers = {int(m) for m in CITATION_PATTERN.findall(content)}
        valid_cited = {n for n in cited_numbers if 1 <= n <= len(papers)}
        if not valid_cited:
            return []

        numbered_papers = _build_numbered_papers_summary(papers)
        try:
            result = await invoke_with_retry(
                self.citation_chain,
                {
                    "section_title": section_title,
                    "content": content,
                    "numbered_papers": numbered_papers,
                },
            )
            raw_issues = result.get("issues", [])
            return [
                ReviewIssue(
                    type="citation_mismatch",
                    description=item.get("description", ""),
                    suggestion=item.get("suggestion", ""),
                    location=f"[{item.get('citation_number', '?')}]",
                )
                for item in raw_issues
            ]
        except Exception as e:
            logger.warning("citation alignment check failed", error=str(e))
            return []

    async def _handle_approved(
        self, state: AgentState, section: Any, review: ReviewResult, record: dict[str, Any]
    ) -> dict[str, Any]:
        sections = state["sections"]
        index = state["current_section_index"]
        updated = sections.copy()
        updated[index] = section.model_copy(update={"status": "approved"})

        completed = state.get("sections_completed", [])
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

    async def _handle_rejected(
        self, state: AgentState, section: Any, review: ReviewResult, record: dict[str, Any]
    ) -> dict[str, Any]:
        sections = state["sections"]
        index = state["current_section_index"]
        revision_count = state.get("revision_count", 0)
        max_revisions = state.get("max_revisions", settings.MAX_REVISIONS)
        lang = state.get("language", "zh")

        if revision_count >= max_revisions:
            logger.warning("max revisions reached, forcing approval")
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
