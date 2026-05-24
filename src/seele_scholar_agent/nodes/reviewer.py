from collections.abc import AsyncIterator
from datetime import UTC, datetime
from difflib import SequenceMatcher
from typing import Any

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from ..agent_config import PromptsConfig
from ..config import settings
from ..i18n import t
from ..logging import get_logger
from ..state import (
    AgentState,
    ClaimEvidenceBinding,
    PaperMetadata,
    QualityIssue,
    ReviewIssue,
    ReviewResult,
)
from . import CITATION_PATTERN, NodeStreamEvent, _stream_llm_text, invoke_with_retry
from .claim_audit import ExtractedClaim, RuleBasedClaimExtractor
from .methodology_audit import MethodologyAudit, MethodologyAuditFinding

logger = get_logger(__name__)

_CLAIM_TEXT_MATCH_THRESHOLD = 0.65


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
        self.claim_extractor = RuleBasedClaimExtractor()
        self.methodology_audit = MethodologyAudit()

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

        claim_source_issues, claim_quality_issues = self._audit_claim_source_support(
            section.section_id,
            section.content,
            state.get("claim_evidence_bindings", []),
        )
        if claim_source_issues:
            review.issues.extend(claim_source_issues)
            if review.approved:
                review = review.model_copy(update={"approved": False})

        methodology_issues, methodology_quality_issues = self._audit_methodology_statistics(
            section.section_id,
            section.title,
            section.content,
            state,
        )
        quality_issues = [*claim_quality_issues, *methodology_quality_issues]
        if methodology_issues:
            review.issues.extend(methodology_issues)
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
        return await self._handle_rejected(state, section, review, record, quality_issues)

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

        claim_source_issues, claim_quality_issues = self._audit_claim_source_support(
            section.section_id,
            section.content,
            state.get("claim_evidence_bindings", []),
        )
        if claim_source_issues:
            review.issues.extend(claim_source_issues)
            if review.approved:
                review = review.model_copy(update={"approved": False})

        methodology_issues, methodology_quality_issues = self._audit_methodology_statistics(
            section.section_id,
            section.title,
            section.content,
            state,
        )
        quality_issues = [*claim_quality_issues, *methodology_quality_issues]
        if methodology_issues:
            review.issues.extend(methodology_issues)
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
            final_result = await self._handle_rejected(
                state, section, review, record, quality_issues
            )

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
                        description=(
                            f"Citation [{num}] does not correspond to any paper in the "
                            f"reference list (total: {total_papers})"
                        ),
                        suggestion=(
                            f"Remove [{num}] or replace with a valid citation "
                            f"[1]-[{total_papers}]"
                        ),
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

    def _audit_claim_source_support(
        self,
        section_id: str,
        content: str,
        bindings: list[ClaimEvidenceBinding],
    ) -> tuple[list[ReviewIssue], list[QualityIssue]]:
        section_bindings = [binding for binding in bindings if binding.section_id == section_id]
        issues: list[ReviewIssue] = []
        quality_issues: list[QualityIssue] = []
        claims = self.claim_extractor.extract_factual_claims(content)

        bindings_by_citation: dict[int, list[ClaimEvidenceBinding]] = {}
        for binding in section_bindings:
            bindings_by_citation.setdefault(binding.citation_number, []).append(binding)

        for claim in claims:
            if not claim.citation_numbers:
                issue = ReviewIssue(
                    type="missing_citation",
                    description=f"Factual claim lacks citation: {claim.text}",
                    suggestion="Add a valid citation backed by an evidence packet or revise it.",
                    location=claim.text[:80],
                )
                issues.append(issue)
                quality_issues.append(
                    self._claim_quality_issue("UNSUPPORTED_CLAIM", issue, section_id, claim)
                )
                continue

            for citation_number in claim.citation_numbers:
                citation_bindings = [
                    binding
                    for binding in bindings_by_citation.get(citation_number, [])
                    if self._binding_matches_claim(binding, claim)
                ]
                if not citation_bindings:
                    issue = ReviewIssue(
                        type="citation_mismatch",
                        description=(
                            f"Citation [{citation_number}] is not bound to an evidence packet: "
                            f"{claim.text}"
                        ),
                        suggestion="Bind the cited claim to an evidence packet before approval.",
                        location=f"[{citation_number}]",
                    )
                    issues.append(issue)
                    quality_issues.append(
                        self._claim_quality_issue(
                            "CLAIM_MISSING_EVIDENCE_PACKET", issue, section_id, claim
                        )
                    )
                    continue

                best_binding = max(
                    citation_bindings,
                    key=lambda binding: (
                        binding.verdict == "supported",
                        bool(binding.chunk_id),
                        binding.support_score,
                    ),
                )
                if best_binding.verdict == "supported" and best_binding.chunk_id:
                    continue
                issue = self._unsupported_binding_issue(best_binding)
                issues.append(issue)
                quality_issues.append(
                    self._claim_quality_issue("UNSUPPORTED_CLAIM", issue, section_id, claim)
                )

        for binding in section_bindings:
            if binding.verdict == "supported" and binding.chunk_id:
                continue
            if any(
                binding.citation_number in claim.citation_numbers
                for claim in claims
            ):
                continue
            issue = self._unsupported_binding_issue(binding)
            issues.append(issue)
            quality_issues.append(
                self._claim_quality_issue(
                    "UNSUPPORTED_CLAIM",
                    issue,
                    section_id,
                    ExtractedClaim(
                        text=binding.claim_text,
                        citation_numbers=(binding.citation_number,),
                        is_factual=True,
                    ),
                )
            )

        return issues, quality_issues

    def _audit_methodology_statistics(
        self,
        section_id: str,
        section_title: str,
        content: str,
        state: AgentState,
    ) -> tuple[list[ReviewIssue], list[QualityIssue]]:
        outline = state.get("outline")
        paper_type = str(state.get("paper_type") or getattr(outline, "paper_type", "") or "")
        structure_pattern = str(
            state.get("structure_pattern") or getattr(outline, "structure_pattern", "") or ""
        )
        findings = self.methodology_audit.audit(
            section_title=section_title,
            content=content,
            paper_type=paper_type,
            structure_pattern=structure_pattern,
        )

        review_issues = [self._methodology_review_issue(finding) for finding in findings]
        quality_issues = [
            self._methodology_quality_issue(finding, section_id) for finding in findings
        ]
        return review_issues, quality_issues

    def _methodology_review_issue(self, finding: MethodologyAuditFinding) -> ReviewIssue:
        return ReviewIssue(
            type=finding.review_type,
            description=finding.description,
            suggestion=finding.suggestion,
            location=finding.location,
        )

    def _methodology_quality_issue(
        self, finding: MethodologyAuditFinding, section_id: str
    ) -> QualityIssue:
        return QualityIssue(
            code=finding.code,
            message=finding.description,
            severity="error",
            location=finding.location,
            blocking=False,
            details={"section_id": section_id},
        )

    def _unsupported_binding_issue(self, binding: ClaimEvidenceBinding) -> ReviewIssue:
        return ReviewIssue(
            type="citation_mismatch",
            description=(
                f"Claim-source support is {binding.verdict} for citation "
                f"[{binding.citation_number}]: {binding.claim_text}"
            ),
            suggestion=(
                "Replace the citation with a better-supported source, add a more "
                "specific evidence packet, or revise the claim."
            ),
            location=f"[{binding.citation_number}]",
        )

    def _binding_matches_claim(
        self, binding: ClaimEvidenceBinding, claim: ExtractedClaim
    ) -> bool:
        binding_text = self._normalize_claim_text(binding.claim_text)
        claim_text = self._normalize_claim_text(claim.text)
        if not binding_text or not claim_text:
            return False
        if binding_text == claim_text or binding_text in claim_text or claim_text in binding_text:
            return True
        similarity = SequenceMatcher(None, binding_text, claim_text).ratio()
        return similarity >= _CLAIM_TEXT_MATCH_THRESHOLD

    def _normalize_claim_text(self, text: str) -> str:
        text_without_citations = CITATION_PATTERN.sub("", text)
        return " ".join(text_without_citations.casefold().split())

    def _claim_quality_issue(
        self, code: str, review_issue: ReviewIssue, section_id: str, claim: ExtractedClaim
    ) -> QualityIssue:
        return QualityIssue(
            code=code,
            message=review_issue.description,
            severity="error",
            location=review_issue.location or section_id,
            blocking=False,
            details={
                "section_id": section_id,
                "claim_text": claim.text,
                "citation_numbers": list(claim.citation_numbers),
            },
        )

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
        self,
        state: AgentState,
        section: Any,
        review: ReviewResult,
        record: dict[str, Any],
        quality_issues: list[QualityIssue] | None = None,
    ) -> dict[str, Any]:
        sections = state["sections"]
        index = state["current_section_index"]
        total_revision_count = state.get("revision_count", 0)
        section_revision_count = section.revision_count
        max_revisions = state.get("max_revisions", settings.MAX_REVISIONS)
        lang = state.get("language", "zh")

        if section_revision_count >= max_revisions:
            logger.warning(
                "max revisions reached, blocking section instead of forcing approval",
                section=section.title,
                revision_count=section_revision_count,
                max_revisions=max_revisions,
            )
            updated = sections.copy()
            issue_summaries = [issue.description for issue in review.issues if issue.description]
            comments = [
                t(lang, "review_round", round=section_revision_count, score=review.score),
                t(lang, "review_opinion", summary=review.summary),
            ]
            for i, issue in enumerate(review.issues, 1):
                comments.append(
                    t(lang, "review_issue", i=i, type=issue.type, description=issue.description)
                )
                comments.append(t(lang, "review_suggestion", suggestion=issue.suggestion))
            updated[index] = section.model_copy(
                update={"status": "review", "review_comments": comments}
            )

            quality_issue = QualityIssue(
                code="MAX_REVISIONS_REACHED",
                message=(
                    f"Section '{section.title}' failed review after "
                    f"{section_revision_count} revision(s)."
                ),
                severity="blocking",
                location=section.section_id,
                blocking=True,
                details={
                    "section_title": section.title,
                    "revision_count": section_revision_count,
                    "max_revisions": max_revisions,
                    "review_score": review.score,
                    "review_summary": review.summary,
                    "review_issues": issue_summaries,
                },
            )

            result: dict[str, Any] = {
                "sections": updated,
                "review_history": [record],
                "current_review": review.model_dump(),
                "quality_issues": [quality_issue, *(quality_issues or [])],
                "status": "waiting_human",
                "error_message": quality_issue.message,
            }
            return result

        next_section_revision_count = section_revision_count + 1
        comments = [
            t(lang, "review_round", round=next_section_revision_count, score=review.score),
            t(lang, "review_opinion", summary=review.summary),
        ]
        for i, issue in enumerate(review.issues, 1):
            comments.append(
                t(lang, "review_issue", i=i, type=issue.type, description=issue.description)
            )
            comments.append(t(lang, "review_suggestion", suggestion=issue.suggestion))

        updated = sections.copy()
        updated[index] = section.model_copy(
            update={
                "status": "writing",
                "revision_count": next_section_revision_count,
                "review_comments": comments,
            }
        )

        result = {
            "sections": updated,
            "review_history": [record],
            "current_review": review.model_dump(),
            "revision_count": total_revision_count + 1,
            "status": "writing",
        }
        if quality_issues:
            result["quality_issues"] = quality_issues
        return result
