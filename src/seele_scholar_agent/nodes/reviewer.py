import re
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from difflib import SequenceMatcher
from typing import Any

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from ..agent_config import PromptsConfig
from ..config import settings
from ..document_profile import (
    is_compound_section_title,
    is_proposal_plan_sentence,
    is_research_proposal,
    is_schedule_section,
    missing_proposal_core_tasks,
    missing_schedule_phases,
)
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
from .language_style_audit import LanguageStyleAudit, LanguageStyleFinding
from .methodology_audit import MethodologyAudit, MethodologyAuditFinding
from .paragraph_quality_audit import ParagraphQualityAudit, ParagraphQualityFinding

logger = get_logger(__name__)

_CLAIM_TEXT_MATCH_THRESHOLD = settings.CLAIM_TEXT_MATCH_THRESHOLD
_BUDGET_RE = re.compile(
    r"(\d+\s*(?:-\s*\d+\s*)?(?:字|文字|語|words?)|全文|预算|予算|budget|target_words|字数|字數)",
    re.IGNORECASE,
)


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
        self.paragraph_quality_audit = ParagraphQualityAudit()
        self.language_style_audit = LanguageStyleAudit()

    async def review(self, state: AgentState) -> dict[str, Any]:
        sections = state["sections"]
        index = state["current_section_index"]
        lang = state.get("language", "zh")

        if index >= len(sections):
            logger.error("review called with out-of-bounds index", index=index, total=len(sections))
            return {"status": "failed", "error_message": "Review index out of bounds"}

        section = sections[index]
        proposal_profile = is_research_proposal(state)

        logger.info("reviewing section", title=section.title)

        try:
            result = await invoke_with_retry(
                self.chain,
                {
                    "topic": state["topic"],
                    "section_title": section.title,
                    "content": section.content,
                    "document_type": "research_proposal" if proposal_profile else "academic_paper",
                    "review_policy": self._review_policy_text(proposal_profile),
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
                section.title, section.content, papers, proposal_profile=proposal_profile
            )
            if alignment_issues:
                review.issues.extend(alignment_issues)
                if review.approved:
                    review = review.model_copy(update={"approved": False})

        claim_source_issues, claim_quality_issues = self._audit_claim_source_support(
            section.section_id,
            section.title,
            section.content,
            state.get("claim_evidence_bindings", []),
            proposal_profile=proposal_profile,
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

        paragraph_issues, paragraph_quality_issues = self._audit_paragraph_quality(
            section.section_id,
            section.title,
            section.content,
            state,
            proposal_profile=proposal_profile,
        )
        quality_issues = [*quality_issues, *paragraph_quality_issues]
        if paragraph_issues:
            review.issues.extend(paragraph_issues)
            if review.approved:
                review = review.model_copy(update={"approved": False})

        language_style_issues, language_style_quality_issues = self._audit_language_style(
            section.section_id,
            section.content,
            state,
        )
        quality_issues = [*quality_issues, *language_style_quality_issues]
        if language_style_issues:
            review.issues.extend(language_style_issues)
            if review.approved:
                review = review.model_copy(update={"approved": False})

        if proposal_profile:
            schedule_issues, schedule_quality_issues = self._audit_proposal_schedule(
                section.section_id, section.title, section.content
            )
            quality_issues = [*quality_issues, *schedule_quality_issues]
            if schedule_issues:
                review.issues.extend(schedule_issues)
                if review.approved:
                    review = review.model_copy(update={"approved": False})

        if proposal_profile:
            review, quality_issues = self._apply_proposal_review_policy(
                review, quality_issues, section
            )

        record = {
            "section": section.title,
            "score": review.score,
            "approved": review.approved,
            "timestamp": datetime.now(UTC).isoformat(),
        }

        if review.approved:
            result = await self._handle_approved(state, section, review, record)
        else:
            result = await self._handle_rejected(state, section, review, record, quality_issues)
        if quality_issues and "quality_issues" not in result:
            result["quality_issues"] = quality_issues
        result["review_diagnostics"] = self._build_review_diagnostics(
            review, quality_issues, proposal_profile, section, state
        )
        return result

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
        proposal_profile = is_research_proposal(state)
        yield NodeStreamEvent(type="progress", progress=f"reviewing:{section.title}")

        input_data = {
            "topic": state["topic"],
            "section_title": section.title,
            "content": section.content,
            "document_type": "research_proposal" if proposal_profile else "academic_paper",
            "review_policy": self._review_policy_text(proposal_profile),
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
                section.title, section.content, papers, proposal_profile=proposal_profile
            )
            if alignment_issues:
                review.issues.extend(alignment_issues)
                if review.approved:
                    review = review.model_copy(update={"approved": False})

        claim_source_issues, claim_quality_issues = self._audit_claim_source_support(
            section.section_id,
            section.title,
            section.content,
            state.get("claim_evidence_bindings", []),
            proposal_profile=proposal_profile,
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

        paragraph_issues, paragraph_quality_issues = self._audit_paragraph_quality(
            section.section_id,
            section.title,
            section.content,
            state,
            proposal_profile=proposal_profile,
        )
        quality_issues = [*quality_issues, *paragraph_quality_issues]
        if paragraph_issues:
            review.issues.extend(paragraph_issues)
            if review.approved:
                review = review.model_copy(update={"approved": False})

        language_style_issues, language_style_quality_issues = self._audit_language_style(
            section.section_id,
            section.content,
            state,
        )
        quality_issues = [*quality_issues, *language_style_quality_issues]
        if language_style_issues:
            review.issues.extend(language_style_issues)
            if review.approved:
                review = review.model_copy(update={"approved": False})

        if proposal_profile:
            schedule_issues, schedule_quality_issues = self._audit_proposal_schedule(
                section.section_id, section.title, section.content
            )
            quality_issues = [*quality_issues, *schedule_quality_issues]
            if schedule_issues:
                review.issues.extend(schedule_issues)
                if review.approved:
                    review = review.model_copy(update={"approved": False})

        if proposal_profile:
            review, quality_issues = self._apply_proposal_review_policy(
                review, quality_issues, section
            )

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
        if quality_issues and "quality_issues" not in final_result:
            final_result["quality_issues"] = quality_issues
        final_result["review_diagnostics"] = self._build_review_diagnostics(
            review, quality_issues, proposal_profile, section, state
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
                        blocking=True,
                        category="blocking",
                    )
                )
        return issues

    async def _verify_citation_alignment(
        self,
        section_title: str,
        content: str,
        papers: list[PaperMetadata],
        *,
        proposal_profile: bool = False,
    ) -> list[ReviewIssue]:
        cited_numbers = {int(m) for m in CITATION_PATTERN.findall(content)}
        valid_cited = {n for n in cited_numbers if 1 <= n <= len(papers)}
        if not valid_cited:
            return []

        content_for_alignment = (
            self._cited_sentence_context(content) if proposal_profile else content
        )
        numbered_papers = _build_numbered_papers_summary(papers)
        try:
            result = await invoke_with_retry(
                self.citation_chain,
                {
                    "section_title": section_title,
                    "content": content_for_alignment,
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
                    blocking=False,
                    category=(
                        "citation_warning" if proposal_profile else "content_quality"
                    ),
                )
                for item in raw_issues
            ]
        except Exception as e:
            logger.warning("citation alignment check failed", error=str(e))
            return []

    def _review_policy_text(self, proposal_profile: bool) -> str:
        if not proposal_profile:
            return "Review as an academic paper section."
        return (
            "Review as a Japanese graduate-school research proposal application, not as "
            "an academic paper body section. Only block for off-topic content, missing "
            "title-core task, truncation/incomplete sentences, impossible-to-judge "
            "purpose/method/plan feasibility, severe factual error, misleading citation, "
            "or enumeration/structure breakage. Treat most missing citations, citation "
            "mismatches, and unsupported claims as warnings unless they mislead the "
            "proposal."
        )

    def _audit_claim_source_support(
        self,
        section_id: str,
        section_title: str,
        content: str,
        bindings: list[ClaimEvidenceBinding],
        *,
        proposal_profile: bool = False,
    ) -> tuple[list[ReviewIssue], list[QualityIssue]]:
        section_bindings = [binding for binding in bindings if binding.section_id == section_id]
        issues: list[ReviewIssue] = []
        quality_issues: list[QualityIssue] = []
        claims = self.claim_extractor.extract_factual_claims(content)

        bindings_by_citation: dict[int, list[ClaimEvidenceBinding]] = {}
        for binding in section_bindings:
            bindings_by_citation.setdefault(binding.citation_number, []).append(binding)

        for claim in claims:
            if proposal_profile and self._is_proposal_deferred_claim(claim, section_title):
                continue

            if not claim.citation_numbers:
                issue = ReviewIssue(
                    type="missing_citation",
                    description=f"Factual claim lacks citation: {claim.text}",
                    suggestion="Add a valid citation backed by an evidence packet or revise it.",
                    location=claim.text[:80],
                    blocking=False,
                    category=(
                        "citation_warning" if proposal_profile else "content_quality"
                    ),
                )
                quality_issue = self._claim_quality_issue(
                    "UNSUPPORTED_CLAIM", issue, section_id, claim
                )
                if proposal_profile:
                    issues.append(issue)
                    quality_issues.append(
                        quality_issue.model_copy(
                            update={
                                "severity": "warning",
                                "details": {
                                    **quality_issue.details,
                                    "audit_source": "claim_source",
                                    "deferred": True,
                                },
                            }
                        )
                    )
                else:
                    issues.append(issue)
                    quality_issues.append(quality_issue)
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
                        blocking=False,
                        category=(
                            "citation_warning" if proposal_profile else "content_quality"
                        ),
                    )
                    quality_issue = self._claim_quality_issue(
                        "CLAIM_MISSING_EVIDENCE_PACKET", issue, section_id, claim
                    )
                    if proposal_profile:
                        issues.append(issue)
                        quality_issues.append(
                            quality_issue.model_copy(
                                update={
                                    "severity": "warning",
                                    "details": {
                                        **quality_issue.details,
                                        "audit_source": "evidence_binding",
                                        "deferred": True,
                                    },
                                }
                            )
                        )
                    else:
                        issues.append(issue)
                        quality_issues.append(quality_issue)
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
                if proposal_profile:
                    issue = issue.model_copy(update={"category": "citation_warning"})
                quality_issue = self._claim_quality_issue(
                    "UNSUPPORTED_CLAIM", issue, section_id, claim
                )
                if proposal_profile:
                    quality_issues.append(
                        quality_issue.model_copy(
                            update={
                                "severity": "warning",
                                "details": {
                                    **quality_issue.details,
                                    "audit_source": "evidence_binding",
                                    "deferred": True,
                                    "binding_diagnostics": best_binding.diagnostics,
                                },
                            }
                        )
                    )
                else:
                    issues.append(issue)
                    quality_issues.append(quality_issue)

        for binding in section_bindings:
            if binding.verdict == "supported" and binding.chunk_id:
                continue
            if any(
                binding.citation_number in claim.citation_numbers
                for claim in claims
            ):
                continue
            issue = self._unsupported_binding_issue(binding)
            if proposal_profile:
                issue = issue.model_copy(update={"category": "citation_warning"})
            claim = ExtractedClaim(
                text=binding.claim_text,
                citation_numbers=(binding.citation_number,),
                is_factual=True,
            )
            quality_issue = self._claim_quality_issue(
                "UNSUPPORTED_CLAIM",
                issue,
                section_id,
                claim,
            )
            if proposal_profile:
                quality_issues.append(
                    quality_issue.model_copy(
                        update={
                            "severity": "warning",
                            "details": {
                                **quality_issue.details,
                                "audit_source": "evidence_binding",
                                "deferred": True,
                                "binding_diagnostics": binding.diagnostics,
                            },
                        }
                    )
                )
            else:
                issues.append(issue)
                quality_issues.append(quality_issue)

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
            document_type="research_proposal" if is_research_proposal(state) else "",
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
            details={"section_id": section_id, "audit_source": "methodology"},
        )

    def _audit_paragraph_quality(
        self,
        section_id: str,
        section_title: str,
        content: str,
        state: AgentState,
        *,
        proposal_profile: bool = False,
    ) -> tuple[list[ReviewIssue], list[QualityIssue]]:
        outline = state.get("outline")
        section_outline = None
        if outline is not None:
            section_outline = next(
                (section for section in outline.sections if section.title == section_title),
                None,
            )
        findings = self.paragraph_quality_audit.audit(
            section_title=section_title,
            content=content,
            section_outline=section_outline,
            proposal_profile=proposal_profile,
        )

        review_issues = [self._paragraph_review_issue(finding) for finding in findings]
        quality_issues = [
            self._paragraph_quality_issue(finding, section_id) for finding in findings
        ]
        return review_issues, quality_issues

    def _paragraph_review_issue(self, finding: ParagraphQualityFinding) -> ReviewIssue:
        return ReviewIssue(
            type=finding.review_type,
            description=finding.description,
            suggestion=finding.suggestion,
            location=finding.location,
        )

    def _paragraph_quality_issue(
        self, finding: ParagraphQualityFinding, section_id: str
    ) -> QualityIssue:
        return QualityIssue(
            code=finding.code,
            message=finding.description,
            severity="error",
            location=finding.location,
            blocking=False,
            details={"section_id": section_id, "audit_source": "paragraph_quality"},
        )

    def _audit_language_style(
        self,
        section_id: str,
        content: str,
        state: AgentState,
    ) -> tuple[list[ReviewIssue], list[QualityIssue]]:
        findings = self.language_style_audit.audit(content, state)
        review_issues = [self._language_style_review_issue(finding) for finding in findings]
        quality_issues = [
            self._language_style_quality_issue(finding, section_id) for finding in findings
        ]
        return review_issues, quality_issues

    def _language_style_review_issue(self, finding: LanguageStyleFinding) -> ReviewIssue:
        return ReviewIssue(
            type=finding.review_type,
            description=finding.description,
            suggestion=finding.suggestion,
            location=finding.location,
        )

    def _language_style_quality_issue(
        self, finding: LanguageStyleFinding, section_id: str
    ) -> QualityIssue:
        return QualityIssue(
            code=finding.code,
            message=finding.description,
            severity="error",
            location=finding.location,
            blocking=False,
            details={"section_id": section_id, "audit_source": "language_style"},
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
                "audit_source": "claim_source",
            },
        )

    def _audit_proposal_schedule(
        self, section_id: str, section_title: str, content: str
    ) -> tuple[list[ReviewIssue], list[QualityIssue]]:
        if not is_schedule_section(section_title):
            return [], []

        missing = missing_schedule_phases(content)
        if not missing:
            return [], []

        issue = ReviewIssue(
            type="format_issue",
            description=(
                "Research proposal schedule is incomplete; missing phases: "
                + ", ".join(missing)
            ),
            suggestion=(
                "Revise the schedule to cover 1年次前期, 1年次後期, 2年次前期, "
                "and 2年次後期, with tasks and deliverables for each phase."
            ),
            location=section_title,
            blocking=True,
            category="blocking",
        )
        quality_issue = QualityIssue(
            code="PROPOSAL_SCHEDULE_PHASES_MISSING",
            message=issue.description,
            severity="blocking",
            location=section_title,
            blocking=True,
            details={
                "section_id": section_id,
                "audit_source": "structural",
                "missing_phases": missing,
            },
        )
        return [issue], [quality_issue]

    def _apply_proposal_review_policy(
        self,
        review: ReviewResult,
        quality_issues: list[QualityIssue],
        section: Any,
    ) -> tuple[ReviewResult, list[QualityIssue]]:
        structural_issues, structural_quality_issues = self._audit_proposal_core_structure(
            section.section_id, section.title, section.content
        )
        if structural_issues:
            review.issues.extend(structural_issues)
            quality_issues = [*quality_issues, *structural_quality_issues]

        normalized_issues = [
            self._normalize_proposal_review_issue(issue) for issue in review.issues
        ]
        has_blocking_issue = any(issue.blocking for issue in normalized_issues) or any(
            issue.blocking or issue.severity == "blocking" for issue in quality_issues
        )

        approved = review.approved
        if has_blocking_issue:
            approved = False
        elif review.score >= 7:
            approved = True

        return review.model_copy(
            update={"approved": approved, "issues": normalized_issues}
        ), quality_issues

    def _audit_proposal_core_structure(
        self, section_id: str, section_title: str, content: str
    ) -> tuple[list[ReviewIssue], list[QualityIssue]]:
        issues: list[ReviewIssue] = []
        quality_issues: list[QualityIssue] = []
        stripped = content.strip()

        if not stripped:
            issue = self._proposal_blocking_issue(
                "PROPOSAL_SECTION_EMPTY",
                "Proposal section is empty.",
                "Write a compact section that addresses the title's core task.",
                section_title,
            )
            return [issue], [self._proposal_quality_issue(issue, section_id)]

        if self._looks_truncated(stripped):
            issue = self._proposal_blocking_issue(
                "PROPOSAL_SECTION_TRUNCATED",
                "Proposal section appears truncated or ends with an incomplete sentence.",
                "Complete the final sentence while staying within the section budget.",
                section_title,
            )
            issues.append(issue)
            quality_issues.append(self._proposal_quality_issue(issue, section_id))

        enumeration_issue = self._proposal_enumeration_issue(section_title, stripped)
        if enumeration_issue is not None:
            issues.append(enumeration_issue)
            quality_issues.append(self._proposal_quality_issue(enumeration_issue, section_id))

        missing_core_tasks = missing_proposal_core_tasks(section_title, stripped)
        if missing_core_tasks:
            issue = self._proposal_blocking_issue(
                "PROPOSAL_CORE_TASK_MISSING",
                (
                    "Proposal section is missing title-core task(s): "
                    + ", ".join(missing_core_tasks)
                ),
                (
                    "Add overview-level coverage for the missing task(s); do not expand "
                    "into a full paper-style subsection."
                ),
                section_title,
                details={"missing_core_tasks": missing_core_tasks},
            )
            issues.append(issue)
            quality_issues.append(self._proposal_quality_issue(issue, section_id))

        return issues, quality_issues

    def _proposal_blocking_issue(
        self,
        code: str,
        description: str,
        suggestion: str,
        location: str,
        *,
        details: dict[str, Any] | None = None,
    ) -> ReviewIssue:
        _ = code, details
        return ReviewIssue(
            type="format_issue",
            description=description,
            suggestion=suggestion,
            location=location,
            blocking=True,
            category="blocking",
        )

    def _proposal_quality_issue(
        self, review_issue: ReviewIssue, section_id: str
    ) -> QualityIssue:
        return QualityIssue(
            code=self._proposal_quality_code(review_issue.description),
            message=review_issue.description,
            severity="blocking",
            location=review_issue.location,
            blocking=True,
            details={
                "section_id": section_id,
                "audit_source": "proposal_structure",
                "category": review_issue.category,
            },
        )

    def _proposal_quality_code(self, description: str) -> str:
        if "truncated" in description or "incomplete sentence" in description:
            return "PROPOSAL_SECTION_TRUNCATED"
        if "missing title-core task" in description:
            return "PROPOSAL_CORE_TASK_MISSING"
        if "enumeration" in description or "declares" in description:
            return "PROPOSAL_ENUMERATION_INCONSISTENT"
        return "PROPOSAL_SECTION_STRUCTURAL_BLOCK"

    def _looks_truncated(self, content: str) -> bool:
        if content.endswith(("。", ".", "!", "?", "！", "？", "」", "』", "）", ")", "】")):
            return False
        return True

    def _proposal_enumeration_issue(
        self, section_title: str, content: str
    ) -> ReviewIssue | None:
        declared_count = self._declared_enumeration_count(content)
        if declared_count is None:
            return None
        actual_count = self._actual_enumeration_count(content)
        if actual_count >= declared_count:
            return None
        return self._proposal_blocking_issue(
            "PROPOSAL_ENUMERATION_INCONSISTENT",
            (
                f"Section declares {declared_count} points/stages but only "
                f"{actual_count} are explicitly developed."
            ),
            (
                "Make the declared number match the actual text, or add the missing "
                "point/stage compactly."
            ),
            section_title,
            details={"declared_count": declared_count, "actual_count": actual_count},
        )

    def _declared_enumeration_count(self, content: str) -> int | None:
        patterns: tuple[tuple[str, int], ...] = (
            (r"(?:三点|3点|三つ|三段階|3段階|three (?:points|stages))", 3),
            (r"(?:二点|2点|二つ|二段階|2段階|two (?:points|stages))", 2),
        )
        for pattern, count in patterns:
            if re.search(pattern, content, re.IGNORECASE):
                return count
        return None

    def _actual_enumeration_count(self, content: str) -> int:
        markers = re.findall(
            r"(?:第[一二三四五](?:に|段階|ステップ)|[一二三四五]つ目|[（(]?[1-5][）).、]|"
            r"\b(?:first|second|third|fourth|fifth)\b)",
            content,
            re.IGNORECASE,
        )
        return len(set(markers))

    def _normalize_proposal_review_issue(self, issue: ReviewIssue) -> ReviewIssue:
        if issue.blocking:
            return issue.model_copy(update={"category": "blocking"})
        if issue.type in {"missing_citation", "citation_mismatch"}:
            return issue.model_copy(
                update={"blocking": False, "category": "citation_warning"}
            )
        if issue.type == "factual_error":
            return issue.model_copy(update={"blocking": True, "category": "blocking"})
        if issue.type == "format_issue":
            return issue.model_copy(update={"blocking": False, "category": "format"})
        return issue.model_copy(update={"blocking": False, "category": "content_quality"})

    def _is_proposal_deferred_claim(
        self, claim: ExtractedClaim, section_title: str
    ) -> bool:
        return not claim.citation_numbers and is_proposal_plan_sentence(
            claim.text, section_title
        )

    def _cited_sentence_context(self, content: str) -> str:
        cited = [
            claim.text
            for claim in self.claim_extractor.extract_cited_claims(content)
            if claim.citation_numbers
        ]
        if cited:
            return "\n".join(cited)
        return "\n".join(
            line.strip() for line in content.splitlines() if CITATION_PATTERN.search(line)
        )

    def _build_review_diagnostics(
        self,
        review: ReviewResult,
        quality_issues: list[QualityIssue],
        proposal_profile: bool,
        section: Any | None = None,
        state: AgentState | None = None,
    ) -> dict[str, Any]:
        buckets: dict[str, list[dict[str, Any]]] = {
            "structural_issues": [],
            "citation_issues": [],
            "evidence_binding_issues": [],
            "style_issues": [],
            "methodology_issues": [],
        }
        for review_issue in review.issues:
            buckets[self._review_issue_bucket(review_issue)].append(review_issue.model_dump())
        for quality_issue in quality_issues:
            bucket = self._quality_issue_bucket(quality_issue)
            item = quality_issue.model_dump()
            if item not in buckets[bucket]:
                buckets[bucket].append(item)
        issue_categories = {
            "blocking": [
                issue.model_dump() for issue in review.issues if issue.category == "blocking"
            ],
            "content_quality": [
                issue.model_dump()
                for issue in review.issues
                if issue.category == "content_quality"
            ],
            "citation_warning": [
                issue.model_dump()
                for issue in review.issues
                if issue.category == "citation_warning"
            ],
            "format": [
                issue.model_dump() for issue in review.issues if issue.category == "format"
            ],
        }
        section_title = getattr(section, "title", "") if section is not None else ""
        content = getattr(section, "content", "") if section is not None else ""
        return {
            **buckets,
            "proposal_profile": proposal_profile,
            "reviewer_mode": "proposal_review" if proposal_profile else "academic_review",
            "section_title": section_title,
            "section_budget": self._section_budget(section, state),
            "compound_title_detected": (
                is_compound_section_title(section_title) if section_title else False
            ),
            "missing_core_tasks": (
                missing_proposal_core_tasks(section_title, content)
                if proposal_profile and section_title
                else []
            ),
            "issue_categories": issue_categories,
            "blocking_issue_count": len(issue_categories["blocking"]),
            "deferred_quality_issue_count": sum(
                1 for issue in quality_issues if issue.details.get("deferred")
            ),
        }

    def _section_budget(self, section: Any | None, state: AgentState | None) -> int | str | None:
        if section is None:
            return None
        outline = state.get("outline") if state is not None else None
        if outline is not None:
            section_outline = next(
                (
                    outline_section
                    for outline_section in outline.sections
                    if outline_section.title == section.title
                ),
                None,
            )
            if section_outline is not None and section_outline.target_words is not None:
                return section_outline.target_words
        description = str(getattr(section, "description", "") or "")
        match = _BUDGET_RE.search(description)
        if match:
            return match.group(1)
        return None

    def _review_issue_bucket(self, issue: ReviewIssue) -> str:
        if issue.type in {"missing_citation", "citation_mismatch"}:
            if (
                "evidence packet" in issue.description
                or "Claim-source support" in issue.description
            ):
                return "evidence_binding_issues"
            return "citation_issues"
        if issue.type == "format_issue":
            return "structural_issues"
        if issue.type == "weak_argument":
            return "style_issues"
        return "structural_issues"

    def _quality_issue_bucket(self, issue: QualityIssue) -> str:
        source = str(issue.details.get("audit_source", ""))
        if source in {"claim_source", "evidence_binding"}:
            return "evidence_binding_issues"
        if source == "methodology":
            return "methodology_issues"
        if source in {"paragraph_quality", "language_style"}:
            return "style_issues"
        if "CITATION" in issue.code:
            return "citation_issues"
        return "structural_issues"

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
