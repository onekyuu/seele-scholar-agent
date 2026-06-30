import re
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from difflib import SequenceMatcher
from typing import Any, Literal

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from ..agent_config import PromptsConfig
from ..config import settings
from ..document_profile import is_compound_section_title
from ..i18n import t
from ..logging import get_logger
from ..policy import SectionExecutionStrategy, WritingPolicy
from ..profiles import DocumentProfile, get_document_profile
from ..review import (
    ReviewDecision,
    SectionCandidate,
    decide_review_action,
    select_best_candidate,
)
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
    def __init__(
        self,
        llm: ChatOpenAI,
        prompts: PromptsConfig,
        execution_strategy: SectionExecutionStrategy | None = None,
        writing_policy: WritingPolicy | None = None,
    ):
        self.llm = llm
        self.prompts = prompts
        self.execution_strategy = execution_strategy or SectionExecutionStrategy()
        self.writing_policy = writing_policy or WritingPolicy()
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
        document_profile = get_document_profile(state)

        logger.info("reviewing section", title=section.title)

        try:
            result = await invoke_with_retry(
                self.chain,
                {
                    "topic": state["topic"],
                    "section_title": section.title,
                    "content": section.content,
                    "document_type": document_profile.review_document_type,
                    "review_policy": document_profile.review_policy_text(),
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
                section.title, section.content, papers, document_profile=document_profile
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
            document_profile=document_profile,
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
            document_profile=document_profile,
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

        structural_issues, structural_quality_issues = (
            document_profile.structural_review_issues(
                section.section_id, section.title, section.content
            )
        )
        quality_issues = [*quality_issues, *structural_quality_issues]
        if structural_issues:
            review.issues.extend(structural_issues)
            if review.approved:
                review = review.model_copy(update={"approved": False})

        review, quality_issues = document_profile.apply_review_policy(
            review, quality_issues
        )

        record = {
            "section": section.title,
            "score": review.score,
            "approved": review.approved,
            "timestamp": datetime.now(UTC).isoformat(),
        }
        candidate = self._section_candidate(section, review, quality_issues)
        decision = decide_review_action(
            review,
            quality_issues,
            revision_count=section.revision_count,
            max_revisions=self._max_revisions(state),
            writing_policy=self.writing_policy,
            generation_mode=self.execution_strategy.policy.generation_mode,
        )

        if decision.action == "pass":
            result = await self._handle_approved(state, review, record, candidate, decision)
        elif decision.action == "accept_best":
            result = await self._handle_accept_best(
                state, section, review, record, quality_issues, candidate, decision
            )
        else:
            result = await self._handle_rejected(
                state, section, review, record, quality_issues, candidate, decision
            )
        if quality_issues and "quality_issues" not in result:
            result["quality_issues"] = quality_issues
        result["review_diagnostics"] = self._build_review_diagnostics(
            review, quality_issues, document_profile, section, state
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
        document_profile = get_document_profile(state)
        yield NodeStreamEvent(type="progress", progress=f"reviewing:{section.title}")

        input_data = {
            "topic": state["topic"],
            "section_title": section.title,
            "content": section.content,
            "document_type": document_profile.review_document_type,
            "review_policy": document_profile.review_policy_text(),
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
                section.title, section.content, papers, document_profile=document_profile
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
            document_profile=document_profile,
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
            document_profile=document_profile,
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

        structural_issues, structural_quality_issues = (
            document_profile.structural_review_issues(
                section.section_id, section.title, section.content
            )
        )
        quality_issues = [*quality_issues, *structural_quality_issues]
        if structural_issues:
            review.issues.extend(structural_issues)
            if review.approved:
                review = review.model_copy(update={"approved": False})

        review, quality_issues = document_profile.apply_review_policy(
            review, quality_issues
        )

        record = {
            "section": section.title,
            "score": review.score,
            "approved": review.approved,
            "timestamp": datetime.now(UTC).isoformat(),
        }
        candidate = self._section_candidate(section, review, quality_issues)
        decision = decide_review_action(
            review,
            quality_issues,
            revision_count=section.revision_count,
            max_revisions=self._max_revisions(state),
            writing_policy=self.writing_policy,
            generation_mode=self.execution_strategy.policy.generation_mode,
        )

        if decision.action == "pass":
            final_result = await self._handle_approved(
                state, review, record, candidate, decision
            )
        elif decision.action == "accept_best":
            final_result = await self._handle_accept_best(
                state, section, review, record, quality_issues, candidate, decision
            )
        else:
            final_result = await self._handle_rejected(
                state, section, review, record, quality_issues, candidate, decision
            )
        if quality_issues and "quality_issues" not in final_result:
            final_result["quality_issues"] = quality_issues
        final_result["review_diagnostics"] = self._build_review_diagnostics(
            review, quality_issues, document_profile, section, state
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
        document_profile: DocumentProfile,
    ) -> list[ReviewIssue]:
        cited_numbers = {int(m) for m in CITATION_PATTERN.findall(content)}
        valid_cited = {n for n in cited_numbers if 1 <= n <= len(papers)}
        if not valid_cited:
            return []

        content_for_alignment = (
            self._cited_sentence_context(content)
            if document_profile.citation_alignment_uses_cited_context()
            else content
        )
        category = document_profile.citation_review_category()
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
                    category=category,
                )
                for item in raw_issues
            ]
        except Exception as e:
            logger.warning("citation alignment check failed", error=str(e))
            return []

    def _audit_claim_source_support(
        self,
        section_id: str,
        section_title: str,
        content: str,
        bindings: list[ClaimEvidenceBinding],
        *,
        document_profile: DocumentProfile,
    ) -> tuple[list[ReviewIssue], list[QualityIssue]]:
        citation_category = document_profile.citation_review_category()
        section_bindings = [binding for binding in bindings if binding.section_id == section_id]
        issues: list[ReviewIssue] = []
        quality_issues: list[QualityIssue] = []
        claims = self.claim_extractor.extract_factual_claims(content)

        bindings_by_citation: dict[int, list[ClaimEvidenceBinding]] = {}
        for binding in section_bindings:
            bindings_by_citation.setdefault(binding.citation_number, []).append(binding)

        for claim in claims:
            if document_profile.should_defer_claim(
                claim.text, claim.citation_numbers, section_title
            ):
                continue

            if not claim.citation_numbers:
                issue = ReviewIssue(
                    type="missing_citation",
                    description=f"Factual claim lacks citation: {claim.text}",
                    suggestion="Add a valid citation backed by an evidence packet or revise it.",
                    location=claim.text[:80],
                    blocking=False,
                    category=citation_category,
                )
                quality_issue = self._claim_quality_issue(
                    "UNSUPPORTED_CLAIM", issue, section_id, claim
                )
                quality_issue = document_profile.claim_source_quality_issue(
                    quality_issue,
                    audit_source="claim_source",
                )
                if document_profile.should_emit_claim_source_review_issue("missing_citation"):
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
                        category=citation_category,
                    )
                    quality_issue = self._claim_quality_issue(
                        "CLAIM_MISSING_EVIDENCE_PACKET", issue, section_id, claim
                    )
                    quality_issue = document_profile.claim_source_quality_issue(
                        quality_issue,
                        audit_source="evidence_binding",
                    )
                    if document_profile.should_emit_claim_source_review_issue(
                        "missing_evidence_packet"
                    ):
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
                issue = issue.model_copy(update={"category": citation_category})
                quality_issue = self._claim_quality_issue(
                    "UNSUPPORTED_CLAIM", issue, section_id, claim
                )
                quality_issue = document_profile.claim_source_quality_issue(
                    quality_issue,
                    audit_source="evidence_binding",
                    binding_diagnostics=best_binding.diagnostics,
                )
                if document_profile.should_emit_claim_source_review_issue("unsupported_binding"):
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
            issue = issue.model_copy(update={"category": citation_category})
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
            quality_issue = document_profile.claim_source_quality_issue(
                quality_issue,
                audit_source="evidence_binding",
                binding_diagnostics=binding.diagnostics,
            )
            if document_profile.should_emit_claim_source_review_issue("unsupported_binding"):
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
        document_profile = get_document_profile(state)
        findings = self.methodology_audit.audit(
            section_title=section_title,
            content=content,
            paper_type=paper_type,
            structure_pattern=structure_pattern,
            skip_methodology_audit=document_profile.skip_methodology_audit(content),
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
        document_profile: DocumentProfile,
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
            include_structure_check=document_profile.include_paragraph_structure_check(),
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
        document_profile: DocumentProfile,
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
        profile_fields = document_profile.review_diagnostic_fields(section_title, content)
        return {
            **buckets,
            **profile_fields,
            "section_title": section_title,
            "section_budget": self._section_budget(section, state),
            "compound_title_detected": (
                is_compound_section_title(section_title) if section_title else False
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
        self,
        state: AgentState,
        review: ReviewResult,
        record: dict[str, Any],
        candidate: SectionCandidate,
        decision: ReviewDecision,
    ) -> dict[str, Any]:
        return {
            **self.execution_strategy.approved_section_delta(
                state, section_status="approved"
            ),
            "review_history": [record],
            "section_candidates": [candidate],
            "current_review": review.model_dump(),
            "review_decision": decision.model_dump(),
        }

    async def _handle_accept_best(
        self,
        state: AgentState,
        section: Any,
        review: ReviewResult,
        record: dict[str, Any],
        quality_issues: list[QualityIssue],
        candidate: SectionCandidate,
        decision: ReviewDecision,
    ) -> dict[str, Any]:
        candidates = self._section_candidates_for_section(state, section.section_id)
        candidates.append(candidate)
        best = select_best_candidate(candidates)

        sections = list(state["sections"])
        index = state["current_section_index"]
        sections[index] = section.model_copy(update={"content": best.content})
        state_for_strategy = {**state, "sections": sections}
        quality_issue = self._max_revisions_quality_issue(
            section,
            review,
            max_revisions=self._max_revisions(state),
            severity="warning",
            blocking=False,
            accepted_with_issues=True,
        )

        return {
            **self.execution_strategy.approved_section_delta(
                state_for_strategy, section_status="accepted_with_issues"
            ),
            "review_history": [record],
            "section_candidates": [candidate],
            "current_review": review.model_dump(),
            "review_decision": decision.model_dump(),
            "quality_issues": [quality_issue, *quality_issues],
        }

    async def _handle_rejected(
        self,
        state: AgentState,
        section: Any,
        review: ReviewResult,
        record: dict[str, Any],
        quality_issues: list[QualityIssue] | None = None,
        candidate: SectionCandidate | None = None,
        decision: ReviewDecision | None = None,
    ) -> dict[str, Any]:
        sections = state["sections"]
        index = state["current_section_index"]
        total_revision_count = state.get("revision_count", 0)
        section_revision_count = section.revision_count
        max_revisions = self._max_revisions(state)
        lang = state.get("language", "zh")

        if decision is not None and decision.action == "human_required":
            logger.warning(
                "max revisions reached, blocking section instead of forcing approval",
                section=section.title,
                revision_count=section_revision_count,
                max_revisions=max_revisions,
            )
            updated = sections.copy()
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

            quality_issue = self._max_revisions_quality_issue(
                section,
                review,
                max_revisions=max_revisions,
                severity="blocking",
                blocking=True,
                accepted_with_issues=False,
            )

            result: dict[str, Any] = {
                "sections": updated,
                "review_history": [record],
                "section_candidates": [candidate] if candidate is not None else [],
                "current_review": review.model_dump(),
                "review_decision": decision.model_dump(),
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
            "section_candidates": [candidate] if candidate is not None else [],
            "current_review": review.model_dump(),
            "review_decision": decision.model_dump() if decision is not None else None,
            "revision_count": total_revision_count + 1,
            "status": "writing",
        }
        if quality_issues:
            result["quality_issues"] = quality_issues
        return result

    def _max_revisions(self, state: AgentState) -> int:
        return int(state.get("max_revisions", self.writing_policy.max_revisions))

    def _section_candidate(
        self,
        section: Any,
        review: ReviewResult,
        quality_issues: list[QualityIssue],
    ) -> SectionCandidate:
        blocking_issues = [
            issue for issue in quality_issues if issue.blocking or issue.severity == "blocking"
        ]
        warnings = [issue for issue in quality_issues if issue not in blocking_issues]
        return SectionCandidate(
            section_id=section.section_id,
            content=section.content,
            revision_count=section.revision_count,
            review_score=review.score,
            blocking_issues=blocking_issues,
            warnings=warnings,
            length=len(section.content),
            diagnostics={"review_summary": review.summary},
        )

    def _section_candidates_for_section(
        self, state: AgentState, section_id: str
    ) -> list[SectionCandidate]:
        candidates: list[SectionCandidate] = []
        for item in state.get("section_candidates", []):
            candidate = item if isinstance(item, SectionCandidate) else SectionCandidate(**item)
            if candidate.section_id == section_id:
                candidates.append(candidate)
        return candidates

    def _max_revisions_quality_issue(
        self,
        section: Any,
        review: ReviewResult,
        *,
        max_revisions: int,
        severity: Literal["warning", "blocking"],
        blocking: bool,
        accepted_with_issues: bool,
    ) -> QualityIssue:
        issue_summaries = [issue.description for issue in review.issues if issue.description]
        section_revision_count = section.revision_count
        return QualityIssue(
            code="MAX_REVISIONS_REACHED",
            message=(
                f"Section '{section.title}' failed review after "
                f"{section_revision_count} revision(s)."
            ),
            severity=severity,
            location=section.section_id,
            blocking=blocking,
            details={
                "section_title": section.title,
                "revision_count": section_revision_count,
                "max_revisions": max_revisions,
                "review_score": review.score,
                "review_summary": review.summary,
                "review_issues": issue_summaries,
                "accepted_with_issues": accepted_with_issues,
                "recommended_action": f"Review and revise section '{section.title}'.",
            },
        )
