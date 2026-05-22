from collections.abc import AsyncIterator
from typing import Any

from ..logging import get_logger
from ..state import AgentState, ClaimEvidenceBinding, EvidencePacket, QualityIssue
from . import CITATION_PATTERN, NodeStreamEvent

logger = get_logger(__name__)

_STRICT_MIN_SUPPORT_SCORE = 0.35
_STRICT_MIN_EVIDENCE_RELEVANCE = 0.2


def _is_blocking_issue(issue: QualityIssue) -> bool:
    return issue.blocking or issue.severity == "blocking"


class IntegrityGateNode:
    """Final quality gate that prevents completed status when blocking issues exist."""

    async def check(self, state: AgentState) -> dict[str, Any]:
        quality_issues = list(state.get("quality_issues") or [])
        strict_issues: list[QualityIssue] = []
        if state.get("strict_academic_mode", False):
            strict_issues = self._strict_academic_issues(state)
            quality_issues.extend(strict_issues)

        blocking_issues = [issue for issue in quality_issues if _is_blocking_issue(issue)]

        if blocking_issues:
            logger.warning(
                "integrity gate blocked completion",
                blocking_issue_count=len(blocking_issues),
                issue_codes=[issue.code for issue in blocking_issues],
            )
            return {
                "status": "waiting_human",
                "error_message": self._build_error_message(blocking_issues),
                "quality_issues": strict_issues,
            }

        logger.info("integrity gate passed")
        return {"status": "completed", "error_message": None, "quality_issues": strict_issues}

    async def astream(self, state: AgentState) -> AsyncIterator[NodeStreamEvent]:
        yield NodeStreamEvent(type="progress", progress="checking_integrity")
        result = await self.check(state)
        yield NodeStreamEvent(type="result", result=result)

    def _build_error_message(self, blocking_issues: list[QualityIssue]) -> str:
        first = blocking_issues[0]
        if len(blocking_issues) == 1:
            return first.message
        return f"{first.message} (+{len(blocking_issues) - 1} more blocking issue(s))"

    def _strict_academic_issues(self, state: AgentState) -> list[QualityIssue]:
        sections = state.get("sections", [])
        references = {ref.number: ref for ref in state.get("references", [])}
        bindings = list(state.get("claim_evidence_bindings") or [])
        packets = {packet.chunk_id: packet for packet in state.get("evidence_packets", [])}
        cited_numbers: set[int] = set()
        for section in sections:
            cited_numbers.update(int(num) for num in CITATION_PATTERN.findall(section.content))

        issues: list[QualityIssue] = []
        issues.extend(self._strict_reference_issues(cited_numbers, references))
        issues.extend(self._strict_binding_issues(cited_numbers, bindings, packets))
        return issues

    def _strict_reference_issues(
        self, cited_numbers: set[int], references: dict[int, Any]
    ) -> list[QualityIssue]:
        issues: list[QualityIssue] = []
        for citation_number in sorted(cited_numbers):
            ref = references.get(citation_number)
            if ref is None:
                issues.append(
                    self._blocking_issue(
                        "STRICT_INVALID_CITATION",
                        f"Citation [{citation_number}] has no generated reference entry.",
                        f"[{citation_number}]",
                    )
                )
                continue
            if not ref.doi:
                issues.append(
                    self._blocking_issue(
                        "STRICT_UNVERIFIED_REFERENCE",
                        f"Reference [{citation_number}] has no DOI for strict verification.",
                        f"[{citation_number}]",
                    )
                )
            elif not ref.metadata_verified:
                issues.append(
                    self._blocking_issue(
                        "STRICT_UNVERIFIED_REFERENCE",
                        (
                            f"Reference [{citation_number}] was not verified by CrossRef "
                            "or OpenAlex metadata."
                        ),
                        f"[{citation_number}]",
                    )
                )
        return issues

    def _strict_binding_issues(
        self,
        cited_numbers: set[int],
        bindings: list[ClaimEvidenceBinding],
        packets: dict[str, EvidencePacket],
    ) -> list[QualityIssue]:
        issues: list[QualityIssue] = []
        bindings_by_citation: dict[int, list[ClaimEvidenceBinding]] = {}
        for binding in bindings:
            bindings_by_citation.setdefault(binding.citation_number, []).append(binding)

        for citation_number in sorted(cited_numbers):
            citation_bindings = bindings_by_citation.get(citation_number, [])
            if not citation_bindings:
                issues.append(
                    self._blocking_issue(
                        "STRICT_MISSING_CHUNK_BINDING",
                        f"Citation [{citation_number}] has no claim-evidence binding.",
                        f"[{citation_number}]",
                    )
                )
                continue

            for binding in citation_bindings:
                issues.extend(self._strict_single_binding_issues(binding, packets))
        return issues

    def _strict_single_binding_issues(
        self, binding: ClaimEvidenceBinding, packets: dict[str, EvidencePacket]
    ) -> list[QualityIssue]:
        issues: list[QualityIssue] = []
        location = f"[{binding.citation_number}]"
        if not binding.chunk_id:
            issues.append(
                self._blocking_issue(
                    "STRICT_MISSING_CHUNK_BINDING",
                    f"Citation {location} is not bound to a chunk.",
                    location,
                )
            )
            return issues

        packet = packets.get(binding.chunk_id)
        if packet is None:
            issues.append(
                self._blocking_issue(
                    "STRICT_MISSING_CHUNK_BINDING",
                    f"Citation {location} references missing chunk '{binding.chunk_id}'.",
                    location,
                )
            )
            return issues

        if binding.verdict != "supported" or binding.support_score < _STRICT_MIN_SUPPORT_SCORE:
            issues.append(
                self._blocking_issue(
                    "STRICT_UNSUPPORTED_CLAIM",
                    (
                        f"Citation {location} has insufficient claim support "
                        f"({binding.verdict}, score={binding.support_score:.2f})."
                    ),
                    location,
                )
            )

        if packet.relevance_score < _STRICT_MIN_EVIDENCE_RELEVANCE or not packet.source_paper_id:
            issues.append(
                self._blocking_issue(
                    "STRICT_LOW_CONFIDENCE_SOURCE",
                    (
                        f"Citation {location} uses low-confidence evidence chunk "
                        f"'{packet.chunk_id}'."
                    ),
                    location,
                )
            )
        return issues

    def _blocking_issue(self, code: str, message: str, location: str) -> QualityIssue:
        return QualityIssue(
            code=code,
            message=message,
            severity="blocking",
            location=location,
            blocking=True,
        )
