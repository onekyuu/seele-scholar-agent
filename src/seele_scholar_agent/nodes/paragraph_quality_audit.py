import re
from dataclasses import dataclass
from typing import Literal

from ..state import SectionOutline
from . import CITATION_PATTERN

_WORD_RE = re.compile(r"[A-Za-z0-9_\u4e00-\u9fff]+")
_FIGURE_TABLE_RE = re.compile(r"\{\{(?:FIGURE|TABLE):.*?\}\}", re.IGNORECASE)
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[。！？.!?])\s+")
_NUMBER_RE = re.compile(r"\b\d+(?:\.\d+)?\s?%?\b")
_MIN_STRUCTURED_PARAGRAPH_CHARS = 120

_GENERIC_TEMPLATE_MARKERS = (
    "plays an important role",
    "it is important to note",
    "in today's world",
    "with the development of",
    "this section discusses",
    "this paper discusses",
    "has attracted increasing attention",
    "is a hot topic",
    "具有重要意义",
    "发挥着重要作用",
    "值得注意的是",
    "随着",
    "快速发展",
    "受到广泛关注",
)
_EVIDENCE_MARKERS = (
    "study",
    "studies",
    "evidence",
    "data",
    "experiment",
    "results",
    "finding",
    "findings",
    "survey",
    "研究",
    "证据",
    "数据",
    "实验",
    "结果",
    "发现",
    "调研",
)
_ANALYSIS_MARKERS = (
    "therefore",
    "thus",
    "because",
    "however",
    "suggests",
    "indicates",
    "implies",
    "this means",
    "this suggests",
    "in contrast",
    "因此",
    "所以",
    "然而",
    "但是",
    "说明",
    "表明",
    "意味着",
    "相比之下",
)


@dataclass(frozen=True)
class ParagraphQualityFinding:
    code: str
    review_type: Literal["weak_argument", "format_issue"]
    description: str
    suggestion: str
    location: str


class ParagraphQualityAudit:
    """Paragraph-level style and argument structure checks."""

    def audit(
        self,
        *,
        section_title: str,
        content: str,
        section_outline: SectionOutline | None = None,
        proposal_profile: bool = False,
    ) -> list[ParagraphQualityFinding]:
        paragraphs = self._paragraphs(content)
        findings: list[ParagraphQualityFinding] = []
        findings.extend(self._duplicate_paragraph_findings(paragraphs, section_title))
        findings.extend(self._generic_template_findings(paragraphs, section_title))
        findings.extend(
            self._target_claim_findings(content, section_outline, section_title)
        )
        if not proposal_profile:
            findings.extend(self._structure_findings(paragraphs, section_title))
        return findings

    def _paragraphs(self, content: str) -> list[str]:
        clean_content = _FIGURE_TABLE_RE.sub(" ", content)
        return [paragraph.strip() for paragraph in clean_content.split("\n\n") if paragraph.strip()]

    def _duplicate_paragraph_findings(
        self, paragraphs: list[str], section_title: str
    ) -> list[ParagraphQualityFinding]:
        normalized_seen: dict[str, int] = {}
        findings: list[ParagraphQualityFinding] = []
        for index, paragraph in enumerate(paragraphs, 1):
            normalized = self._normalize_paragraph(paragraph)
            if len(normalized) < 40:
                continue
            first_seen = normalized_seen.get(normalized)
            if first_seen is not None:
                findings.append(
                    ParagraphQualityFinding(
                        code="PARAGRAPH_DUPLICATE",
                        review_type="format_issue",
                        description=(
                            f"Paragraph {index} duplicates paragraph {first_seen} "
                            f"in section '{section_title}'."
                        ),
                        suggestion=(
                            "Remove the duplicate paragraph or replace it with new analysis."
                        ),
                        location=f"{section_title}:paragraph:{index}",
                    )
                )
            else:
                normalized_seen[normalized] = index
        return findings

    def _generic_template_findings(
        self, paragraphs: list[str], section_title: str
    ) -> list[ParagraphQualityFinding]:
        findings: list[ParagraphQualityFinding] = []
        for index, paragraph in enumerate(paragraphs, 1):
            lowered = paragraph.casefold()
            if not any(marker in lowered for marker in _GENERIC_TEMPLATE_MARKERS):
                continue
            if self._has_evidence(paragraph) or self._has_analysis(paragraph):
                continue
            findings.append(
                ParagraphQualityFinding(
                    code="PARAGRAPH_GENERIC_TEMPLATE",
                    review_type="weak_argument",
                    description=(
                        f"Paragraph {index} in section '{section_title}' uses generic "
                        "template language without concrete evidence or analysis."
                    ),
                    suggestion=(
                        "Replace generic framing with section-specific claims, evidence, "
                        "and analysis."
                    ),
                    location=f"{section_title}:paragraph:{index}",
                )
            )
        return findings

    def _target_claim_findings(
        self,
        content: str,
        section_outline: SectionOutline | None,
        section_title: str,
    ) -> list[ParagraphQualityFinding]:
        if section_outline is None or not section_outline.target_claims:
            return []

        content_tokens = self._tokens(content)
        findings: list[ParagraphQualityFinding] = []
        for claim in section_outline.target_claims:
            claim_tokens = self._tokens(claim)
            if not claim_tokens:
                continue
            overlap_ratio = len(claim_tokens & content_tokens) / len(claim_tokens)
            if overlap_ratio >= 0.35:
                continue
            findings.append(
                ParagraphQualityFinding(
                    code="SECTION_TARGET_CLAIM_UNCOVERED",
                    review_type="weak_argument",
                    description=(
                        f"Section '{section_title}' does not cover target claim: {claim}"
                    ),
                    suggestion=(
                        "Revise the section to explicitly address this planned target claim."
                    ),
                    location=section_title,
                )
            )
        return findings

    def _structure_findings(
        self, paragraphs: list[str], section_title: str
    ) -> list[ParagraphQualityFinding]:
        findings: list[ParagraphQualityFinding] = []
        substantive = [
            (index, paragraph)
            for index, paragraph in enumerate(paragraphs, 1)
            if len(paragraph) >= _MIN_STRUCTURED_PARAGRAPH_CHARS
        ]
        if not substantive:
            return []

        for index, paragraph in substantive:
            missing_parts = []
            if not self._has_topic_sentence(paragraph):
                missing_parts.append("topic sentence")
            if not self._has_evidence(paragraph):
                missing_parts.append("evidence")
            if not self._has_analysis(paragraph):
                missing_parts.append("analysis")
            if not missing_parts:
                continue
            findings.append(
                ParagraphQualityFinding(
                    code="PARAGRAPH_STRUCTURE_INCOMPLETE",
                    review_type="weak_argument",
                    description=(
                        f"Paragraph {index} in section '{section_title}' lacks "
                        f"{', '.join(missing_parts)}."
                    ),
                    suggestion=(
                        "Revise the paragraph to include a clear topic sentence, "
                        "supporting evidence, and interpretive analysis."
                    ),
                    location=f"{section_title}:paragraph:{index}",
                )
            )
        return findings

    def _has_topic_sentence(self, paragraph: str) -> bool:
        first_sentence = self._sentences(paragraph)[0] if self._sentences(paragraph) else paragraph
        return len(self._tokens(first_sentence)) >= 5

    def _has_evidence(self, paragraph: str) -> bool:
        lowered = paragraph.casefold()
        return bool(
            CITATION_PATTERN.search(paragraph)
            or _NUMBER_RE.search(paragraph)
            or any(marker in lowered for marker in _EVIDENCE_MARKERS)
        )

    def _has_analysis(self, paragraph: str) -> bool:
        lowered = paragraph.casefold()
        return any(marker in lowered for marker in _ANALYSIS_MARKERS)

    def _sentences(self, paragraph: str) -> list[str]:
        return [
            sentence.strip()
            for sentence in _SENTENCE_SPLIT_RE.split(paragraph)
            if sentence.strip()
        ]

    def _normalize_paragraph(self, paragraph: str) -> str:
        return " ".join(paragraph.casefold().split())

    def _tokens(self, text: str) -> set[str]:
        return {match.group(0).casefold() for match in _WORD_RE.finditer(text)}
