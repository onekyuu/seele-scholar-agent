import re
from dataclasses import dataclass

from . import CITATION_PATTERN

_SENTENCE_RE = re.compile(r"[^。！？.!?\n]+(?:[。！？.!?]|$)")
_NUMERIC_RE = re.compile(r"(\d{2,4}|[0-9]+(?:\.[0-9]+)?\s?%|p\s?[<=>]\s?0\.\d+)")
_FIGURE_TABLE_RE = re.compile(r"\{\{(?:FIGURE|TABLE):.*?\}\}", re.IGNORECASE)

_FACTUAL_MARKERS = (
    "show",
    "shows",
    "shown",
    "demonstrate",
    "demonstrates",
    "find",
    "finds",
    "found",
    "indicate",
    "indicates",
    "suggest",
    "suggests",
    "improve",
    "improves",
    "reduce",
    "reduces",
    "increase",
    "increases",
    "outperform",
    "outperforms",
    "significant",
    "correlate",
    "correlates",
    "cause",
    "causes",
    "is associated with",
    "are associated with",
    "研究表明",
    "结果显示",
    "实验表明",
    "发现",
    "证明",
    "表明",
    "显示",
    "指出",
    "提升",
    "提高",
    "降低",
    "减少",
    "优于",
    "显著",
    "相关",
    "导致",
)

_RHETORICAL_MARKERS = (
    "this section",
    "this paper",
    "we discuss",
    "we describe",
    "we introduce",
    "we first",
    "we then",
    "in summary",
    "overall",
    "本文",
    "本节",
    "本章",
    "下文",
    "接下来",
    "综上",
)


@dataclass(frozen=True)
class ExtractedClaim:
    text: str
    citation_numbers: tuple[int, ...]
    is_factual: bool


class RuleBasedClaimExtractor:
    """Rule-based factual claim extractor used before LLM review."""

    def extract_factual_claims(self, content: str) -> list[ExtractedClaim]:
        return [claim for claim in self.extract_claims(content) if claim.is_factual]

    def extract_cited_claims(self, content: str) -> list[ExtractedClaim]:
        return [claim for claim in self.extract_claims(content) if claim.citation_numbers]

    def extract_claims(self, content: str) -> list[ExtractedClaim]:
        claims: list[ExtractedClaim] = []
        for sentence in self._split_sentences(content):
            citation_numbers = tuple(
                sorted({int(number) for number in CITATION_PATTERN.findall(sentence)})
            )
            claims.append(
                ExtractedClaim(
                    text=sentence,
                    citation_numbers=citation_numbers,
                    is_factual=self._is_factual_claim(sentence, citation_numbers),
                )
            )
        return claims

    def _split_sentences(self, content: str) -> list[str]:
        clean_content = _FIGURE_TABLE_RE.sub(" ", content)
        sentences: list[str] = []
        for raw_line in clean_content.splitlines():
            line = raw_line.strip()
            if not line or line.startswith(("#", "-", "*", "|")):
                continue
            matches = [match.group(0).strip() for match in _SENTENCE_RE.finditer(line)]
            if matches:
                sentences.extend(match for match in matches if match)
            else:
                sentences.append(line)
        return sentences

    def _is_factual_claim(self, sentence: str, citation_numbers: tuple[int, ...]) -> bool:
        if citation_numbers:
            return True

        normalized = sentence.casefold()
        if "?" in sentence or "？" in sentence:
            return False
        if any(marker in normalized for marker in _RHETORICAL_MARKERS):
            return False
        if _NUMERIC_RE.search(sentence):
            return True
        return any(marker in normalized for marker in _FACTUAL_MARKERS)
