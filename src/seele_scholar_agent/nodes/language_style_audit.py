import re
from dataclasses import dataclass
from typing import Literal

from ..state import AgentState
from ..style_packs import resolve_writing_locale

ReviewIssueType = Literal[
    "factual_error",
    "missing_citation",
    "weak_argument",
    "format_issue",
    "citation_mismatch",
    "other",
]


@dataclass(frozen=True)
class LanguageStyleFinding:
    code: str
    description: str
    suggestion: str
    location: str | None = None
    review_type: ReviewIssueType = "format_issue"


_CJK_RE = re.compile(r"[\u4e00-\u9fff]")
_CHINESE_SENTENCE_SPLIT_RE = re.compile(r"(?<=[。！？；])")

_ZH_TRANSLATIONESE_PATTERNS: tuple[tuple[str, str], ...] = (
    ("本文将会", "改为“本文将”或直接说明研究动作，避免英文 will 的直译感。"),
    ("本章节将会", "改为“本节将”或直接进入论述。"),
    ("起到了重要的作用", "改为具体机制或影响，说明究竟如何发挥作用。"),
    ("起到了至关重要的作用", "改为可验证的具体作用，不用空泛强化词。"),
    ("有着重要的意义", "说明意义体现在哪个理论、方法或应用层面。"),
    ("在另一方面", "改为“另一方面”或使用更自然的承接方式。"),
    ("被广泛地应用于", "改为“已用于”或说明应用场景与边界。"),
    ("对于", "仅在明显需要比较对象时使用，避免英文 for/to 的机械对应。"),
)


class LanguageStyleAudit:
    """Locale-aware writing style checks.

    Currently only zh-CN has deterministic checks. Other locales pass through so language-
    specific rules do not leak across writing tasks.
    """

    def audit(self, content: str, state: AgentState) -> list[LanguageStyleFinding]:
        locale = resolve_writing_locale(state)
        if not locale.startswith("zh"):
            return []
        if not _CJK_RE.search(content):
            return []
        return [
            *self._audit_zh_translationese(content),
            *self._audit_zh_sentence_length(content),
            *self._audit_term_glossary(content, state.get("term_glossary") or {}),
        ]

    def _audit_zh_translationese(self, content: str) -> list[LanguageStyleFinding]:
        findings: list[LanguageStyleFinding] = []
        seen: set[str] = set()
        for phrase, suggestion in _ZH_TRANSLATIONESE_PATTERNS:
            if phrase == "对于" and content.count("对于") < 3:
                continue
            if phrase in content and phrase not in seen:
                seen.add(phrase)
                findings.append(
                    LanguageStyleFinding(
                        code="ZH_TRANSLATIONESE_PHRASE",
                        description=f"中文表达存在明显翻译腔或模板化短语：{phrase}",
                        suggestion=suggestion,
                        location=phrase,
                    )
                )
        return findings

    def _audit_zh_sentence_length(self, content: str) -> list[LanguageStyleFinding]:
        findings: list[LanguageStyleFinding] = []
        sentences = [
            sentence.strip()
            for sentence in _CHINESE_SENTENCE_SPLIT_RE.split(content)
            if sentence.strip()
        ]
        for sentence in sentences:
            chinese_chars = len(_CJK_RE.findall(sentence))
            if chinese_chars < 120:
                continue
            findings.append(
                LanguageStyleFinding(
                    code="ZH_OVERLONG_SENTENCE",
                    description="中文句子过长，读者难以跟踪主语、谓语和论证关系。",
                    suggestion="拆分为两到三句，分别表达背景、证据和分析判断。",
                    location=sentence[:80],
                )
            )
            break
        return findings

    def _audit_term_glossary(
        self, content: str, term_glossary: dict[str, str]
    ) -> list[LanguageStyleFinding]:
        findings: list[LanguageStyleFinding] = []
        for source_term, preferred_term in term_glossary.items():
            if not source_term or not preferred_term:
                continue
            if source_term == preferred_term:
                continue
            if source_term in content:
                findings.append(
                    LanguageStyleFinding(
                        code="TERM_GLOSSARY_MISMATCH",
                        description=(
                            f"术语表要求使用“{preferred_term}”，但正文仍出现“{source_term}”。"
                        ),
                        suggestion=f"统一改为“{preferred_term}”，或更新主项目传入的术语表。",
                        location=source_term,
                    )
                )
        return findings
