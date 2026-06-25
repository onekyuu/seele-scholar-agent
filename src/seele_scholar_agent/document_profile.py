import re
from collections.abc import Mapping
from typing import Any

from .state import AgentState

_PROPOSAL_TYPES = {
    "research_proposal",
    "proposal",
    "graduate_research_proposal",
    "masters_research_proposal",
    "master_research_proposal",
    "研究計画書",
    "研究计划书",
}

_SCHEDULE_TITLE_MARKERS = (
    "schedule",
    "timeline",
    "スケジュール",
    "时间表",
    "時間表",
)

_PLAN_MARKERS = (
    "予定",
    "計画",
    "実施",
    "検証",
    "開発",
    "評価",
    "執筆",
    "作成",
    "行う",
    "進める",
    "取り組む",
    "目指す",
    "本研究では",
    "本研究は",
    "拟",
    "计划",
    "开展",
    "实施",
    "验证",
    "开发",
    "评价",
    "评估",
    "撰写",
    "will",
    "plan to",
    "aim to",
)
_OWN_PLAN_MARKERS = (
    "本研究では",
    "本研究は",
    "本稿では",
    "本計画",
    "拟",
    "计划",
    "we will",
    "we plan",
    "we aim",
    "this research will",
)

_LITERATURE_MARKERS = (
    "先行研究",
    "既存研究",
    "既往研究",
    "研究表明",
    "结果显示",
    "実験結果",
    "結果は",
    "previous studies",
    "prior work",
    "the literature",
    "studies show",
    "results show",
)

_SCHEDULE_PHASES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("1年次前期", ("1年次前期", "1 年次前期", "一年次前期", "1年目前期")),
    ("1年次後期", ("1年次後期", "1 年次後期", "一年次後期", "1年目後期")),
    ("2年次前期", ("2年次前期", "2 年次前期", "二年次前期", "2年目前期")),
    ("2年次後期", ("2年次後期", "2 年次後期", "二年次後期", "2年目後期")),
)

DEFAULT_PROPOSAL_TARGET_CHARS = 2200

_COMPOUND_TITLE_SEPARATORS = ("・", "/", "／", "&", " and ", "、")

_CORE_TASK_TITLE_MARKERS: dict[str, tuple[str, ...]] = {
    "background": ("研究背景", "背景", "問題意識", "motivation", "background"),
    "prior_work": ("先行研究", "既存研究", "課題", "literature", "prior work"),
    "purpose": ("研究目的", "目的", "研究課題", "research question", "objective"),
    "method": ("研究方法", "方法", "method"),
    "plan": ("研究計画", "計画", "schedule", "timeline", "スケジュール"),
    "novelty": ("新規性", "独自性", "novelty", "originality"),
    "outcome": ("期待される成果", "期待成果", "成果", "貢献", "将来展望", "outcome"),
}

_CORE_TASK_CONTENT_MARKERS: dict[str, tuple[str, ...]] = {
    "background": (
        "背景",
        "問題",
        "必要",
        "関心",
        "動機",
        "先行研究",
        "既存研究",
        "motivation",
        "background",
    ),
    "prior_work": ("先行研究", "既存研究", "課題", "未解決", "限界", "prior work"),
    "purpose": ("目的", "明らか", "解明", "目指", "研究課題", "問い", "RQ", "objective"),
    "method": (
        "方法",
        "分析",
        "検証",
        "資料",
        "データ",
        "ツール",
        "設計",
        "実装",
        "開発",
        "調査",
        "比較",
        "用い",
        "method",
    ),
    "plan": (
        "計画",
        "予定",
        "段階",
        "年次",
        "前期",
        "後期",
        "修士",
        "期間",
        "進め",
        "schedule",
        "timeline",
    ),
    "novelty": ("新規", "独自", "特徴", "差別", "未解決", "novel", "original"),
    "outcome": ("成果", "貢献", "期待", "価値", "意義", "展望", "可能性", "outcome"),
}


def get_document_type(state: AgentState | Mapping[str, Any]) -> str:
    """Resolve the caller-supplied document type from known state locations."""

    candidates: list[Any] = [state.get("document_type")]
    generation_config = state.get("generation_config")
    metadata = state.get("metadata")
    candidates.append(_get_attr_or_key(generation_config, "document_type"))
    candidates.append(_get_attr_or_key(metadata, "document_type"))
    candidates.append(state.get("paper_type"))

    for value in candidates:
        if value is None:
            continue
        normalized = str(value).strip()
        if normalized:
            return normalized
    return ""


def is_research_proposal(state: AgentState | Mapping[str, Any]) -> bool:
    normalized = get_document_type(state).casefold().replace("-", "_").replace(" ", "_")
    return normalized in _PROPOSAL_TYPES or "research_proposal" in normalized


def get_config_value(
    state: AgentState | Mapping[str, Any],
    key: str,
    default: Any = None,
) -> Any:
    """Resolve a config value from state, generation_config, or metadata."""

    for source in (state, state.get("generation_config"), state.get("metadata")):
        value = _get_attr_or_key(source, key)
        if value is not None:
            return value
    return default


def get_target_word_count(state: AgentState | Mapping[str, Any]) -> int | None:
    raw = get_config_value(state, "target_word_count")
    if raw is None:
        raw = get_config_value(state, "target_chars")
    if raw is None:
        raw = get_config_value(state, "target_character_count")
    if raw is None:
        return DEFAULT_PROPOSAL_TARGET_CHARS if is_research_proposal(state) else None
    try:
        return int(raw)
    except (TypeError, ValueError):
        return DEFAULT_PROPOSAL_TARGET_CHARS if is_research_proposal(state) else None


def is_schedule_section(section_title: str) -> bool:
    lowered = section_title.casefold()
    return any(marker.casefold() in lowered for marker in _SCHEDULE_TITLE_MARKERS)


def is_compound_section_title(section_title: str) -> bool:
    lowered = section_title.casefold()
    return any(separator in lowered for separator in _COMPOUND_TITLE_SEPARATORS)


def proposal_core_tasks_for_title(section_title: str) -> list[str]:
    """Infer the application-review tasks implied by a proposal section title."""

    lowered = section_title.casefold()
    tasks: list[str] = []
    for task, markers in _CORE_TASK_TITLE_MARKERS.items():
        if any(marker.casefold() in lowered for marker in markers):
            tasks.append(task)
    if not tasks and "研究" in section_title:
        return ["purpose"]
    return tasks


def missing_proposal_core_tasks(section_title: str, content: str) -> list[str]:
    """Return title-implied proposal tasks absent at overview level from section content."""

    lowered = content.casefold()
    missing: list[str] = []
    for task in proposal_core_tasks_for_title(section_title):
        markers = _CORE_TASK_CONTENT_MARKERS[task]
        if not any(marker.casefold() in lowered for marker in markers):
            missing.append(task)
    return missing


def missing_schedule_phases(content: str) -> list[str]:
    return [
        phase
        for phase, aliases in _SCHEDULE_PHASES
        if not any(alias in content for alias in aliases)
    ]


def is_proposal_plan_sentence(sentence: str, section_title: str = "") -> bool:
    normalized = sentence.casefold()
    if any(marker.casefold() in normalized for marker in _LITERATURE_MARKERS):
        return False
    has_plan_marker = any(marker.casefold() in normalized for marker in _PLAN_MARKERS)
    has_own_plan_marker = any(marker.casefold() in normalized for marker in _OWN_PLAN_MARKERS)
    has_schedule_marker = bool(re.search(r"[12]\s*年次|[一二]年次|前期|後期", sentence))
    return has_plan_marker and (
        has_own_plan_marker or has_schedule_marker or is_schedule_section(section_title)
    )


def _get_attr_or_key(value: Any, key: str) -> Any:
    if value is None:
        return None
    if isinstance(value, Mapping):
        return value.get(key)
    return getattr(value, key, None)
