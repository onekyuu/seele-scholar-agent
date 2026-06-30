import re
from typing import Any

from ..state import OutlineStructure, SectionDraft
from .models import (
    DraftIntegrationState,
    DraftIntent,
    DraftOutlineMapping,
    DraftRole,
    DraftSectionContext,
    DraftSegment,
    ExistingContentRef,
    OutlineAdaptationDecision,
)

_WORD_RE = re.compile(r"[A-Za-z0-9_\u4e00-\u9fff]+")

_ROLE_KEYWORDS: dict[DraftRole, tuple[str, ...]] = {
    "title": ("title", "标题", "題名"),
    "abstract": ("abstract", "摘要", "要旨"),
    "background": ("introduction", "background", "背景", "研究背景", "序論", "引言"),
    "prior_work": ("related", "prior", "先行研究", "文献", "literature"),
    "purpose": ("purpose", "objective", "目的", "研究目的"),
    "method": ("method", "methods", "方法", "研究方法"),
    "plan": ("plan", "schedule", "スケジュール", "計画", "计划"),
    "expected_outcome": ("outcome", "contribution", "成果", "期待", "貢献"),
    "conclusion": ("conclusion", "结论", "結論", "総括"),
    "other": (),
}


def build_draft_state(
    existing_content: ExistingContentRef,
    *,
    outline: OutlineStructure | None,
    sections: list[SectionDraft],
) -> DraftIntegrationState:
    normalized_content = _with_inferred_roles(existing_content)
    mappings = _map_segments(normalized_content.segments, sections)
    conflicts = _detect_conflicts(normalized_content.segments)
    return DraftIntegrationState(
        existing_content=normalized_content,
        mappings=mappings,
        outline_decision=_outline_decision(
            outline=outline,
            sections=sections,
            mappings=mappings,
            segment_count=len(normalized_content.segments),
            conflicts=conflicts,
        ),
        uncovered_requirements=_uncovered_segments(normalized_content.segments, mappings),
        conflicts=conflicts,
    )


def build_draft_section_context(
    draft_state: DraftIntegrationState | None,
    *,
    sections: list[SectionDraft],
    current_index: int,
) -> DraftSectionContext | None:
    if draft_state is None:
        return None
    if current_index < 0 or current_index >= len(sections):
        return None

    section = sections[current_index]
    segment_by_id = {
        segment.segment_id: segment
        for segment in draft_state.existing_content.segments
        if segment.text.strip()
    }
    mapped_segment_ids = {
        mapping.segment_id
        for mapping in draft_state.mappings
        if mapping.section_id == section.section_id
    }
    mapped_segments = [segment_by_id[segment_id] for segment_id in mapped_segment_ids]

    unmapped = [
        segment
        for segment_id, segment in segment_by_id.items()
        if segment_id not in mapped_segment_ids
    ]
    related = [
        segment
        for segment in unmapped
        if _section_segment_score(section, segment) >= 0.15
        or not draft_state.mappings
    ][:3]

    if not mapped_segments and not related:
        return None

    return DraftSectionContext(
        mapped_segments=sorted(mapped_segments, key=lambda segment: segment.order),
        unmapped_related_segments=sorted(related, key=lambda segment: segment.order),
        preserve_policy=draft_state.existing_content.preserve_policy,
        user_intent=draft_state.existing_content.user_intent,
    )


def _with_inferred_roles(existing_content: ExistingContentRef) -> ExistingContentRef:
    segments = [
        segment.model_copy(
            update={
                "inferred_role": _infer_role(segment),
                "confidence": max(
                    segment.confidence,
                    0.55 if segment.inferred_role == "other" else 0.0,
                ),
            }
        )
        for segment in existing_content.segments
    ]
    return existing_content.model_copy(update={"segments": segments})


def _infer_role(segment: DraftSegment) -> DraftRole:
    if segment.inferred_role != "other":
        return segment.inferred_role
    text = " ".join(
        part for part in (segment.detected_heading or "", segment.text[:200]) if part
    ).casefold()
    for role, keywords in _ROLE_KEYWORDS.items():
        if role == "other":
            continue
        if any(keyword.casefold() in text for keyword in keywords):
            return role
    return "other"


def _map_segments(
    segments: list[DraftSegment], sections: list[SectionDraft]
) -> list[DraftOutlineMapping]:
    if not sections:
        return []

    mappings: list[DraftOutlineMapping] = []
    for segment in segments:
        candidates = [
            (_section_segment_score(section, segment), section)
            for section in sections
        ]
        score, section = max(candidates, key=lambda item: item[0])
        if score < 0.15:
            mappings.append(
                DraftOutlineMapping(
                    segment_id=segment.segment_id,
                    section_id=None,
                    confidence=score,
                    mapping_reason="No sufficiently similar section title or role.",
                )
            )
            continue
        mappings.append(
            DraftOutlineMapping(
                segment_id=segment.segment_id,
                section_id=section.section_id,
                confidence=score,
                mapping_reason=(
                    "Matched by heading/title token overlap or inferred draft role."
                ),
            )
        )
    return mappings


def _section_segment_score(section: SectionDraft, segment: DraftSegment) -> float:
    section_label = f"{section.title} {section.description}"
    segment_label = " ".join(
        part
        for part in (
            segment.detected_heading or "",
            segment.inferred_role,
            " ".join(_ROLE_KEYWORDS.get(segment.inferred_role, ())),
            segment.text[:400],
        )
        if part
    )
    return max(
        _token_overlap_score(section_label, segment_label),
        _token_overlap_score(segment_label, section_label) * 0.75,
    )


def _outline_decision(
    *,
    outline: OutlineStructure | None,
    sections: list[SectionDraft],
    mappings: list[DraftOutlineMapping],
    segment_count: int,
    conflicts: list[str],
) -> OutlineAdaptationDecision:
    if outline is None and not sections:
        return OutlineAdaptationDecision(
            action="create_outline_from_draft",
            reasons=["No outline is present; use structured draft segments to seed planning."],
        )
    if conflicts:
        return OutlineAdaptationDecision(
            action="revise_outline",
            reasons=["Draft metadata reports conflicts that should be resolved in the outline."],
        )
    unmapped_count = sum(1 for mapping in mappings if mapping.section_id is None)
    if segment_count and unmapped_count / segment_count >= 0.5:
        return OutlineAdaptationDecision(
            action="revise_outline",
            reasons=["Many draft segments could not be mapped to the current outline."],
        )
    return OutlineAdaptationDecision(
        action="keep_outline",
        reasons=["Most draft segments can be mapped to existing sections."],
    )


def _uncovered_segments(
    segments: list[DraftSegment], mappings: list[DraftOutlineMapping]
) -> list[str]:
    mapped_ids = {
        mapping.segment_id for mapping in mappings if mapping.section_id is not None
    }
    return [segment.segment_id for segment in segments if segment.segment_id not in mapped_ids]


def _detect_conflicts(segments: list[DraftSegment]) -> list[str]:
    conflicts: list[str] = []
    for segment in segments:
        raw_conflict: Any = segment.metadata.get("conflict")
        if isinstance(raw_conflict, str) and raw_conflict.strip():
            conflicts.append(f"{segment.segment_id}: {raw_conflict.strip()}")
        elif raw_conflict is True:
            conflicts.append(f"{segment.segment_id}: draft segment marked as conflicting")
    return conflicts


def _token_overlap_score(left: str, right: str) -> float:
    left_tokens = {token.lower() for token in _WORD_RE.findall(left)}
    right_tokens = {token.lower() for token in _WORD_RE.findall(right)}
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / len(left_tokens)


def intent_instruction(intent: DraftIntent) -> str:
    instructions = {
        "continue": "Preserve existing draft content and continue missing parts.",
        "expand": "Preserve core draft content while adding argument, detail, and transitions.",
        "rewrite": "You may reorganize and rewrite, but preserve the user's core intent.",
        "polish": "Focus on language and structure polish without changing main content.",
        "reference_only": "Use draft ideas only as reference; do not preserve original wording.",
    }
    return instructions[intent]
