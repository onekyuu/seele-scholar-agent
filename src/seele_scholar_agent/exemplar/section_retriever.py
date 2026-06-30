import re
from typing import Any

from ..state import AgentState, SectionDraft
from .models import (
    ExemplarChunk,
    ExemplarContext,
    ExemplarPolicy,
    coerce_exemplar_chunks,
    coerce_exemplar_context,
)

_WORD_RE = re.compile(r"[A-Za-z0-9_\u4e00-\u9fff]+")


class ExemplarSectionRetrieverNode:
    """Select section-level exemplar chunks for the current section."""

    def __init__(self, policy: ExemplarPolicy | None = None) -> None:
        self.policy = policy or ExemplarPolicy()

    def retrieve(self, state: AgentState) -> dict[str, Any]:
        if not self.policy.enabled:
            return {}

        context = coerce_exemplar_context(state.get("exemplar_context"))
        if self.policy.max_examples_per_section <= 0:
            return {"exemplar_context": context.model_copy(update={"section_examples": []})}

        section = _current_section(state)
        if section is None:
            return {"exemplar_context": context}

        chunks = coerce_exemplar_chunks(state.get("exemplar_chunks"))
        if not chunks:
            return {"exemplar_context": context.model_copy(update={"section_examples": []})}

        query = _section_query(section)
        ranked = sorted(
            (
                (_section_match_score(query, chunk), index, chunk)
                for index, chunk in enumerate(chunks)
            ),
            key=lambda item: (item[0], -item[1]),
            reverse=True,
        )
        selected = [
            chunk.model_copy(update={"similarity_score": score})
            for score, _, chunk in ranked
            if score > 0.0
        ][: self.policy.max_examples_per_section]

        updated_context = ExemplarContext(
            outline_patterns=list(context.outline_patterns),
            section_examples=selected,
            style_notes=list(context.style_notes),
            anti_copying_notes=list(context.anti_copying_notes),
        )
        return {"exemplar_context": updated_context}


def _current_section(state: AgentState) -> SectionDraft | None:
    sections = state.get("sections", [])
    current_index = state.get("current_section_index", 0)
    if current_index < 0 or current_index >= len(sections):
        return None
    return sections[current_index]


def _section_query(section: SectionDraft) -> str:
    return " ".join([section.title, section.description]).strip()


def _section_match_score(query: str, chunk: ExemplarChunk) -> float:
    labels = " ".join(
        [
            chunk.section_title,
            chunk.section_role,
            " ".join(chunk.structure_notes),
            " ".join(chunk.style_tags),
        ]
    )
    label_score = _token_overlap_score(query, labels)
    content_score = _token_overlap_score(query, chunk.text[:500]) * 0.5
    return max(label_score, content_score, chunk.similarity_score)


def _token_overlap_score(left: str, right: str) -> float:
    left_tokens = {token.lower() for token in _WORD_RE.findall(left)}
    right_tokens = {token.lower() for token in _WORD_RE.findall(right)}
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / len(left_tokens)
