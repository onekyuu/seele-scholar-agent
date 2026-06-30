from typing import Any

from ..state import AgentState
from .models import (
    ExemplarContext,
    ExemplarPolicy,
    coerce_exemplar_context,
    coerce_exemplar_materials,
)


class ExemplarPlannerContextNode:
    """Build planner-level structure/style context from approved exemplar materials."""

    def __init__(self, policy: ExemplarPolicy | None = None) -> None:
        self.policy = policy or ExemplarPolicy()

    def build(self, state: AgentState) -> dict[str, Any]:
        if not self.policy.enabled:
            return {}

        existing_context = coerce_exemplar_context(state.get("exemplar_context"))
        materials = coerce_exemplar_materials(state.get("exemplar_materials"))
        if not materials:
            return {"exemplar_context": existing_context}

        outline_patterns = list(existing_context.outline_patterns)
        style_notes = list(existing_context.style_notes)
        anti_copying_notes = list(existing_context.anti_copying_notes)
        _append_unique(
            anti_copying_notes,
            "Use exemplars as structure/style references only; do not copy wording.",
        )

        for material in materials:
            if material.usage_role == "negative_example":
                notes = material.style_notes or material.structure_notes
                if not notes:
                    notes = [f"Avoid copying patterns from exemplar {material.exemplar_id}."]
                for note in notes:
                    _append_unique(anti_copying_notes, note)
                continue

            if material.usage_role in {"structure_reference", "section_reference"}:
                for pattern in material.outline_patterns:
                    _append_unique(outline_patterns, pattern)
                for note in material.structure_notes:
                    _append_unique(outline_patterns, note)

            if material.usage_role in {"style_reference", "section_reference"}:
                for note in material.style_notes:
                    _append_unique(style_notes, note)

        return {
            "exemplar_context": ExemplarContext(
                outline_patterns=outline_patterns,
                section_examples=list(existing_context.section_examples),
                style_notes=style_notes,
                anti_copying_notes=anti_copying_notes,
            )
        }


def _append_unique(values: list[str], value: str) -> None:
    normalized = value.strip()
    if normalized and normalized not in values:
        values.append(normalized)
