from collections.abc import Mapping
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel


class GenerationMode(StrEnum):
    FULL_DOCUMENT = "full_document"
    SINGLE_SECTION = "single_section"


class SectionExecutionPolicy(BaseModel):
    generation_mode: GenerationMode = GenerationMode.FULL_DOCUMENT
    auto_advance_sections: bool = True
    require_section_approval: bool = False
    stop_after_section_review: bool = False


class SectionExecutionStrategy:
    """Decides section progress without relying on append-only state fields."""

    def __init__(self, policy: SectionExecutionPolicy | None = None) -> None:
        self.policy = policy or SectionExecutionPolicy()

    def approved_section_delta(
        self,
        state: Mapping[str, Any],
        *,
        section_status: Literal["approved", "accepted_with_issues"] = "approved",
    ) -> dict[str, Any]:
        sections = list(state["sections"])
        index = int(state["current_section_index"])
        section = sections[index]
        sections[index] = section.model_copy(update={"status": section_status})

        delta: dict[str, Any] = {
            "sections": sections,
            "sections_completed": completed_section_titles(sections),
        }

        if self.policy.require_section_approval:
            delta["status"] = "waiting_human"
            return delta

        if (
            self.policy.generation_mode == GenerationMode.SINGLE_SECTION
            or self.policy.stop_after_section_review
            or not self.policy.auto_advance_sections
        ):
            delta["status"] = "section_done"
            return delta

        if index + 1 >= len(sections):
            delta["status"] = "completed"
            return delta

        delta["current_section_index"] = index + 1
        delta["status"] = "writing"
        return delta

    def skip_completed_section_delta(self, state: Mapping[str, Any]) -> dict[str, Any]:
        sections = list(state["sections"])
        index = int(state["current_section_index"])
        delta: dict[str, Any] = {"sections_completed": completed_section_titles(sections)}

        if self.policy.generation_mode == GenerationMode.SINGLE_SECTION:
            delta["status"] = "section_done"
            return delta

        if index + 1 >= len(sections):
            delta["status"] = "completed"
            return delta

        delta["current_section_index"] = index + 1
        delta["status"] = "writing"
        return delta

    def route_after_review(
        self,
        state: Mapping[str, Any],
        *,
        has_blocking_quality_issues: bool,
        completed_route: str,
    ) -> str:
        status = state.get("status")
        if status in {"waiting_human", "section_done", "failed"}:
            return "end"
        if status == "writing":
            return "writer"
        if status == "completed":
            return completed_route
        if has_blocking_quality_issues:
            return "end"
        return "writer"


def completed_section_titles(sections: list[Any]) -> list[str]:
    return [
        section.title
        for section in sections
        if getattr(section, "status", None) in {"approved", "accepted_with_issues"}
    ]
