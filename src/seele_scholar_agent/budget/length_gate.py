import re
from typing import Any

from ..logging import get_logger
from ..state import AgentState, QualityIssue
from .models import BudgetPolicy, BudgetState, BudgetUnit, SectionBudget

logger = get_logger(__name__)

_WORD_RE = re.compile(r"[A-Za-z0-9_]+(?:[-'][A-Za-z0-9_]+)?")


def count_length(text: str, unit: BudgetUnit) -> int:
    if unit == "chars":
        return len([char for char in text if not char.isspace()])
    if unit == "tokens":
        return len([part for part in text.split() if part])
    return len(_WORD_RE.findall(text))


class LengthGateNode:
    """Deterministic section length check before expensive review."""

    def __init__(self, budget_policy: BudgetPolicy | None = None) -> None:
        self.budget_policy = budget_policy or BudgetPolicy()

    async def check(self, state: AgentState) -> dict[str, Any]:
        budget_state = _budget_state_from_state(state)
        sections = state.get("sections", [])
        index = state.get("current_section_index", 0)
        if (
            not self.budget_policy.enabled
            or budget_state is None
            or index >= len(sections)
        ):
            return {"budget_diagnostics": {"passed": True, "reason": "no_budget"}}

        section = sections[index]
        section_budget = _section_budget_for(section.section_id, budget_state)
        if section_budget is None:
            return {"budget_diagnostics": {"passed": True, "reason": "no_section_budget"}}

        unit = section_budget.unit or budget_state.unit
        actual = count_length(section.content, unit)
        diagnostics = _diagnostics(section_budget, actual)
        updated_budget_state = _with_actual(budget_state, section.section_id, actual)

        quality_issues = _without_current_budget_issue(
            list(state.get("quality_issues") or []), section.section_id
        )
        if diagnostics["over_budget"]:
            quality_issues.append(
                QualityIssue(
                    code="SECTION_OVER_BUDGET",
                    message=(
                        f"Section '{section.title}' is over budget "
                        f"({actual} {unit})."
                    ),
                    severity="warning",
                    location=section.section_id,
                    blocking=False,
                    details=diagnostics,
                )
            )

        logger.info(
            "length gate checked section",
            section=section.title,
            actual=actual,
            unit=unit,
            over_budget=diagnostics["over_budget"],
        )
        result: dict[str, Any] = {
            "budget_state": updated_budget_state,
            "budget_diagnostics": diagnostics,
            "quality_issues": quality_issues,
        }
        return result


def _budget_state_from_state(state: AgentState) -> BudgetState | None:
    raw = state.get("budget_state")
    if raw is None:
        return None
    if isinstance(raw, BudgetState):
        return raw
    if isinstance(raw, dict):
        return BudgetState(**raw)
    return None


def _section_budget_for(section_id: str, budget_state: BudgetState) -> SectionBudget | None:
    raw = budget_state.sections.get(section_id)
    if raw is None:
        return None
    if isinstance(raw, SectionBudget):
        return raw
    return SectionBudget(**raw)


def _diagnostics(section_budget: SectionBudget, actual: int) -> dict[str, Any]:
    target = section_budget.target
    hard_limit = section_budget.hard_limit
    tolerated = int(target * section_budget.tolerance_ratio) if target is not None else None
    limit = hard_limit if hard_limit is not None else tolerated
    over_budget = limit is not None and actual > limit
    return {
        "passed": not over_budget,
        "needs_revision": over_budget,
        "over_budget": over_budget,
        "section_id": section_budget.section_id,
        "actual": actual,
        "target": target,
        "hard_limit": hard_limit,
        "tolerance_ratio": section_budget.tolerance_ratio,
        "unit": section_budget.unit,
        "limit": limit,
    }


def _with_actual(
    budget_state: BudgetState, section_id: str, actual: int
) -> BudgetState:
    actuals = dict(budget_state.section_actuals)
    actuals[section_id] = actual
    used = sum(actuals.values())
    remaining = (
        max(budget_state.total_target - used, 0)
        if budget_state.total_target is not None
        else budget_state.remaining
    )
    return budget_state.model_copy(
        update={"section_actuals": actuals, "used": used, "remaining": remaining}
    )


def _without_current_budget_issue(
    issues: list[QualityIssue], section_id: str
) -> list[QualityIssue]:
    return [
        issue
        for issue in issues
        if not (issue.code == "SECTION_OVER_BUDGET" and issue.location == section_id)
    ]
