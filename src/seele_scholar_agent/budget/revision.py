from typing import Any

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from ..agent_config import PromptsConfig
from ..logging import get_logger
from ..state import AgentState
from .length_gate import count_length
from .models import BudgetPolicy, BudgetState, SectionBudget

logger = get_logger(__name__)

_SYSTEM_PROMPT = (
    "You revise academic writing to satisfy a section length budget. "
    "Return only the complete replacement body. Do not add citations that were not "
    "already present in the original content."
)

_USER_PROMPT = """Topic: {topic}
Section: {section_title}
Budget: target={target}, hard_limit={hard_limit}, unit={unit}

Current section body:
{content}

Revise this section so it stays within the budget while preserving the core argument,
necessary evidence, existing citation numbers, and section role. Do not output a heading."""


class BudgetRevisionNode:
    def __init__(
        self,
        llm: ChatOpenAI,
        prompts: PromptsConfig,
        budget_policy: BudgetPolicy | None = None,
    ) -> None:
        self.llm = llm
        self.prompts = prompts
        self.budget_policy = budget_policy or BudgetPolicy()
        self.prompt = ChatPromptTemplate.from_messages(
            [("system", _SYSTEM_PROMPT), ("user", _USER_PROMPT)]
        )
        self.chain = self.prompt | self.llm

    async def revise(self, state: AgentState) -> dict[str, Any]:
        sections = state["sections"]
        index = state["current_section_index"]
        section = sections[index]
        budget_state = _budget_state_from_state(state)
        section_budget = (
            _section_budget_for(section.section_id, budget_state)
            if budget_state is not None
            else None
        )
        if section_budget is None:
            return {"budget_diagnostics": {"passed": True, "reason": "no_section_budget"}}

        result = await self.chain.ainvoke(
            {
                "topic": state["topic"],
                "section_title": section.title,
                "target": section_budget.target or "none",
                "hard_limit": section_budget.hard_limit or "none",
                "unit": section_budget.unit,
                "content": section.content,
            }
        )
        content = result.content if hasattr(result, "content") else str(result)
        if isinstance(content, list):
            content = "\n".join(str(item) for item in content)
        revised_content = str(content).strip()

        updated_sections = list(sections)
        updated_sections[index] = section.model_copy(update={"content": revised_content})
        rounds = dict(state.get("budget_revision_rounds", {}) or {})
        rounds[section.section_id] = int(rounds.get(section.section_id, 0)) + 1

        actual = count_length(revised_content, section_budget.unit)
        updated_budget_state = _with_actual(budget_state, section.section_id, actual)
        logger.info(
            "budget revision completed",
            section=section.title,
            revision_round=rounds[section.section_id],
            actual=actual,
            unit=section_budget.unit,
        )
        return {
            "sections": updated_sections,
            "budget_state": updated_budget_state,
            "budget_revision_rounds": rounds,
            "budget_diagnostics": {
                "passed": True,
                "reason": "budget_revision_completed",
                "section_id": section.section_id,
                "actual": actual,
                "unit": section_budget.unit,
            },
        }


def _budget_state_from_state(state: AgentState) -> BudgetState | None:
    raw = state.get("budget_state")
    if raw is None:
        return None
    if isinstance(raw, BudgetState):
        return raw
    if isinstance(raw, dict):
        return BudgetState(**raw)
    return None


def _section_budget_for(
    section_id: str, budget_state: BudgetState | None
) -> SectionBudget | None:
    if budget_state is None:
        return None
    raw = budget_state.sections.get(section_id)
    if raw is None:
        return None
    if isinstance(raw, SectionBudget):
        return raw
    return SectionBudget(**raw)


def _with_actual(
    budget_state: BudgetState | None, section_id: str, actual: int
) -> BudgetState | None:
    if budget_state is None:
        return None
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
