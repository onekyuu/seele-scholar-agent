from typing import TYPE_CHECKING, Any

from ..logging import get_logger
from ..state import AgentState
from .models import BudgetState

if TYPE_CHECKING:
    from ..agent_config import BudgetAllocatorFunc

logger = get_logger(__name__)


class BudgetAllocatorNode:
    def __init__(self, allocator: "BudgetAllocatorFunc | None" = None) -> None:
        self.allocator = allocator

    async def allocate(self, state: AgentState) -> dict[str, Any]:
        if self.allocator is None:
            return {}
        budget_state = _budget_state_from_state(state)
        if budget_state is None:
            return {}
        updated = await self.allocator(
            budget_state,
            state.get("sections", []),
            state.get("current_section_index", 0),
        )
        logger.info(
            "budget allocator updated state",
            used=updated.used,
            remaining=updated.remaining,
            section_count=len(updated.sections),
        )
        return {"budget_state": updated}


def _budget_state_from_state(state: AgentState) -> BudgetState | None:
    raw = state.get("budget_state")
    if raw is None:
        return None
    if isinstance(raw, BudgetState):
        return raw
    if isinstance(raw, dict):
        return BudgetState(**raw)
    return None
