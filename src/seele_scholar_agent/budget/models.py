from typing import Literal

from pydantic import BaseModel, Field

BudgetUnit = Literal["words", "chars", "tokens"]


class SectionBudget(BaseModel):
    section_id: str
    target: int | None = None
    hard_limit: int | None = None
    unit: BudgetUnit = "words"
    tolerance_ratio: float = 1.15


class BudgetState(BaseModel):
    total_target: int | None = None
    used: int = 0
    remaining: int | None = None
    unit: BudgetUnit = "words"
    sections: dict[str, SectionBudget] = Field(default_factory=dict)
    section_actuals: dict[str, int] = Field(default_factory=dict)


class BudgetPolicy(BaseModel):
    enabled: bool = True
    revise_when_over_budget: bool = True
    revise_when_under_budget: bool = False
    max_budget_revision_rounds: int = 1
