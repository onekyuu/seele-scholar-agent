from typing import Literal

from pydantic import BaseModel


class WritingPolicy(BaseModel):
    max_revisions: int = 3
    require_inline_citations: bool = True
    strict_citation_alignment: bool = True
    strict_claim_evidence_binding: bool = False
    allow_uncited_plan_statements: bool = False
    enable_budget_gate: bool = True
    on_max_revisions: Literal["block", "accept_best_with_report"] = (
        "accept_best_with_report"
    )
