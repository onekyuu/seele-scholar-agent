from collections.abc import AsyncIterator
from typing import Any

from ..logging import get_logger
from ..state import AgentState, QualityIssue
from . import NodeStreamEvent

logger = get_logger(__name__)


def _is_blocking_issue(issue: QualityIssue) -> bool:
    return issue.blocking or issue.severity == "blocking"


class IntegrityGateNode:
    """Final quality gate that prevents completed status when blocking issues exist."""

    async def check(self, state: AgentState) -> dict[str, Any]:
        quality_issues = list(state.get("quality_issues") or [])
        blocking_issues = [issue for issue in quality_issues if _is_blocking_issue(issue)]

        if blocking_issues:
            logger.warning(
                "integrity gate blocked completion",
                blocking_issue_count=len(blocking_issues),
                issue_codes=[issue.code for issue in blocking_issues],
            )
            return {
                "status": "waiting_human",
                "error_message": self._build_error_message(blocking_issues),
            }

        logger.info("integrity gate passed")
        return {"status": "completed", "error_message": None}

    async def astream(self, state: AgentState) -> AsyncIterator[NodeStreamEvent]:
        yield NodeStreamEvent(type="progress", progress="checking_integrity")
        result = await self.check(state)
        yield NodeStreamEvent(type="result", result=result)

    def _build_error_message(self, blocking_issues: list[QualityIssue]) -> str:
        first = blocking_issues[0]
        if len(blocking_issues) == 1:
            return first.message
        return f"{first.message} (+{len(blocking_issues) - 1} more blocking issue(s))"
