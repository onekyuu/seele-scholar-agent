import pytest
from seele_scholar_agent.nodes.integrity_gate import IntegrityGateNode
from seele_scholar_agent.state import QualityIssue


@pytest.mark.asyncio
async def test_integrity_gate_passes_without_blocking_issues(base_state):
    node = IntegrityGateNode()
    result = await node.check(base_state)

    assert result["status"] == "completed"
    assert result["error_message"] is None


@pytest.mark.asyncio
async def test_integrity_gate_blocks_completion_for_blocking_issue(base_state):
    issue = QualityIssue(
        code="NO_INLINE_CITATIONS",
        message="No inline citations were found.",
        severity="blocking",
        blocking=True,
    )
    node = IntegrityGateNode()
    result = await node.check({**base_state, "quality_issues": [issue]})

    assert result["status"] == "waiting_human"
    assert result["error_message"] == "No inline citations were found."
