import pytest
from seele_scholar_agent.nodes.integrity_gate import IntegrityGateNode
from seele_scholar_agent.state import (
    ClaimEvidenceBinding,
    EvidencePacket,
    QualityIssue,
    ReferenceEntry,
    SectionDraft,
)


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


@pytest.mark.asyncio
async def test_integrity_gate_strict_mode_passes_with_verified_evidence(base_state):
    section = SectionDraft(
        section_id="s1",
        title="Introduction",
        content="Transformer attention improves sequence modeling [1].",
        order_index=1,
        status="approved",
    )
    reference = ReferenceEntry(
        number=1,
        paper_id="p1",
        title="Attention Is All You Need",
        authors=["Vaswani"],
        doi="10.48550/arXiv.1706.03762",
        metadata_verified=True,
        verification_source="crossref",
        formatted="[1] Vaswani. Attention Is All You Need",
    )
    packet = EvidencePacket(
        chunk_id="chunk-1",
        title="Attention Is All You Need",
        source_paper_id="p1",
        relevance_score=0.85,
        quote="Attention improves sequence modeling.",
    )
    binding = ClaimEvidenceBinding(
        section_id="s1",
        claim_text="Transformer attention improves sequence modeling.",
        citation_number=1,
        chunk_id="chunk-1",
        source_paper_id="p1",
        support_score=0.9,
        verdict="supported",
    )

    node = IntegrityGateNode()
    result = await node.check(
        {
            **base_state,
            "strict_academic_mode": True,
            "sections": [section],
            "references": [reference],
            "evidence_packets": [packet],
            "claim_evidence_bindings": [binding],
        }
    )

    assert result["status"] == "completed"
    assert result["quality_issues"] == []


@pytest.mark.asyncio
async def test_integrity_gate_strict_mode_blocks_unverified_reference_and_missing_binding(
    base_state,
):
    section = SectionDraft(
        section_id="s1",
        title="Introduction",
        content="This claim cites an unverified source [1].",
        order_index=1,
        status="approved",
    )
    reference = ReferenceEntry(
        number=1,
        paper_id="p1",
        title="Unverified Paper",
        authors=["Author"],
        formatted="[1] Author. Unverified Paper",
    )

    node = IntegrityGateNode()
    result = await node.check(
        {
            **base_state,
            "strict_academic_mode": True,
            "sections": [section],
            "references": [reference],
            "claim_evidence_bindings": [],
        }
    )

    codes = {issue.code for issue in result["quality_issues"]}
    assert result["status"] == "waiting_human"
    assert "STRICT_UNVERIFIED_REFERENCE" in codes
    assert "STRICT_MISSING_CHUNK_BINDING" in codes


@pytest.mark.asyncio
async def test_integrity_gate_strict_mode_blocks_unsupported_claim(base_state):
    section = SectionDraft(
        section_id="s1",
        title="Introduction",
        content="A weakly supported claim appears here [1].",
        order_index=1,
        status="approved",
    )
    reference = ReferenceEntry(
        number=1,
        paper_id="p1",
        title="Verified Paper",
        authors=["Author"],
        doi="10.1000/example",
        metadata_verified=True,
        verification_source="openalex",
        formatted="[1] Author. Verified Paper",
    )
    packet = EvidencePacket(
        chunk_id="chunk-1",
        title="Verified Paper",
        source_paper_id="p1",
        relevance_score=0.9,
        quote="Different evidence.",
    )
    binding = ClaimEvidenceBinding(
        section_id="s1",
        claim_text="A weakly supported claim appears here.",
        citation_number=1,
        chunk_id="chunk-1",
        support_score=0.1,
        verdict="weak",
    )

    node = IntegrityGateNode()
    result = await node.check(
        {
            **base_state,
            "strict_academic_mode": True,
            "sections": [section],
            "references": [reference],
            "evidence_packets": [packet],
            "claim_evidence_bindings": [binding],
        }
    )

    codes = {issue.code for issue in result["quality_issues"]}
    assert result["status"] == "waiting_human"
    assert "STRICT_UNSUPPORTED_CLAIM" in codes


@pytest.mark.asyncio
async def test_integrity_gate_skips_strict_academic_checks_for_proposal(base_state):
    section = SectionDraft(
        section_id="s1",
        title="研究背景",
        content="先行研究を参照する [1]。",
        order_index=1,
        status="approved",
    )
    reference = ReferenceEntry(
        number=1,
        paper_id="p1",
        title="Unverified Paper",
        authors=["Author"],
        formatted="[1] Author. Unverified Paper",
    )

    node = IntegrityGateNode()
    result = await node.check(
        {
            **base_state,
            "document_type": "research_proposal",
            "strict_academic_mode": True,
            "sections": [section],
            "references": [reference],
            "claim_evidence_bindings": [],
        }
    )

    assert result["status"] == "completed"
    assert result["quality_issues"] == []
