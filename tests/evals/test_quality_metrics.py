import pytest
from seele_scholar_agent.evals import evaluate_quality
from seele_scholar_agent.state import (
    ClaimEvidenceBinding,
    OutlineStructure,
    ReferenceEntry,
    SectionDraft,
    SectionOutline,
)


def _section(title: str, content: str, order: int = 1) -> SectionDraft:
    return SectionDraft(
        section_id=f"s{order}",
        title=title,
        content=content,
        order_index=order,
        status="approved",
    )


def test_evaluate_quality_computes_core_metrics(base_state):
    sections = [
        _section(
            "Introduction",
            (
                "Transformers use attention to improve sequence modeling [1].\n\n"
                "Repeated paragraph.\n\n"
                "Repeated paragraph."
            ),
        ),
        _section("Methods", "This cites a missing reference [2].", order=2),
    ]
    outline = OutlineStructure(
        title="T",
        abstract="A",
        sections=[
            SectionOutline(
                title="Introduction",
                order=1,
                target_claims=[
                    "Transformers use attention",
                    "Graph neural networks improve molecules",
                ],
            )
        ],
    )
    reference = ReferenceEntry(
        number=1,
        paper_id="p1",
        title="Attention Is All You Need",
        authors=["Vaswani"],
        formatted="[1] Vaswani. Attention Is All You Need",
    )
    bindings = [
        ClaimEvidenceBinding(
            section_id="s1",
            claim_text="Transformers use attention.",
            citation_number=1,
            chunk_id="chunk-1",
            support_score=0.8,
            verdict="supported",
        ),
        ClaimEvidenceBinding(
            section_id="s2",
            claim_text="Missing support.",
            citation_number=2,
            support_score=0.0,
            verdict="unverified",
        ),
    ]

    metrics = evaluate_quality(
        {
            **base_state,
            "sections": sections,
            "outline": outline,
            "references": [reference],
            "claim_evidence_bindings": bindings,
            "review_history": [{"approved": True}, {"approved": False}],
        },
        user_edited_text="Transformers use attention to improve sequence modeling [1].",
    )

    assert metrics.citation_validity_rate == pytest.approx(0.5)
    assert metrics.chunk_support_rate == pytest.approx(0.5)
    assert metrics.reviewer_pass_rate == pytest.approx(0.5)
    assert metrics.duplicate_paragraph_ratio == pytest.approx(1 / 4)
    assert metrics.section_target_coverage == pytest.approx(0.5)
    assert metrics.user_edit_ratio is not None
    assert metrics.user_edit_ratio > 0.0


def test_evaluate_quality_empty_state_uses_zero_baselines(base_state):
    metrics = evaluate_quality(base_state)

    assert metrics.citation_validity_rate == 0.0
    assert metrics.chunk_support_rate == 0.0
    assert metrics.reviewer_pass_rate == 0.0
    assert metrics.duplicate_paragraph_ratio == 0.0
    assert metrics.section_target_coverage == 0.0
    assert metrics.user_edit_ratio is None
