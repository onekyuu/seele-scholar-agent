import pytest
from seele_scholar_agent.nodes.outline_quality_gate import OutlineQualityGateNode
from seele_scholar_agent.state import (
    MaterialRegistry,
    MaterialRegistryEntry,
    OutlineStructure,
    PaperMetadata,
    SectionEvidencePlan,
    SectionOutline,
)


def _section(
    title: str = "Introduction",
    order: int = 1,
    *,
    purpose: str = "Frame the paper.",
    transition_to_next: str = "",
    target_claims: list[str] | None = None,
    key_sources: list[str] | None = None,
    citation_plan: list[str] | None = None,
    evidence_gaps: list[str] | None = None,
) -> SectionOutline:
    return SectionOutline(
        title=title,
        order=order,
        purpose=purpose,
        content_summary="Summary.",
        transition_to_next=transition_to_next,
        target_claims=["Claim A"] if target_claims is None else target_claims,
        key_sources=["[1] Source A"] if key_sources is None else key_sources,
        citation_plan=["Use [1] for Claim A"] if citation_plan is None else citation_plan,
        evidence_gaps=[] if evidence_gaps is None else evidence_gaps,
    )


def _outline(sections: list[SectionOutline]) -> OutlineStructure:
    return OutlineStructure(
        title="T",
        abstract="A",
        sections=sections,
        paper_type="literature_review",
        structure_pattern="thematic_review",
        evidence_map=[
            SectionEvidencePlan(
                section_title=section.title,
                target_claims=section.target_claims,
                key_sources=section.key_sources,
                citation_plan=section.citation_plan,
                evidence_gaps=section.evidence_gaps,
            )
            for section in sections
        ],
    )


@pytest.mark.asyncio
async def test_outline_quality_gate_passes_complete_outline(base_state):
    sections = [
        _section("Background", order=1, transition_to_next="Next, compare approaches."),
        _section("Synthesis", order=2),
    ]

    result = await OutlineQualityGateNode().check({**base_state, "outline": _outline(sections)})

    assert result["quality_issues"] == []


@pytest.mark.asyncio
async def test_outline_quality_gate_blocks_missing_purpose_transition_and_claims(base_state):
    section = _section(
        "Background",
        purpose="",
        transition_to_next="",
        target_claims=[],
        key_sources=[],
        citation_plan=[],
    )
    outline = _outline([section, _section("Synthesis", order=2)])

    result = await OutlineQualityGateNode().check({**base_state, "outline": outline})

    codes = {issue.code for issue in result["quality_issues"]}
    assert result["status"] == "waiting_human"
    assert "OUTLINE_MISSING_PURPOSE" in codes
    assert "OUTLINE_MISSING_TRANSITION" in codes
    assert "OUTLINE_MISSING_TARGET_CLAIMS" in codes


@pytest.mark.asyncio
async def test_outline_quality_gate_warns_for_evidence_gaps_without_blocking(base_state):
    section = _section(evidence_gaps=["Need newer source."])

    result = await OutlineQualityGateNode().check({**base_state, "outline": _outline([section])})

    assert result["quality_issues"][0].code == "OUTLINE_EVIDENCE_GAPS"
    assert result["quality_issues"][0].blocking is False


@pytest.mark.asyncio
async def test_outline_quality_gate_blocks_missing_evidence_map_claim_coverage(base_state):
    section = _section(target_claims=["Claim A", "Claim B"])
    outline = _outline([section])
    outline = outline.model_copy(
        update={
            "evidence_map": [
                SectionEvidencePlan(
                    section_title=section.title,
                    target_claims=["Claim A"],
                    key_sources=["[1] Source A"],
                    citation_plan=["Use [1] for Claim A"],
                )
            ]
        }
    )

    result = await OutlineQualityGateNode().check({**base_state, "outline": outline})

    assert result["status"] == "waiting_human"
    assert result["quality_issues"][0].code == "OUTLINE_EVIDENCE_MAP_MISSING_CLAIMS"


@pytest.mark.asyncio
async def test_outline_quality_gate_blocks_imrad_for_non_empirical_type(base_state):
    outline = _outline(
        [
            _section("Methods", order=1, transition_to_next="Next, results."),
            _section("Results", order=2),
        ]
    ).model_copy(update={"paper_type": "literature_review", "structure_pattern": "IMRaD"})

    result = await OutlineQualityGateNode().check({**base_state, "outline": outline})

    assert result["status"] == "waiting_human"
    assert result["quality_issues"][0].code == "OUTLINE_EXPERIMENTAL_TEMPLATE_MISMATCH"


@pytest.mark.asyncio
async def test_outline_quality_gate_blocks_required_material_not_planned(base_state):
    papers = [
        PaperMetadata(
            paper_id="p1",
            title="Required Paper",
            authors=["Author"],
            abstract="Abstract.",
            relevance_score=0.8,
        )
    ]
    registry = MaterialRegistry(entries=[MaterialRegistryEntry(paper_id="p1", required=True)])
    section = _section(key_sources=[], citation_plan=[])

    result = await OutlineQualityGateNode().check(
        {
            **base_state,
            "outline": _outline([section]),
            "papers": papers,
            "material_registry": registry,
        }
    )

    codes = {issue.code for issue in result["quality_issues"]}
    assert result["status"] == "waiting_human"
    assert "REQUIRED_MATERIAL_NOT_PLANNED" in codes


@pytest.mark.asyncio
async def test_outline_quality_gate_blocks_non_citable_material_planned(base_state):
    papers = [
        PaperMetadata(
            paper_id="p1",
            title="Background Paper",
            authors=["Author"],
            abstract="Abstract.",
            relevance_score=0.8,
        )
    ]
    registry = MaterialRegistry(
        entries=[MaterialRegistryEntry(paper_id="p1", citation_role="background")]
    )
    section = _section(citation_plan=["Use [1] for a claim."])

    result = await OutlineQualityGateNode().check(
        {
            **base_state,
            "outline": _outline([section]),
            "papers": papers,
            "material_registry": registry,
        }
    )

    assert result["status"] == "waiting_human"
    assert result["quality_issues"][0].code == "OUTLINE_CITES_NON_CITABLE_MATERIAL"


@pytest.mark.asyncio
async def test_outline_quality_gate_skips_required_material_relevance_by_default(base_state):
    papers = [
        PaperMetadata(
            paper_id="p1",
            title="Required Paper",
            authors=["Author"],
            abstract="Abstract.",
            relevance_score=0.05,
            query_overlap_score=0.0,
        )
    ]
    registry = MaterialRegistry(entries=[MaterialRegistryEntry(paper_id="p1", required=True)])
    section = _section(citation_plan=["Use [1] for a claim."])

    result = await OutlineQualityGateNode().check(
        {
            **base_state,
            "outline": _outline([section]),
            "papers": papers,
            "material_registry": registry,
        }
    )

    assert result["quality_issues"] == []


@pytest.mark.asyncio
async def test_outline_quality_gate_warns_required_material_low_relevance_when_enabled(
    base_state,
):
    papers = [
        PaperMetadata(
            paper_id="p1",
            title="Required Paper",
            authors=["Author"],
            abstract="Abstract.",
            relevance_score=0.05,
            query_overlap_score=0.0,
        )
    ]
    registry = MaterialRegistry(entries=[MaterialRegistryEntry(paper_id="p1", required=True)])
    section = _section(citation_plan=["Use [1] for a claim."])

    result = await OutlineQualityGateNode().check(
        {
            **base_state,
            "outline": _outline([section]),
            "papers": papers,
            "material_registry": registry,
            "check_required_material_relevance": True,
        }
    )

    assert result["quality_issues"][0].code == "REQUIRED_MATERIAL_LOW_RELEVANCE"
    assert result["quality_issues"][0].blocking is False
