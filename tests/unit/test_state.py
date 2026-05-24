import pytest
from pydantic import ValidationError
from seele_scholar_agent.state import (
    ClaimEvidenceBinding,
    EvidencePacket,
    MaterialRegistry,
    MaterialRegistryEntry,
    OutlineStructure,
    PaperMetadata,
    QualityIssue,
    ReferenceEntry,
    ReviewIssue,
    ReviewResult,
    SectionDraft,
    SectionOutline,
)


def test_paper_metadata_defaults():
    paper = PaperMetadata(paper_id="p1", title="T", authors=["A"], abstract="abs")
    assert paper.relevance_score == 0.0
    assert paper.source == "openalex"
    assert paper.url is None
    assert paper.pdf_url is None
    assert paper.doi is None
    assert paper.year is None
    assert paper.venue is None
    assert paper.query_overlap_score == 0.0
    assert paper.embedding_similarity is None
    assert paper.user_priority == 0.0


def test_paper_metadata_invalid_source():
    with pytest.raises(ValidationError):
        PaperMetadata(paper_id="p1", title="T", authors=[], abstract="", source="github")  # type: ignore[arg-type]


def test_material_registry_entry_defaults():
    entry = MaterialRegistryEntry(paper_id="p1")
    registry = MaterialRegistry(entries=[entry])

    assert registry.entries[0].citation_role == "citable"
    assert registry.entries[0].confidence == "normal"
    assert registry.entries[0].required is False


def test_section_draft_defaults():
    s = SectionDraft(section_id="s0", title="Intro", order_index=1)
    assert s.status == "pending"
    assert s.revision_count == 0
    assert s.review_comments == []
    assert s.content == ""


def test_section_draft_model_copy_immutable():
    s = SectionDraft(section_id="s0", title="Intro", order_index=1)
    s2 = s.model_copy(update={"status": "approved"})
    assert s.status == "pending"
    assert s2.status == "approved"


def test_review_result_score_out_of_range_high():
    with pytest.raises(ValidationError):
        ReviewResult(approved=False, score=11, summary="test")


def test_review_result_score_out_of_range_zero():
    with pytest.raises(ValidationError):
        ReviewResult(approved=False, score=0, summary="test")


def test_review_result_score_boundary_valid():
    r1 = ReviewResult(approved=False, score=1, summary="low")
    r2 = ReviewResult(approved=True, score=10, summary="high")
    assert r1.score == 1
    assert r2.score == 10


def test_review_issue_invalid_type():
    with pytest.raises(ValidationError):
        ReviewIssue(type="unknown_type", description="desc", suggestion="sug")  # type: ignore[arg-type]


def test_outline_structure_defaults():
    outline = OutlineStructure(title="T", abstract="A")
    assert outline.sections == []
    assert outline.keywords == []
    assert outline.paper_type == "auto"
    assert outline.structure_pattern == "auto"
    assert outline.evidence_map == []


def test_agent_state_annotated_add(base_state, sample_papers):
    from seele_scholar_agent.state import AgentState

    p1 = sample_papers[0]
    p2 = sample_papers[1]

    state1 = AgentState(**{**base_state, "papers": [p1]})
    state2 = AgentState(**{**base_state, "papers": [p2]})

    combined_papers = state1["papers"] + state2["papers"]
    assert len(combined_papers) == 2
    assert combined_papers[0].paper_id == p1.paper_id
    assert combined_papers[1].paper_id == p2.paper_id


def test_section_outline_key_points_default():
    s = SectionOutline(title="Intro", order=1)
    assert s.key_points == []
    assert s.description == ""
    assert s.purpose == ""
    assert s.content_summary == ""
    assert s.transition_to_next == ""
    assert s.target_claims == []
    assert s.key_sources == []
    assert s.evidence_gaps == []


def test_quality_issue_defaults():
    issue = QualityIssue(code="TEST", message="Test issue")
    assert issue.severity == "error"
    assert issue.blocking is False
    assert issue.details == {}


def test_reference_entry_verification_defaults():
    reference = ReferenceEntry(
        number=1,
        paper_id="p1",
        title="T",
        authors=["A"],
        formatted="[1] A. T",
    )
    assert reference.metadata_verified is False
    assert reference.verification_source == "none"


def test_evidence_packet_defaults():
    packet = EvidencePacket(chunk_id="c1", quote="Quoted text")
    assert packet.title == ""
    assert packet.authors == []
    assert packet.relevance_score == 0.0
    assert packet.quote == "Quoted text"


def test_claim_evidence_binding_defaults():
    binding = ClaimEvidenceBinding(
        section_id="s1",
        claim_text="Claim [1].",
        citation_number=1,
    )
    assert binding.chunk_id is None
    assert binding.support_score == 0.0
    assert binding.verdict == "unverified"
