from seele_scholar_agent.nodes.paragraph_quality_audit import ParagraphQualityAudit
from seele_scholar_agent.state import SectionOutline


def test_paragraph_quality_audit_flags_duplicate_paragraphs():
    audit = ParagraphQualityAudit()
    paragraph = "This paragraph provides concrete evidence [1], and therefore explains the point."

    findings = audit.audit(section_title="Discussion", content=f"{paragraph}\n\n{paragraph}")

    assert findings[0].code == "PARAGRAPH_DUPLICATE"


def test_paragraph_quality_audit_flags_generic_template_language():
    audit = ParagraphQualityAudit()

    findings = audit.audit(
        section_title="Introduction",
        content="In today's world, this topic plays an important role in many fields.",
    )

    assert findings[0].code == "PARAGRAPH_GENERIC_TEMPLATE"


def test_paragraph_quality_audit_flags_uncovered_target_claim():
    audit = ParagraphQualityAudit()
    outline = SectionOutline(
        title="Discussion",
        order=1,
        target_claims=["retrieval augmented generation improves citation grounding"],
    )

    findings = audit.audit(
        section_title="Discussion",
        content="The section explains interface design and deployment constraints.",
        section_outline=outline,
    )

    assert findings[0].code == "SECTION_TARGET_CLAIM_UNCOVERED"


def test_paragraph_quality_audit_flags_missing_evidence_and_analysis():
    audit = ParagraphQualityAudit()

    findings = audit.audit(
        section_title="Discussion",
        content=(
            "The proposed approach offers a broad conceptual framing for the system. "
            "The paragraph describes the architecture, the workflow, and the expected "
            "contribution across several parts of the paper without grounding details."
        ),
    )

    assert findings[0].code == "PARAGRAPH_STRUCTURE_INCOMPLETE"
    assert "evidence" in findings[0].description
    assert "analysis" in findings[0].description


def test_paragraph_quality_audit_can_skip_structure_check():
    audit = ParagraphQualityAudit()

    findings = audit.audit(
        section_title="Discussion",
        content=(
            "The proposed approach offers a broad conceptual framing for the system. "
            "The paragraph describes the architecture, the workflow, and the expected "
            "contribution across several parts of the paper without grounding details."
        ),
        include_structure_check=False,
    )

    assert findings == []


def test_paragraph_quality_audit_accepts_complete_paragraph():
    audit = ParagraphQualityAudit()

    findings = audit.audit(
        section_title="Discussion",
        content=(
            "Retrieval quality shapes downstream citation reliability. Evidence from "
            "the audit shows 42% higher chunk support after query expansion [1]. "
            "Therefore, retrieval improvements should be treated as a prerequisite "
            "for reliable generated academic claims."
        ),
    )

    assert findings == []
