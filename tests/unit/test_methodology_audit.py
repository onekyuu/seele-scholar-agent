from seele_scholar_agent.nodes.methodology_audit import MethodologyAudit
from seele_scholar_agent.profiles.research_proposal import ResearchProposalProfile


def test_methodology_audit_flags_missing_sample_size():
    audit = MethodologyAudit()

    findings = audit.audit(
        section_title="Methods",
        content="We recruited participants through an online questionnaire.",
    )

    assert findings[0].code == "METHODOLOGY_SAMPLE_SIZE_MISSING"


def test_methodology_audit_flags_baseline_metric_and_uncertainty_gaps():
    audit = MethodologyAudit()

    findings = audit.audit(
        section_title="Results",
        content="Our model improves accuracy and significantly outperforms prior systems.",
        paper_type="empirical",
    )
    codes = {finding.code for finding in findings}

    assert "METHODOLOGY_BASELINE_FAIRNESS_MISSING" in codes
    assert "METHODOLOGY_METRIC_DEFINITION_MISSING" in codes
    assert "METHODOLOGY_SIGNIFICANCE_UNCERTAINTY_MISSING" in codes


def test_methodology_audit_flags_correlation_causation_mix():
    audit = MethodologyAudit()

    findings = audit.audit(
        section_title="Analysis",
        content="The correlation between usage and performance causes improved outcomes.",
    )

    assert "METHODOLOGY_CORRELATION_CAUSATION_MIXED" in {finding.code for finding in findings}


def test_methodology_audit_accepts_defined_metrics_and_uncertainty():
    audit = MethodologyAudit()

    findings = audit.audit(
        section_title="Results",
        content=(
            "We compare against a baseline. Accuracy is defined as correct predictions "
            "divided by all predictions. The improvement is significant with p < 0.05."
        ),
        paper_type="empirical",
    )

    assert findings == []


def test_methodology_audit_skips_generic_non_method_section():
    audit = MethodologyAudit()

    findings = audit.audit(
        section_title="Results",
        content="Prior work shows attention improves sequence modeling.",
    )

    assert findings == []


def test_research_proposal_profile_skips_methodology_audit_until_completed_results():
    profile = ResearchProposalProfile()

    assert profile.skip_methodology_audit(
        "This section explains participants, datasets, and planned analysis."
    )
    assert not profile.skip_methodology_audit(
        "Results show the planned analysis improved accuracy."
    )
