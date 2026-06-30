from seele_scholar_agent.profiles import (
    DefaultDocumentProfile,
    ResearchProposalProfile,
)


def test_default_profile_review_input_policy():
    profile = DefaultDocumentProfile()

    assert profile.review_document_type == "academic_paper"
    assert profile.review_policy_text() == "Review as an academic paper section."


def test_research_proposal_profile_review_input_policy():
    profile = ResearchProposalProfile()

    assert profile.review_document_type == "research_proposal"
    assert "Japanese graduate-school research proposal" in profile.review_policy_text()
