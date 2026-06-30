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


def test_research_proposal_profile_flags_incomplete_schedule():
    profile = ResearchProposalProfile()

    review_issues, quality_issues = profile.structural_review_issues(
        "section-1",
        "研究計画・スケジュール",
        "1年次後期にプロトタイプを実装する。",
    )

    assert review_issues[0].type == "format_issue"
    assert quality_issues[0].code == "PROPOSAL_SCHEDULE_PHASES_MISSING"
