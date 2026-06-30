from seele_scholar_agent.profiles import (
    DefaultDocumentProfile,
    ResearchProposalProfile,
)
from seele_scholar_agent.state import ReviewIssue, ReviewResult


def test_default_profile_review_input_policy():
    profile = DefaultDocumentProfile()

    assert profile.review_document_type == "academic_paper"
    assert profile.uses_specialized_review_policy is False
    assert profile.review_policy_text() == "Review as an academic paper section."
    assert profile.review_diagnostic_fields("Methods", "content") == {
        "proposal_profile": False,
        "reviewer_mode": "academic_review",
        "missing_core_tasks": [],
    }


def test_research_proposal_profile_review_input_policy():
    profile = ResearchProposalProfile()

    assert profile.review_document_type == "research_proposal"
    assert profile.uses_specialized_review_policy is True
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


def test_research_proposal_profile_flags_missing_core_tasks():
    profile = ResearchProposalProfile()

    review_issues, quality_issues = profile.structural_review_issues(
        "section-1",
        "研究方法・計画",
        "本研究の目的は音響体験を明らかにすることである。",
    )

    assert any(issue.blocking for issue in review_issues)
    assert "PROPOSAL_CORE_TASK_MISSING" in {issue.code for issue in quality_issues}
    diagnostics = profile.review_diagnostic_fields(
        "研究方法・計画",
        "本研究の目的は音響体験を明らかにすることである。",
    )
    assert set(diagnostics["missing_core_tasks"]) >= {"method", "plan"}


def test_research_proposal_profile_defers_plan_claim_without_citation():
    profile = ResearchProposalProfile()

    assert profile.should_defer_claim(
        "1年次後期にプロトタイプを実装し、評価手法を検証する。",
        (),
        "研究計画・スケジュール",
    )
    assert not profile.should_defer_claim(
        "先行研究は音響設計の重要性を示している。",
        (),
        "研究背景",
    )


def test_research_proposal_profile_applies_review_policy():
    profile = ResearchProposalProfile()
    review = ReviewResult(
        approved=False,
        score=8,
        issues=[
            ReviewIssue(
                type="missing_citation",
                description="Missing citation.",
                suggestion="Add one.",
                location="sentence",
            )
        ],
        summary="Needs citation.",
    )

    updated, quality_issues = profile.apply_review_policy(review, [])

    assert updated.approved is True
    assert quality_issues == []
    assert updated.issues[0].category == "citation_warning"
