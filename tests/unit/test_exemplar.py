from seele_scholar_agent.exemplar import (
    ExemplarChunk,
    ExemplarMaterial,
    ExemplarPlannerContextNode,
    ExemplarPolicy,
    ExemplarSectionRetrieverNode,
    SimilarityGateNode,
)
from seele_scholar_agent.state import QualityIssue


def test_exemplar_planner_context_builds_structure_and_style(base_state):
    node = ExemplarPlannerContextNode(policy=ExemplarPolicy(enabled=True))
    state = {
        **base_state,
        "exemplar_materials": [
            ExemplarMaterial(
                exemplar_id="ex-1",
                title="Review exemplar",
                usage_role="section_reference",
                outline_patterns=["Motivation -> gap -> contribution"],
                style_notes=["Use cautious synthesis language."],
            )
        ],
    }

    result = node.build(state)

    context = result["exemplar_context"]
    assert "Motivation -> gap -> contribution" in context.outline_patterns
    assert "Use cautious synthesis language." in context.style_notes
    assert context.anti_copying_notes


def test_exemplar_section_retriever_selects_current_section_examples(state_with_outline):
    node = ExemplarSectionRetrieverNode(
        policy=ExemplarPolicy(enabled=True, max_examples_per_section=1)
    )
    state = {
        **state_with_outline,
        "exemplar_chunks": [
            ExemplarChunk(
                exemplar_id="ex-1",
                chunk_id="intro-example",
                section_title="Introduction",
                text="This introduction moves from broad motivation to the research gap.",
            ),
            ExemplarChunk(
                exemplar_id="ex-1",
                chunk_id="method-example",
                section_title="Methods",
                text="This methods section explains data and metrics.",
            ),
        ],
    }

    result = node.retrieve(state)

    examples = result["exemplar_context"].section_examples
    assert [example.chunk_id for example in examples] == ["intro-example"]
    assert examples[0].similarity_score > 0


def test_similarity_gate_reports_copy_risk(state_with_outline):
    node = SimilarityGateNode(
        policy=ExemplarPolicy(enabled=True, max_similarity_ratio=0.3)
    )
    sections = list(state_with_outline["sections"])
    copied_text = "This paragraph is copied from the exemplar almost exactly."
    sections[0] = sections[0].model_copy(update={"content": copied_text, "status": "review"})
    stale_issue = QualityIssue(
        code="EXEMPLAR_COPY_RISK",
        message="old",
        location=sections[0].section_id,
    )
    state = {
        **state_with_outline,
        "sections": sections,
        "quality_issues": [stale_issue],
        "exemplar_context": {
            "section_examples": [
                {
                    "exemplar_id": "ex-1",
                    "chunk_id": "copy-source",
                    "section_title": "Introduction",
                    "text": copied_text,
                }
            ],
        },
    }

    result = node.check(state)

    issues = result["quality_issues"]
    assert len(issues) == 1
    assert issues[0].code == "EXEMPLAR_COPY_RISK"
    assert issues[0].details["matched_exemplar_chunk_id"] == "copy-source"
    assert result["exemplar_similarity_diagnostics"]["status"] == "copy_risk"
