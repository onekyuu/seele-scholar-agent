from seele_scholar_agent.nodes.material_registry import (
    annotate_paper_summaries,
    find_material_entry,
    material_policy_suffix,
)
from seele_scholar_agent.state import MaterialRegistry, MaterialRegistryEntry, PaperMetadata


def _paper() -> PaperMetadata:
    return PaperMetadata(
        paper_id="p1",
        title="Required Paper",
        authors=["Author"],
        abstract="Abstract.",
        doi="10.1000/example",
        source="user_library",
    )


def test_material_registry_matches_paper_by_doi():
    registry = MaterialRegistry(
        entries=[
            MaterialRegistryEntry(
                doi="https://doi.org/10.1000/example",
                required=True,
                source_origin="user_upload",
            )
        ]
    )

    entry = find_material_entry(_paper(), registry)

    assert entry is not None
    assert entry.required is True


def test_material_policy_suffix_describes_optional_boundary():
    registry = MaterialRegistry(
        entries=[
            MaterialRegistryEntry(
                paper_id="p1",
                citation_role="background",
                confidence="low",
                notes="Use for context only.",
            )
        ]
    )

    suffix = material_policy_suffix(_paper(), registry)

    assert "citation_role=background" in suffix
    assert "confidence=low" in suffix
    assert "Use for context only." in suffix


def test_annotate_paper_summaries_is_noop_without_registry():
    summaries = ["[1] Required Paper — Author. Abstract."]

    assert annotate_paper_summaries(summaries, [_paper()], None) == summaries


def test_annotate_paper_summaries_does_not_duplicate_policy():
    registry = MaterialRegistry(entries=[MaterialRegistryEntry(paper_id="p1", required=True)])
    summaries = ["[1] Required Paper — Author. Abstract. | material_policy: required_by_user=true"]

    assert annotate_paper_summaries(summaries, [_paper()], registry) == summaries
