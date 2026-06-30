from seele_scholar_agent.draft import (
    ConflictGate,
    DraftIntegrationNode,
    DraftSegment,
    ExistingContentRef,
    PreservationGate,
    PreservePolicy,
)


def _existing_content(
    *,
    segments: list[DraftSegment],
    preserve_policy: PreservePolicy | None = None,
    user_intent: str = "expand",
) -> ExistingContentRef:
    return ExistingContentRef(
        draft_id="draft-1",
        version_id="v1",
        normalized_content="\n\n".join(segment.text for segment in segments),
        segments=segments,
        preserve_policy=preserve_policy or PreservePolicy(),
        user_intent=user_intent,  # type: ignore[arg-type]
    )


def test_draft_integration_rejects_raw_only_input(base_state):
    node = DraftIntegrationNode()
    state = {
        **base_state,
        "existing_content": {
            "draft_id": "draft-1",
            "version_id": "v1",
            "normalized_content": "One unparsed full draft.",
            "segments": [],
            "preserve_policy": PreservePolicy(),
            "user_intent": "expand",
        },
    }

    result = node.integrate(state)

    assert result["status"] == "failed"
    assert result["quality_issues"][0].code == "DRAFT_SEGMENTS_REQUIRED"
    assert result["quality_issues"][0].blocking is True


def test_draft_integration_maps_segments_to_existing_outline(state_with_outline):
    node = DraftIntegrationNode()
    existing_content = _existing_content(
        segments=[
            DraftSegment(
                segment_id="seg-intro",
                detected_heading="Introduction",
                text="Introduction paragraph describing the motivation.",
                order=1,
            )
        ]
    )
    state = {**state_with_outline, "existing_content": existing_content}

    result = node.integrate(state)

    draft_state = result["draft_integration"]
    assert draft_state.mappings[0].section_id == "section_0"
    assert draft_state.outline_decision.action == "keep_outline"
    assert result["draft_diagnostics"]["mapped_count"] == 1


def test_draft_integration_decides_to_create_outline_without_outline(base_state):
    node = DraftIntegrationNode()
    existing_content = _existing_content(
        segments=[
            DraftSegment(
                segment_id="seg-background",
                detected_heading="Background",
                text="Background paragraph about the research problem.",
                order=1,
            )
        ]
    )
    state = {**base_state, "existing_content": existing_content}

    result = node.integrate(state)

    draft_state = result["draft_integration"]
    assert draft_state.outline_decision.action == "create_outline_from_draft"
    assert draft_state.uncovered_requirements == ["seg-background"]


def test_preservation_gate_reports_missing_protected_segment(state_with_outline):
    protected = DraftSegment(
        segment_id="seg-protected",
        detected_heading="Introduction",
        text="This protected sentence must remain.",
        order=1,
    )
    existing_content = _existing_content(
        segments=[protected],
        preserve_policy=PreservePolicy(protected_segment_ids=["seg-protected"]),
    )
    draft_state = DraftIntegrationNode().integrate(
        {**state_with_outline, "existing_content": existing_content}
    )["draft_integration"]
    sections = list(state_with_outline["sections"])
    sections[0] = sections[0].model_copy(
        update={
            "content": "Completely different generated material.",
            "status": "review",
        }
    )
    state = {
        **state_with_outline,
        "sections": sections,
        "draft_integration": draft_state,
    }

    result = PreservationGate().check(state)

    assert result["quality_issues"][0].code == "DRAFT_PROTECTED_SEGMENT_REMOVED"
    assert result["quality_issues"][0].details["missing_segment_ids"] == ["seg-protected"]


def test_conflict_gate_surfaces_draft_conflicts(state_with_outline):
    existing_content = _existing_content(
        segments=[
            DraftSegment(
                segment_id="seg-conflict",
                detected_heading="Introduction",
                text="Conflicting draft segment.",
                order=1,
                metadata={"conflict": "contradicts citation evidence"},
            )
        ]
    )
    draft_state = DraftIntegrationNode().integrate(
        {**state_with_outline, "existing_content": existing_content}
    )["draft_integration"]
    state = {**state_with_outline, "draft_integration": draft_state}

    result = ConflictGate().check(state)

    assert result["quality_issues"][0].code == "DRAFT_OUTLINE_CONFLICT"
    assert "contradicts citation evidence" in result["quality_issues"][0].details["conflicts"][0]
