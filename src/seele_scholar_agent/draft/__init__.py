from .gates import ConflictGate, CoverageGate, PreservationGate
from .integration import DraftIntegrationNode
from .mapping import build_draft_section_context, build_draft_state
from .models import (
    DraftIntegrationState,
    DraftOutlineMapping,
    DraftSectionContext,
    DraftSegment,
    ExistingContentRef,
    OutlineAdaptationDecision,
    PreservePolicy,
    coerce_draft_integration_state,
    coerce_existing_content_ref,
)

__all__ = [
    "ExistingContentRef",
    "DraftSegment",
    "PreservePolicy",
    "DraftOutlineMapping",
    "OutlineAdaptationDecision",
    "DraftIntegrationState",
    "DraftSectionContext",
    "DraftIntegrationNode",
    "PreservationGate",
    "CoverageGate",
    "ConflictGate",
    "build_draft_state",
    "build_draft_section_context",
    "coerce_existing_content_ref",
    "coerce_draft_integration_state",
]
