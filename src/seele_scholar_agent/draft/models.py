from typing import Any, Literal

from pydantic import BaseModel, Field, ValidationError

DraftSource = Literal["user_paste", "user_upload", "business_import"]
DraftIntent = Literal["continue", "expand", "rewrite", "polish", "reference_only"]
DraftRole = Literal[
    "title",
    "abstract",
    "background",
    "prior_work",
    "purpose",
    "method",
    "plan",
    "expected_outcome",
    "conclusion",
    "other",
]
PreserveMode = Literal[
    "preserve_as_much_as_possible",
    "rewrite_if_needed",
    "use_as_reference_only",
]
OutlineAction = Literal[
    "keep_outline",
    "revise_outline",
    "create_outline_from_draft",
]


class PreservePolicy(BaseModel):
    mode: PreserveMode = "preserve_as_much_as_possible"
    protected_segment_ids: list[str] = Field(default_factory=list)
    allow_reorder: bool = True
    allow_split_merge: bool = True


class DraftSegment(BaseModel):
    segment_id: str
    text: str
    order: int
    detected_heading: str | None = None
    inferred_role: DraftRole = "other"
    language: str | None = None
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ExistingContentRef(BaseModel):
    draft_id: str
    version_id: str
    source: DraftSource = "user_paste"
    raw_content_preview: str | None = None
    normalized_content: str | None = None
    segments: list[DraftSegment] = Field(default_factory=list)
    preserve_policy: PreservePolicy
    user_intent: DraftIntent = "expand"


class DraftOutlineMapping(BaseModel):
    segment_id: str
    section_id: str | None = None
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    mapping_reason: str = ""


class OutlineAdaptationDecision(BaseModel):
    action: OutlineAction
    reasons: list[str] = Field(default_factory=list)


class DraftIntegrationState(BaseModel):
    existing_content: ExistingContentRef
    mappings: list[DraftOutlineMapping] = Field(default_factory=list)
    outline_decision: OutlineAdaptationDecision | None = None
    uncovered_requirements: list[str] = Field(default_factory=list)
    conflicts: list[str] = Field(default_factory=list)


class DraftSectionContext(BaseModel):
    mapped_segments: list[DraftSegment] = Field(default_factory=list)
    unmapped_related_segments: list[DraftSegment] = Field(default_factory=list)
    preserve_policy: PreservePolicy
    user_intent: DraftIntent


def coerce_existing_content_ref(raw: Any) -> ExistingContentRef | None:
    if raw is None:
        return None
    if isinstance(raw, ExistingContentRef):
        return raw
    try:
        return ExistingContentRef.model_validate(raw)
    except ValidationError:
        return None


def coerce_draft_integration_state(raw: Any) -> DraftIntegrationState | None:
    if raw is None:
        return None
    if isinstance(raw, DraftIntegrationState):
        return raw
    try:
        return DraftIntegrationState.model_validate(raw)
    except ValidationError:
        return None
