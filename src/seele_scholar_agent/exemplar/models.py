from typing import Any, Literal

from pydantic import BaseModel, Field, ValidationError


class ExemplarMaterial(BaseModel):
    exemplar_id: str
    title: str = ""
    document_type: str = "generic"
    language: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)
    usage_role: Literal[
        "structure_reference",
        "style_reference",
        "section_reference",
        "negative_example",
    ] = "style_reference"
    outline_patterns: list[str] = Field(default_factory=list)
    structure_notes: list[str] = Field(default_factory=list)
    style_notes: list[str] = Field(default_factory=list)


class ExemplarChunk(BaseModel):
    exemplar_id: str
    chunk_id: str
    text: str
    section_title: str = ""
    section_role: str = ""
    style_tags: list[str] = Field(default_factory=list)
    structure_notes: list[str] = Field(default_factory=list)
    similarity_score: float = 0.0
    metadata: dict[str, Any] = Field(default_factory=dict)


class ExemplarContext(BaseModel):
    outline_patterns: list[str] = Field(default_factory=list)
    section_examples: list[ExemplarChunk] = Field(default_factory=list)
    style_notes: list[str] = Field(default_factory=list)
    anti_copying_notes: list[str] = Field(default_factory=list)


class ExemplarPolicy(BaseModel):
    enabled: bool = False
    max_examples_per_section: int = Field(default=2, ge=0)
    allow_direct_quotation: bool = False
    require_similarity_gate: bool = True
    max_similarity_ratio: float = Field(default=0.18, ge=0.0, le=1.0)


def coerce_exemplar_materials(raw: Any) -> list[ExemplarMaterial]:
    if not isinstance(raw, list):
        return []
    materials: list[ExemplarMaterial] = []
    for item in raw:
        if isinstance(item, ExemplarMaterial):
            materials.append(item)
            continue
        try:
            materials.append(ExemplarMaterial.model_validate(item))
        except ValidationError:
            continue
    return materials


def coerce_exemplar_chunks(raw: Any) -> list[ExemplarChunk]:
    if not isinstance(raw, list):
        return []
    chunks: list[ExemplarChunk] = []
    for item in raw:
        if isinstance(item, ExemplarChunk):
            chunks.append(item)
            continue
        try:
            chunks.append(ExemplarChunk.model_validate(item))
        except ValidationError:
            continue
    return chunks


def coerce_exemplar_context(raw: Any) -> ExemplarContext:
    if isinstance(raw, ExemplarContext):
        return raw
    if raw is None:
        return ExemplarContext()
    try:
        return ExemplarContext.model_validate(raw)
    except ValidationError:
        return ExemplarContext()
