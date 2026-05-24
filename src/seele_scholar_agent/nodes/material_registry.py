import re
import unicodedata

from ..state import AgentState, MaterialRegistry, MaterialRegistryEntry, PaperMetadata

_DOI_RE = re.compile(r"10\.\d{4,9}/[-._;()/:A-Z0-9]+", re.IGNORECASE)
_NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")


def get_material_registry(state: AgentState) -> MaterialRegistry | None:
    return state.get("material_registry")


def has_material_registry(state: AgentState) -> bool:
    registry = get_material_registry(state)
    return bool(registry and registry.entries)


def find_material_entry(
    paper: PaperMetadata, registry: MaterialRegistry | None
) -> MaterialRegistryEntry | None:
    if registry is None:
        return None
    for entry in registry.entries:
        if _entry_matches_paper(entry, paper):
            return entry
    return None


def find_required_entry_number(
    entry: MaterialRegistryEntry, papers: list[PaperMetadata]
) -> int | None:
    for index, paper in enumerate(papers, 1):
        if _entry_matches_paper(entry, paper):
            return index
    return None


def material_policy_suffix(
    paper: PaperMetadata, registry: MaterialRegistry | None
) -> str:
    entry = find_material_entry(paper, registry)
    if entry is None:
        return ""

    parts = [
        f"citation_role={entry.citation_role}",
        f"confidence={entry.confidence}",
    ]
    if entry.source_origin != "unknown":
        parts.append(f"origin={entry.source_origin}")
    if entry.required:
        parts.append("required_by_user=true")
    if entry.notes.strip():
        parts.append(f"notes={entry.notes.strip()}")
    return " | material_policy: " + "; ".join(parts)


def annotate_paper_summaries(
    summaries: list[str], papers: list[PaperMetadata], registry: MaterialRegistry | None
) -> list[str]:
    if registry is None:
        return summaries

    annotated: list[str] = []
    for index, summary in enumerate(summaries):
        if "material_policy:" in summary:
            annotated.append(summary)
            continue
        suffix = material_policy_suffix(papers[index], registry) if index < len(papers) else ""
        annotated.append(summary + suffix)
    return annotated


def apply_material_registry_priority(
    papers: list[PaperMetadata], registry: MaterialRegistry | None
) -> list[PaperMetadata]:
    if registry is None:
        return papers

    enriched: list[PaperMetadata] = []
    for paper in papers:
        entry = find_material_entry(paper, registry)
        if entry is None:
            enriched.append(paper)
            continue

        priority = paper.user_priority
        if entry.required:
            priority = max(priority, 0.6)
        if entry.source_origin == "user_upload":
            priority = max(priority, 0.4)
        enriched.append(paper.model_copy(update={"user_priority": priority}))
    return enriched


def required_entries(registry: MaterialRegistry | None) -> list[MaterialRegistryEntry]:
    if registry is None:
        return []
    return [entry for entry in registry.entries if entry.required]


def material_display_name(entry: MaterialRegistryEntry) -> str:
    return entry.title or entry.paper_id or entry.doi or "unknown material"


def _entry_matches_paper(entry: MaterialRegistryEntry, paper: PaperMetadata) -> bool:
    if entry.paper_id and entry.paper_id == paper.paper_id:
        return True

    entry_doi = _normalize_doi(entry.doi)
    paper_doi = (
        _normalize_doi(paper.doi)
        or _normalize_doi(paper.url)
        or _normalize_doi(paper.pdf_url)
    )
    if entry_doi and paper_doi and entry_doi == paper_doi:
        return True

    entry_title = _normalize_title(entry.title or "")
    paper_title = _normalize_title(paper.title)
    return bool(entry_title and paper_title and entry_title == paper_title)


def _normalize_doi(value: str | None) -> str | None:
    if not value:
        return None
    match = _DOI_RE.search(value)
    if not match:
        return None
    return match.group(0).rstrip(".,;").lower()


def _normalize_title(title: str) -> str:
    normalized = unicodedata.normalize("NFKD", title).lower()
    normalized = _NON_ALNUM_RE.sub(" ", normalized)
    return " ".join(normalized.split())
