import re
from dataclasses import dataclass

import httpx

from ..logging import get_logger

logger = get_logger(__name__)

_CROSSREF_BASE = "https://api.crossref.org/works"
_DOI_ORG_BASE = "https://doi.org"
_USER_AGENT = "SeeleScholarAgent/1.0 (mailto:agent@seele-scholar.local)"

_DOI_FROM_URL_RE = re.compile(r"(?:https?://doi\.org/|https?://dx\.doi\.org/)(.+)")
_ARXIV_DOI_RE = re.compile(r"arxiv\.org/abs/([\d.]+(?:v\d+)?)")
_BIBTEX_FIELD_RE = re.compile(r"^\s*(\w+)\s*=\s*\{(.*)\}\s*,?\s*$", re.MULTILINE | re.DOTALL)
_BIBTEX_YEAR_RE = re.compile(r"year\s*=\s*\{?(\d{4})\}?", re.IGNORECASE)


@dataclass
class CrossRefMetadata:
    doi: str
    year: int | None
    venue: str | None
    authors: list[str]


def extract_doi_from_url(url: str) -> str | None:
    m = _DOI_FROM_URL_RE.match(url)
    if m:
        return m.group(1)
    m = _ARXIV_DOI_RE.search(url)
    if m:
        return f"10.48550/arXiv.{m.group(1)}"
    return None


def _parse_bibtex_year(bibtex: str) -> int | None:
    m = _BIBTEX_YEAR_RE.search(bibtex)
    if m:
        try:
            return int(m.group(1))
        except ValueError:
            pass
    return None


def _parse_bibtex_field(bibtex: str, field: str) -> str | None:
    pattern = re.compile(
        rf"^\s*{field}\s*=\s*\{{(.*?)\}}\s*,?\s*$",
        re.MULTILINE | re.DOTALL | re.IGNORECASE,
    )
    m = pattern.search(bibtex)
    if m:
        return m.group(1).strip()
    return None


def _parse_crossref_json(data: dict) -> CrossRefMetadata:
    doi = data.get("DOI", "")

    year: int | None = None
    for date_key in ("published-print", "published-online", "created"):
        parts = data.get(date_key, {}).get("date-parts", [[]])
        if parts and parts[0]:
            try:
                year = int(parts[0][0])
                break
            except (ValueError, TypeError):
                pass

    container = data.get("container-title", [])
    venue: str | None = container[0] if container else None
    if not venue:
        venue = data.get("publisher")

    raw_authors: list[dict] = data.get("author", [])
    authors: list[str] = []
    for a in raw_authors:
        given = a.get("given", "")
        family = a.get("family", "")
        if family:
            authors.append(f"{family}, {given}".strip(", "))
        elif given:
            authors.append(given)

    return CrossRefMetadata(doi=doi, year=year, venue=venue, authors=authors)


async def fetch_metadata(doi: str, *, timeout: float = 15.0) -> CrossRefMetadata | None:
    """Fetch paper metadata from CrossRef REST API by DOI.

    Returns None on any error (network, 404, rate-limit, etc.).
    """
    url = f"{_CROSSREF_BASE}/{doi}"
    headers = {"User-Agent": _USER_AGENT}
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.get(url, headers=headers)
            if resp.status_code == 404:
                logger.debug("crossref doi not found", doi=doi)
                return None
            resp.raise_for_status()
            data = resp.json().get("message", {})
            return _parse_crossref_json(data)
    except httpx.HTTPStatusError as exc:
        logger.warning("crossref api error", doi=doi, status=exc.response.status_code)
        return None
    except Exception as exc:
        logger.warning("crossref fetch failed", doi=doi, error=str(exc))
        return None
