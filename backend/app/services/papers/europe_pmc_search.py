"""Europe PMC REST search → PaperHit (no API key required)."""

from __future__ import annotations

import logging
import re
from typing import Any, Callable

import httpx

from ...config import Settings, get_settings
from ...models.papers import PaperHit

logger = logging.getLogger(__name__)

EUROPE_PMC_SEARCH = "https://www.europepmc.org/api/search"
_PMID_RE = re.compile(r"^\d+$")

JsonGetter = Callable[..., dict[str, Any]]


def search_europe_pmc(
    query: str,
    limit: int = 10,
    *,
    get_json: JsonGetter | None = None,
    settings: Settings | None = None,
) -> list[PaperHit]:
    query = (query or "").strip()
    if not query:
        return []
    settings = settings or get_settings()
    limit = max(1, min(int(limit), 25))
    params = {
        "query": query,
        "format": "json",
        "pageSize": limit,
        "resultType": "core",
    }
    try:
        if get_json is not None:
            payload = get_json(EUROPE_PMC_SEARCH, params=params, timeout=settings.paper_search_timeout_s)
        else:
            with httpx.Client(timeout=settings.paper_search_timeout_s, follow_redirects=True) as client:
                response = client.get(EUROPE_PMC_SEARCH, params=params)
                response.raise_for_status()
                payload = response.json()
    except Exception as exc:
        logger.warning("Europe PMC search failed: %s", exc)
        return []

    result_list = payload.get("resultList") if isinstance(payload, dict) else None
    rows = (result_list or {}).get("result") if isinstance(result_list, dict) else []
    if not isinstance(rows, list):
        return []
    return [hit for row in rows if (hit := paper_from_europe_pmc(row))][:limit]


def paper_from_europe_pmc(row: dict[str, Any] | None) -> PaperHit | None:
    if not row:
        return None
    title = str(row.get("title") or "").strip()
    if not title:
        return None

    source_id = str(row.get("id") or "").strip()
    pmid = str(row.get("pmid") or "").strip()
    pmcid = str(row.get("pmcid") or "").strip()
    doi = str(row.get("doi") or "").strip() or None
    if doi:
        doi = doi.removeprefix("https://doi.org/").removeprefix("http://doi.org/")

    paper_id = pmcid or pmid or source_id or (f"doi:{doi}" if doi else title)
    authors: list[str] = []
    author_string = str(row.get("authorString") or "").strip()
    if author_string:
        authors = [a.strip() for a in author_string.split(",") if a.strip()][:12]

    year = None
    pub_year = row.get("pubYear")
    try:
        year = int(pub_year) if pub_year is not None else None
    except (TypeError, ValueError):
        year = None

    abstract = str(row.get("abstractText") or "").strip() or None
    is_oa = str(row.get("isOpenAccess") or "").lower() in {"y", "yes", "true", "1"}
    has_pdf = str(row.get("hasPDF") or "").lower() in {"y", "yes", "true", "1"}

    url = None
    pdf_url = None
    if pmcid:
        pmc = pmcid if pmcid.upper().startswith("PMC") else f"PMC{pmcid}"
        url = f"https://europepmc.org/articles/{pmc}"
        if has_pdf or is_oa:
            pdf_url = f"https://europepmc.org/articles/{pmc}?pdf=render"
    elif pmid and _PMID_RE.match(pmid):
        url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
    elif doi:
        url = f"https://doi.org/{doi}"
    elif source_id:
        url = f"https://europepmc.org/search?query=EXT_ID:{source_id}"

    return PaperHit(
        paper_id=paper_id,
        title=title,
        authors=authors,
        year=year,
        venue=str(row.get("journalTitle") or row.get("bookOrReportDetails") or "").strip() or None,
        citation_count=0,
        abstract=abstract,
        summary=abstract,
        doi=doi,
        pdf_url=pdf_url,
        source="europe_pmc",
        open_access=is_oa or bool(pdf_url),
        url=url,
        result_kind="paper",
        full_text_available=bool(pdf_url) or is_oa,
    )
