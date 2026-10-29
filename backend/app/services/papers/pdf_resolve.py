"""Resolve legally available PDF URLs. Does not bypass paywalls."""

from __future__ import annotations

import logging
import re
from urllib.parse import quote, urlparse

from ...config import Settings, get_settings
from ...models.papers import PaperHit
from .search import JsonGetter, OPENALEX_WORKS, default_get_json, openalex_headers

logger = logging.getLogger(__name__)

UNPAYWALL = "https://api.unpaywall.org/v2/{doi}"
_PMC_ARTICLE_RE = re.compile(r"/articles/(?:PMC)?(\d+)", re.I)


def looks_like_pdf_url(url: str | None) -> bool:
    if not url:
        return False
    path = urlparse(url).path.lower()
    return (
        path.endswith(".pdf")
        or path.endswith("/pdf")
        or "/pdf/" in path
        or "arxiv.org/pdf/" in url.lower()
    )


def publisher_pdf_rewrite(url: str | None) -> str | None:
    """Turn known OA article HTML pages into the publisher PDF path."""

    if not url:
        return None
    parsed = urlparse(url)
    host = (parsed.netloc or "").lower()
    base = url.split("?")[0].rstrip("/")
    if looks_like_pdf_url(base):
        return None
    if "mdpi.com" in host:
        return f"{base}/pdf"
    if "frontiersin.org" in host:
        return f"{base}/pdf"
    pmc_id = _pmc_id_from_path(parsed.path)
    if pmc_id and ("ncbi.nlm.nih.gov" in host or host.startswith("pmc.")):
        if "pmc.ncbi.nlm.nih.gov" in host:
            return f"https://pmc.ncbi.nlm.nih.gov/articles/{pmc_id}/pdf/"
        return f"https://www.ncbi.nlm.nih.gov/pmc/articles/{pmc_id}/pdf/"
    if pmc_id and "europepmc.org" in host:
        return f"https://europepmc.org/articles/{pmc_id}?pdf=render"
    return None


def _pmc_id_from_path(path: str) -> str | None:
    match = _PMC_ARTICLE_RE.search(path or "")
    if not match:
        return None
    return f"PMC{match.group(1)}"


def official_paper_url(hit: PaperHit) -> str | None:
    if hit.doi:
        doi = hit.doi.removeprefix("https://doi.org/").removeprefix("http://doi.org/")
        return f"https://doi.org/{doi}"
    if hit.url:
        return hit.url
    if hit.pdf_url:
        return hit.pdf_url
    return None


def pdf_candidates(
    hit: PaperHit,
    *,
    get_json: JsonGetter | None = None,
    settings: Settings | None = None,
) -> list[str]:
    """Ordered unique URLs that might be an open PDF."""

    get_json = get_json or default_get_json
    settings = settings or get_settings()
    ordered: list[str] = []

    def add(url: str | None) -> None:
        if not url:
            return
        url = url.strip()
        rewritten = publisher_pdf_rewrite(url)
        if rewritten:
            url = rewritten
        elif not looks_like_pdf_url(url):
            return
        if url and url not in ordered:
            ordered.append(url)

    add(hit.pdf_url)
    # Prefer arXiv PDF when id/url looks like arXiv.
    from .arxiv_search import arxiv_pdf_url

    add(arxiv_pdf_url(hit.paper_id))
    add(arxiv_pdf_url(hit.url))
    add(arxiv_pdf_url(hit.doi))
    add(hit.url)
    for url in unpaywall_pdf_urls(hit.doi, get_json=get_json, settings=settings):
        add(url)
    for url in openalex_pdf_urls_by_doi(hit.doi, get_json=get_json, settings=settings):
        add(url)
    return ordered


def unpaywall_pdf_urls(
    doi: str | None,
    *,
    get_json: JsonGetter,
    settings: Settings,
) -> list[str]:
    if not doi:
        return []
    doi = doi.removeprefix("https://doi.org/").removeprefix("http://doi.org/")
    email = settings.unpaywall_email or "selfrag@localhost"
    try:
        payload = get_json(
            UNPAYWALL.format(doi=quote(doi, safe="/")),
            params={"email": email},
            timeout=settings.paper_search_timeout_s,
        )
    except Exception as exc:
        logger.warning("Unpaywall lookup failed for %s: %s", doi, exc)
        return []
    urls: list[str] = []
    _collect_oa_location_urls(urls, payload.get("best_oa_location") or {})
    for loc in payload.get("oa_locations") or []:
        _collect_oa_location_urls(urls, loc or {})
    return urls


def _collect_oa_location_urls(urls: list[str], loc: dict) -> None:
    pdf = loc.get("url_for_pdf")
    if pdf:
        urls.append(pdf)
        return
    landing = loc.get("url")
    if landing:
        urls.append(landing)


def openalex_pdf_urls_by_doi(
    doi: str | None,
    *,
    get_json: JsonGetter,
    settings: Settings,
) -> list[str]:
    if not doi:
        return []
    doi = doi.removeprefix("https://doi.org/").removeprefix("http://doi.org/")
    try:
        payload = get_json(
            OPENALEX_WORKS,
            params={
                "filter": f"doi:{doi}",
                "per_page": 1,
                "mailto": settings.unpaywall_email or "selfrag@localhost",
            },
            headers=openalex_headers(settings),
            timeout=settings.paper_search_timeout_s,
        )
    except Exception as exc:
        logger.warning("OpenAlex DOI lookup failed for %s: %s", doi, exc)
        return []
    rows = payload.get("results") or []
    if not rows:
        return []
    row = rows[0]
    urls: list[str] = []
    best = row.get("best_oa_location") or {}
    loc = row.get("primary_location") or {}
    oa = row.get("open_access") or {}
    for key in (best.get("pdf_url"), loc.get("pdf_url"), oa.get("oa_url")):
        if key:
            urls.append(key)
    return urls
