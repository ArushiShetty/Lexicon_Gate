"""Tavily web search → PaperHit (result_kind=web)."""

from __future__ import annotations

import logging
from typing import Any, Callable
from urllib.parse import urlparse

import httpx

from ...config import Settings, get_settings
from ...models.papers import PaperHit

logger = logging.getLogger(__name__)

TAVILY_SEARCH = "https://api.tavily.com/search"

# Research-oriented hosts for the research_web filter (no wildcards in Tavily include_domains).
RESEARCH_DOMAINS: list[str] = [
    "arxiv.org",
    "nih.gov",
    "ncbi.nlm.nih.gov",
    "nature.com",
    "science.org",
    "ieee.org",
    "acm.org",
    "springer.com",
    "link.springer.com",
    "wiley.com",
    "onlinelibrary.wiley.com",
    "frontiersin.org",
    "plos.org",
    "mdpi.com",
    "biorxiv.org",
    "medrxiv.org",
    "ssrn.com",
    "scholar.google.com",
    "semanticscholar.org",
    "openalex.org",
    "doi.org",
    "pnas.org",
    "cell.com",
    "sciencedirect.com",
    "tandfonline.com",
    "oup.com",
    "academic.oup.com",
    "mit.edu",
    "stanford.edu",
    "harvard.edu",
    "ox.ac.uk",
    "cam.ac.uk",
]

JsonPoster = Callable[..., dict[str, Any]]


def tavily_available(settings: Settings | None = None) -> bool:
    settings = settings or get_settings()
    return bool((settings.tavily_api_key or "").strip())


def search_tavily(
    query: str,
    limit: int = 10,
    *,
    research_domains_only: bool = False,
    post_json: JsonPoster | None = None,
    settings: Settings | None = None,
) -> list[PaperHit]:
    query = (query or "").strip()
    if not query:
        return []
    settings = settings or get_settings()
    api_key = (settings.tavily_api_key or "").strip()
    if not api_key:
        return []
    limit = max(1, min(int(limit), 25))
    body: dict[str, Any] = {
        "api_key": api_key,
        "query": query,
        "max_results": limit,
        "include_answer": False,
        "search_depth": "basic",
    }
    if research_domains_only:
        body["include_domains"] = RESEARCH_DOMAINS

    try:
        if post_json is not None:
            payload = post_json(TAVILY_SEARCH, json=body, timeout=settings.paper_search_timeout_s)
        else:
            with httpx.Client(timeout=settings.paper_search_timeout_s, follow_redirects=True) as client:
                response = client.post(TAVILY_SEARCH, json=body)
                response.raise_for_status()
                payload = response.json()
    except Exception as exc:
        logger.warning("Tavily search failed: %s", exc)
        return []

    rows = payload.get("results") or []
    return [hit for row in rows if (hit := paper_from_tavily(row))][:limit]


def paper_from_tavily(row: dict[str, Any] | None) -> PaperHit | None:
    if not row:
        return None
    title = str(row.get("title") or "").strip()
    url = str(row.get("url") or "").strip()
    if not title or not url:
        return None
    content = str(row.get("content") or row.get("snippet") or "").strip() or None
    path = urlparse(url).path.lower()
    looks_pdf = path.endswith(".pdf") or "/pdf/" in path or "arxiv.org/pdf/" in url.lower()
    paper_id = url
    return PaperHit(
        paper_id=paper_id,
        title=title,
        authors=[],
        year=None,
        venue=urlparse(url).netloc or None,
        abstract=content,
        summary=content,
        doi=None,
        pdf_url=url if looks_pdf else None,
        source="tavily",
        open_access=bool(looks_pdf),
        url=url,
        result_kind="web",
        full_text_available=bool(looks_pdf),
    )
