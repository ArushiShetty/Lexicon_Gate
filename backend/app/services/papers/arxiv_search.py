"""arXiv Atom API search → PaperHit with OA PDF URLs."""

from __future__ import annotations

import logging
import re
import xml.etree.ElementTree as ET
from typing import Any

import httpx

from ...config import Settings, get_settings
from ...models.papers import PaperHit

logger = logging.getLogger(__name__)

ARXIV_API = "http://export.arxiv.org/api/query"
_ATOM = "{http://www.w3.org/2005/Atom}"
_ARXIV = "{http://arxiv.org/schemas/atom}"
_ID_RE = re.compile(r"arxiv\.org/abs/([^?\s#]+)", re.I)


def search_arxiv(
    query: str,
    limit: int = 10,
    *,
    get_text: Any | None = None,
    settings: Settings | None = None,
) -> list[PaperHit]:
    query = (query or "").strip()
    if not query:
        return []
    settings = settings or get_settings()
    limit = max(1, min(int(limit), 25))
    params = {
        "search_query": f"all:{query}",
        "start": 0,
        "max_results": limit,
        "sortBy": "relevance",
        "sortOrder": "descending",
    }
    try:
        if get_text is not None:
            xml_text = get_text(ARXIV_API, params=params, timeout=settings.paper_search_timeout_s)
        else:
            with httpx.Client(timeout=settings.paper_search_timeout_s, follow_redirects=True) as client:
                response = client.get(ARXIV_API, params=params)
                response.raise_for_status()
                xml_text = response.text
    except Exception as exc:
        logger.warning("arXiv search failed: %s", exc)
        return []
    return parse_arxiv_atom(xml_text)[:limit]


def parse_arxiv_atom(xml_text: str) -> list[PaperHit]:
    if not (xml_text or "").strip():
        return []
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as exc:
        logger.warning("arXiv Atom parse failed: %s", exc)
        return []
    hits: list[PaperHit] = []
    for entry in root.findall(f"{_ATOM}entry"):
        hit = paper_from_arxiv_entry(entry)
        if hit:
            hits.append(hit)
    return hits


def paper_from_arxiv_entry(entry: ET.Element) -> PaperHit | None:
    title = _text(entry.find(f"{_ATOM}title"))
    if not title:
        return None
    title = re.sub(r"\s+", " ", title).strip()
    abs_url = _text(entry.find(f"{_ATOM}id")) or ""
    arxiv_id = _arxiv_id_from_url(abs_url) or abs_url.rsplit("/", 1)[-1]
    if not arxiv_id:
        return None
    authors = [
        _text(a.find(f"{_ATOM}name"))
        for a in entry.findall(f"{_ATOM}author")
        if _text(a.find(f"{_ATOM}name"))
    ]
    summary = _text(entry.find(f"{_ATOM}summary"))
    if summary:
        summary = re.sub(r"\s+", " ", summary).strip()
    published = _text(entry.find(f"{_ATOM}published")) or ""
    year = None
    if len(published) >= 4 and published[:4].isdigit():
        year = int(published[:4])
    doi = None
    doi_el = entry.find(f"{_ARXIV}doi")
    if doi_el is not None and (doi_el.text or "").strip():
        doi = doi_el.text.strip()
    pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
    abs_page = f"https://arxiv.org/abs/{arxiv_id}"
    return PaperHit(
        paper_id=arxiv_id,
        title=title,
        authors=[a for a in authors if a],
        year=year,
        venue="arXiv",
        abstract=summary,
        summary=summary,
        doi=doi,
        pdf_url=pdf_url,
        source="arxiv",
        open_access=True,
        url=abs_page,
        result_kind="paper",
        full_text_available=True,
    )


def arxiv_pdf_url(paper_id_or_url: str | None) -> str | None:
    if not paper_id_or_url:
        return None
    text = paper_id_or_url.strip()
    aid = _arxiv_id_from_url(text)
    # Legacy arXiv ids look like hep-th/9901001 (category starts with a letter).
    # Do not treat DOIs such as 10.1038/… as arXiv ids.
    if not aid and re.match(r"^[A-Za-z][\w.\-]*/\d{4,}(\.v\d+)?$", text):
        aid = text
    if not aid and re.match(r"^\d{4}\.\d{4,5}(v\d+)?$", text):
        aid = text
    if not aid:
        return None
    aid = aid.removesuffix(".pdf")
    return f"https://arxiv.org/pdf/{aid}.pdf"


def _arxiv_id_from_url(url: str) -> str | None:
    match = _ID_RE.search(url or "")
    if match:
        return match.group(1).removesuffix(".pdf")
    if "arxiv.org/pdf/" in (url or "").lower():
        return url.rstrip("/").rsplit("/", 1)[-1].removesuffix(".pdf")
    return None


def _text(el: ET.Element | None) -> str | None:
    if el is None or el.text is None:
        return None
    return el.text.strip() or None
