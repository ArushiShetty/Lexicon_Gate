"""Semantic Scholar + OpenAlex + arXiv + Tavily paper/web search.

Google Scholar is not queried. Hits are normalized to ``PaperHit``.
"""

from __future__ import annotations

import concurrent.futures
import logging
import re
import threading
import time
from typing import Any, Callable
from urllib.parse import quote, urlparse

import httpx

from ...config import Settings, get_settings
from ...models.papers import PaperHit

logger = logging.getLogger(__name__)

SS_SEARCH = "https://api.semanticscholar.org/graph/v1/paper/search"
SS_PAPER = "https://api.semanticscholar.org/graph/v1/paper/{paper_id}"
SS_FIELDS = (
    "title,authors,year,venue,citationCount,abstract,externalIds,openAccessPdf,url"
)
OPENALEX_WORKS = "https://api.openalex.org/works"
OPENALEX_WORK = "https://api.openalex.org/works/{paper_id}"
CROSSREF_WORKS = "https://api.crossref.org/works"
CROSSREF_WORK = "https://api.crossref.org/works/{paper_id}"

RETRYABLE_STATUS = {429, 500, 502, 503, 504}
MAX_RETRIES = 3
MAX_RETRY_WAIT_S = 2.5
HOST_MIN_INTERVAL_S = 1.05

JsonGetter = Callable[..., dict[str, Any]]
BytesGetter = Callable[..., tuple[bytes, str]]

_host_lock = threading.Lock()
_host_last: dict[str, float] = {}
_cache_lock = threading.Lock()
_search_cache: dict[tuple[str, int, str], tuple[float, list[PaperHit], str]] = {}
CACHE_TTL_S = 120.0


def _timeout(settings: Settings | None = None) -> float:
    return float((settings or get_settings()).paper_search_timeout_s)


def _contact_email(settings: Settings | None = None) -> str:
    settings = settings or get_settings()
    return (settings.unpaywall_email or "selfrag@localhost").strip() or "selfrag@localhost"


def _ss_headers(settings: Settings | None = None) -> dict[str, str]:
    settings = settings or get_settings()
    headers = {"User-Agent": f"Enhanced-Self-RAG/1.0 (mailto:{_contact_email(settings)})"}
    if settings.semantic_scholar_api_key:
        headers["x-api-key"] = settings.semantic_scholar_api_key
    return headers


def openalex_headers(settings: Settings | None = None) -> dict[str, str]:
    return {"User-Agent": f"Enhanced-Self-RAG/1.0 (mailto:{_contact_email(settings)})"}


def _pace(url: str) -> None:
    host = urlparse(url).netloc
    with _host_lock:
        now = time.monotonic()
        wait = HOST_MIN_INTERVAL_S - (now - _host_last.get(host, 0.0))
        if wait > 0:
            time.sleep(wait)
        _host_last[host] = time.monotonic()


def _retry_wait(response: httpx.Response, attempt: int) -> float:
    header = response.headers.get("Retry-After")
    if header:
        try:
            return min(MAX_RETRY_WAIT_S, max(0.2, float(header)))
        except ValueError:
            pass
    return min(MAX_RETRY_WAIT_S, 0.8 * (2**attempt))


def default_get_json(
    url: str,
    params: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
    timeout: float | None = None,
) -> dict[str, Any]:
    last_error: Exception | None = None
    for attempt in range(MAX_RETRIES):
        _pace(url)
        try:
            with httpx.Client(timeout=timeout or _timeout(), follow_redirects=True) as client:
                response = client.get(url, params=params, headers=headers or {})
                # 429: fail over to the next index instead of stalling the search.
                if (
                    response.status_code in RETRYABLE_STATUS
                    and response.status_code != 429
                    and attempt < MAX_RETRIES - 1
                ):
                    wait = _retry_wait(response, attempt)
                    logger.warning(
                        "Retrying %s after HTTP %s (wait %.1fs)",
                        urlparse(url).path,
                        response.status_code,
                        wait,
                    )
                    time.sleep(wait)
                    continue
                response.raise_for_status()
                payload = response.json()
                if not isinstance(payload, dict):
                    return {"data": payload}
                return payload
        except (httpx.TimeoutException, httpx.TransportError) as exc:
            last_error = exc
            if attempt < MAX_RETRIES - 1:
                time.sleep(min(MAX_RETRY_WAIT_S, 0.8 * (2**attempt)))
                continue
            raise
    if last_error:
        raise last_error
    raise RuntimeError(f"Request failed: {url}")


DOWNLOAD_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
    ),
    "Accept": "application/pdf,application/octet-stream;q=0.9,*/*;q=0.8",
}


def default_get_bytes(
    url: str,
    timeout: float | None = None,
    max_bytes: int | None = None,
) -> tuple[bytes, str]:
    settings = get_settings()
    cap = max_bytes if max_bytes is not None else settings.paper_pdf_max_bytes
    with httpx.Client(timeout=timeout or _timeout(), follow_redirects=True) as client:
        with client.stream("GET", url, headers=DOWNLOAD_HEADERS) as response:
            response.raise_for_status()
            ctype = response.headers.get("content-type", "")
            chunks: list[bytes] = []
            size = 0
            for chunk in response.iter_bytes():
                size += len(chunk)
                if size > cap:
                    raise ValueError("PDF exceeds size limit")
                chunks.append(chunk)
            return b"".join(chunks), ctype


def search_papers(
    query: str,
    limit: int = 10,
    *,
    search_filter: str = "all",
    get_json: JsonGetter | None = None,
    settings: Settings | None = None,
) -> tuple[list[PaperHit], str]:
    """Return (hits, provider_label). Merges providers by filter; Crossref is last resort."""

    query = (query or "").strip()
    if not query:
        return [], "none"
    injected = get_json is not None
    get_json = get_json or default_get_json
    settings = settings or get_settings()
    limit = max(1, min(int(limit), 25))
    filt = (search_filter or "all").strip().lower()
    if filt not in {"all", "academic", "research_web", "open_access"}:
        filt = "all"
    cache_key = (query.lower(), limit, filt)
    if not injected:
        now = time.monotonic()
        with _cache_lock:
            cached = _search_cache.get(cache_key)
            if cached and now - cached[0] < CACHE_TTL_S:
                return cached[1], cached[2]

    hits, provider = _search_with_filter(
        query, limit, filt, get_json=get_json, settings=settings
    )
    if not injected and hits:
        with _cache_lock:
            _search_cache[cache_key] = (time.monotonic(), hits, provider)
    return hits, provider


def search_papers_detailed(
    query: str,
    limit: int = 10,
    *,
    search_filter: str = "all",
    get_json: JsonGetter | None = None,
    settings: Settings | None = None,
) -> tuple[list[PaperHit], str, list[str], list[str]]:
    """Like search_papers but also returns providers_used and notes."""

    query = (query or "").strip()
    if not query:
        return [], "none", [], []
    get_json = get_json or default_get_json
    settings = settings or get_settings()
    limit = max(1, min(int(limit), 25))
    filt = (search_filter or "all").strip().lower()
    if filt not in {"all", "academic", "research_web", "open_access"}:
        filt = "all"
    return _search_with_filter_detailed(
        query, limit, filt, get_json=get_json, settings=settings
    )


def _search_with_filter(
    query: str,
    limit: int,
    filt: str,
    *,
    get_json: JsonGetter,
    settings: Settings,
) -> tuple[list[PaperHit], str]:
    hits, provider, _used, _notes = _search_with_filter_detailed(
        query, limit, filt, get_json=get_json, settings=settings
    )
    return hits, provider


def _search_with_filter_detailed(
    query: str,
    limit: int,
    filt: str,
    *,
    get_json: JsonGetter,
    settings: Settings,
) -> tuple[list[PaperHit], str, list[str], list[str]]:
    from .arxiv_search import search_arxiv
    from .tavily_search import search_tavily, tavily_available

    notes: list[str] = []
    used: list[str] = []

    def run_academic() -> list[PaperHit]:
        nonlocal used
        buckets: list[list[PaperHit]] = []
        academic_jobs: list[tuple[str, Callable[[], list[PaperHit]]]] = [
            (
                "semantic_scholar",
                lambda: search_semantic_scholar(
                    query, limit, get_json=get_json, settings=settings
                ),
            ),
            (
                "openalex",
                lambda: search_openalex(query, limit, get_json=get_json, settings=settings),
            ),
            (
                "arxiv",
                lambda: search_arxiv(query, limit, settings=settings),
            ),
        ]
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
            futures = {pool.submit(job): name for name, job in academic_jobs}
            for fut in concurrent.futures.as_completed(futures):
                name = futures[fut]
                try:
                    part = fut.result() or []
                except Exception as exc:
                    logger.warning("%s search failed: %s", name, exc)
                    continue
                if part:
                    used.append(name)
                    buckets.append(part)
        merged = merge_dedupe_hits([h for bucket in buckets for h in bucket], limit=limit)
        if not merged:
            try:
                cr = search_crossref(query, limit, get_json=get_json, settings=settings)
            except Exception as exc:
                logger.warning("crossref search failed: %s", exc)
                cr = []
            if cr:
                used.append("crossref")
                merged = cr[:limit]
        return merged

    def run_web(*, research_only: bool) -> list[PaperHit]:
        nonlocal used
        if not tavily_available(settings):
            notes.append(
                "Web search skipped: set SELFRAG_TAVILY_API_KEY for Research Websites / All."
            )
            return []
        try:
            part = search_tavily(
                query,
                limit,
                research_domains_only=research_only,
                settings=settings,
            )
        except Exception as exc:
            logger.warning("tavily search failed: %s", exc)
            return []
        if part:
            used.append("tavily")
        return part

    if filt == "research_web":
        hits = run_web(research_only=True)[:limit]
    elif filt == "academic":
        hits = run_academic()
    elif filt == "open_access":
        hits = [
            h
            for h in run_academic()
            if h.open_access or h.pdf_url or h.full_text_available
        ][:limit]
    else:  # all
        half = max(1, (limit + 1) // 2)
        academic = run_academic()[:half]
        web = run_web(research_only=False)[: max(1, limit - len(academic))]
        hits = merge_dedupe_hits(academic + web, limit=limit)

    provider = "+".join(used) if used else "none"
    return hits, provider, used, notes


def merge_dedupe_hits(hits: list[PaperHit], *, limit: int) -> list[PaperHit]:
    """Dedupe by DOI, arXiv id, or normalized title+year; preserve first occurrence order."""

    seen: set[str] = set()
    out: list[PaperHit] = []
    for hit in hits:
        key = _dedupe_key(hit)
        if key in seen:
            continue
        seen.add(key)
        out.append(hit)
        if len(out) >= limit:
            break
    return out


def _dedupe_key(hit: PaperHit) -> str:
    if hit.doi:
        return f"doi:{hit.doi.strip().lower()}"
    if hit.source == "arxiv" or (hit.url and "arxiv.org" in hit.url.lower()):
        from .arxiv_search import arxiv_pdf_url

        pdf = arxiv_pdf_url(hit.paper_id) or arxiv_pdf_url(hit.url or "")
        if pdf:
            return f"arxiv:{pdf.rsplit('/', 1)[-1].removesuffix('.pdf').lower()}"
    title = re.sub(r"\s+", " ", (hit.title or "").strip().lower())
    year = hit.year or ""
    return f"title:{title}|{year}"


def search_semantic_scholar(
    query: str,
    limit: int,
    *,
    get_json: JsonGetter,
    settings: Settings,
) -> list[PaperHit]:
    payload = get_json(
        SS_SEARCH,
        params={"query": query, "limit": limit, "fields": SS_FIELDS},
        headers=_ss_headers(settings),
        timeout=_timeout(settings),
    )
    rows = payload.get("data") or []
    return [hit for row in rows if (hit := paper_from_semantic_scholar(row))][:limit]


def search_openalex(
    query: str,
    limit: int,
    *,
    get_json: JsonGetter,
    settings: Settings,
) -> list[PaperHit]:
    email = _contact_email(settings)
    payload = get_json(
        OPENALEX_WORKS,
        params={"search": query, "per_page": limit, "mailto": email},
        headers=openalex_headers(settings),
        timeout=_timeout(settings),
    )
    rows = payload.get("results") or []
    return [hit for row in rows if (hit := paper_from_openalex(row))][:limit]


def search_crossref(
    query: str,
    limit: int,
    *,
    get_json: JsonGetter,
    settings: Settings,
) -> list[PaperHit]:
    payload = get_json(
        CROSSREF_WORKS,
        params={"query": query, "rows": limit},
        headers=openalex_headers(settings),
        timeout=_timeout(settings),
    )
    message = payload.get("message") if isinstance(payload.get("message"), dict) else payload
    rows = (message or {}).get("items") or []
    return [hit for row in rows if (hit := paper_from_crossref(row))][:limit]


def fetch_paper(
    paper_id: str,
    source: str,
    *,
    get_json: JsonGetter | None = None,
    settings: Settings | None = None,
) -> PaperHit | None:
    get_json = get_json or default_get_json
    settings = settings or get_settings()
    if source == "semantic_scholar":
        payload = get_json(
            SS_PAPER.format(paper_id=quote(paper_id, safe="")),
            params={"fields": SS_FIELDS},
            headers=_ss_headers(settings),
            timeout=_timeout(settings),
        )
        return paper_from_semantic_scholar(payload)
    if source == "openalex":
        oid = paper_id
        if oid.startswith("https://"):
            oid = oid.rsplit("/", 1)[-1]
        payload = get_json(
            OPENALEX_WORK.format(paper_id=oid),
            headers=openalex_headers(settings),
            timeout=_timeout(settings),
        )
        return paper_from_openalex(payload)
    if source == "crossref":
        doi = paper_id.removeprefix("https://doi.org/").removeprefix("http://doi.org/")
        payload = get_json(
            CROSSREF_WORK.format(paper_id=quote(doi, safe="")),
            headers=openalex_headers(settings),
            timeout=_timeout(settings),
        )
        message = payload.get("message") if isinstance(payload.get("message"), dict) else payload
        return paper_from_crossref(message if isinstance(message, dict) else payload)
    if source == "arxiv":
        from .arxiv_search import search_arxiv

        hits = search_arxiv(paper_id, limit=1, settings=settings)
        for hit in hits:
            if hit.paper_id == paper_id or paper_id in (hit.paper_id or ""):
                return hit
        return hits[0] if hits else None
    if source == "tavily":
        # Snapshot-only; no server-side fetch for ephemeral web hits.
        return None
    return None


def _as_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def paper_from_semantic_scholar(row: dict[str, Any] | None) -> PaperHit | None:
    if not row or not row.get("title"):
        return None
    authors = []
    for author in row.get("authors") or []:
        name = author.get("name") if isinstance(author, dict) else str(author)
        if name:
            authors.append(name)
    ids = row.get("externalIds") or {}
    doi = ids.get("DOI") if isinstance(ids, dict) else None
    arxiv_id = ids.get("ArXiv") if isinstance(ids, dict) else None
    oa = row.get("openAccessPdf") or {}
    pdf_url = oa.get("url") if isinstance(oa, dict) else None
    if not pdf_url and arxiv_id:
        from .arxiv_search import arxiv_pdf_url

        pdf_url = arxiv_pdf_url(str(arxiv_id))
    year = row.get("year")
    try:
        year_int = int(year) if year is not None else None
    except (TypeError, ValueError):
        year_int = None
    return PaperHit(
        paper_id=str(row.get("paperId") or row.get("paper_id") or ""),
        title=str(row["title"]).strip(),
        authors=authors,
        year=year_int,
        venue=(str(row["venue"]).strip() if row.get("venue") else None) or None,
        citation_count=_as_int(row.get("citationCount"), 0),
        abstract=(str(row["abstract"]).strip() if row.get("abstract") else None) or None,
        doi=str(doi).strip() if doi else None,
        pdf_url=pdf_url,
        source="semantic_scholar",
        open_access=bool(pdf_url),
        url=row.get("url"),
        result_kind="paper",
        full_text_available=bool(pdf_url),
    )


def paper_from_openalex(row: dict[str, Any] | None) -> PaperHit | None:
    if not row:
        return None
    title = row.get("display_name") or row.get("title")
    if not title:
        return None
    authors = []
    for item in row.get("authorships") or []:
        author = (item or {}).get("author") or {}
        name = author.get("display_name")
        if name:
            authors.append(name)
    venue = None
    loc = row.get("primary_location") or {}
    source = loc.get("source") or {}
    if source.get("display_name"):
        venue = source["display_name"]
    oa = row.get("open_access") or {}
    best = row.get("best_oa_location") or {}
    pdf_url = best.get("pdf_url") or loc.get("pdf_url") or oa.get("oa_url")
    doi = row.get("doi")
    if isinstance(doi, str):
        doi = doi.replace("https://doi.org/", "").replace("http://doi.org/", "")
    raw_id = str(row.get("id") or "")
    paper_id = raw_id.rsplit("/", 1)[-1] if raw_id else ""
    year = row.get("publication_year")
    try:
        year_int = int(year) if year is not None else None
    except (TypeError, ValueError):
        year_int = None
    return PaperHit(
        paper_id=paper_id or raw_id,
        title=str(title).strip(),
        authors=authors,
        year=year_int,
        venue=venue,
        citation_count=_as_int(row.get("cited_by_count"), 0),
        abstract=abstract_from_inverted(row.get("abstract_inverted_index")),
        doi=doi or None,
        pdf_url=pdf_url,
        source="openalex",
        open_access=bool(pdf_url),
        url=raw_id or None,
        result_kind="paper",
        full_text_available=bool(pdf_url),
    )


def paper_from_crossref(row: dict[str, Any] | None) -> PaperHit | None:
    if not row:
        return None
    titles = row.get("title") or []
    title = titles[0] if isinstance(titles, list) and titles else titles if isinstance(titles, str) else None
    if not title:
        return None
    authors = []
    for author in row.get("author") or []:
        if not isinstance(author, dict):
            continue
        name = " ".join(part for part in (author.get("given"), author.get("family")) if part)
        if name:
            authors.append(name)
    year_int = None
    for key in ("published-print", "published-online", "published", "issued"):
        parts = (row.get(key) or {}).get("date-parts") if isinstance(row.get(key), dict) else None
        if parts and parts[0]:
            try:
                year_int = int(parts[0][0])
            except (TypeError, ValueError, IndexError):
                year_int = None
            if year_int:
                break
    containers = row.get("container-title") or []
    venue = containers[0] if isinstance(containers, list) and containers else None
    doi = row.get("DOI")
    abstract = row.get("abstract")
    if isinstance(abstract, str):
        abstract = abstract.replace("<jats:p>", " ").replace("</jats:p>", " ").strip() or None
    else:
        abstract = None
    return PaperHit(
        paper_id=str(doi or row.get("URL") or title),
        title=str(title).strip(),
        authors=authors,
        year=year_int,
        venue=str(venue).strip() if venue else None,
        citation_count=_as_int(row.get("is-referenced-by-count"), 0),
        abstract=abstract,
        doi=str(doi).strip() if doi else None,
        pdf_url=None,
        source="crossref",
        open_access=False,
        url=row.get("URL"),
        result_kind="paper",
        full_text_available=False,
    )


def abstract_from_inverted(index: dict[str, list[int]] | None) -> str | None:
    if not index or not isinstance(index, dict):
        return None
    positions: list[tuple[int, str]] = []
    for word, slots in index.items():
        for slot in slots or []:
            positions.append((int(slot), str(word)))
    if not positions:
        return None
    positions.sort()
    return " ".join(word for _slot, word in positions)


def hit_from_import_request(payload: Any) -> PaperHit | None:
    """Use a client-supplied snapshot when title is present; otherwise None (caller fetches)."""

    title = getattr(payload, "title", None)
    if not title:
        return None
    return PaperHit(
        paper_id=payload.paper_id,
        title=title,
        authors=list(payload.authors or []),
        year=payload.year,
        venue=payload.venue,
        abstract=payload.abstract or getattr(payload, "summary", None),
        summary=getattr(payload, "summary", None),
        doi=payload.doi,
        pdf_url=payload.pdf_url,
        source=payload.source,
        open_access=bool(payload.pdf_url),
        url=getattr(payload, "url", None),
        result_kind=getattr(payload, "result_kind", None) or "paper",
        full_text_available=bool(payload.pdf_url),
    )
