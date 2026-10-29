"""Local originality check: long verbatim copies from the library and academic indexes.

This is not Turnitin. Uploaded papers are scored against Library full text and
against academic search hits, using an open-access PDF body when one is
already linked on the hit. Paywalled PDFs are not fetched. The same work
(title/DOI) is skipped so a paper is not marked as plagiarising itself.
"""

from __future__ import annotations

import logging
import re
from typing import Sequence

from ..config import Settings, get_settings
from ..models.documents import Chunk, ChunkMetadata, Document
from ..models.papers import PaperHit
from ..models.query import (
    EvidenceItem,
    FlaggedSentence,
    PlagiarismMatch,
    PlagiarismReport,
    SectionOriginality,
)
from ..services.citations import title_from_filename
from ..services.ingestion.chunker import split_into_sections
from ..services.ingestion.loaders import LoadedPage
from ..services.llm.extractive_engine import strip_citations
from ..services.papers.search import search_papers
from ..text_utils import split_sentences, tokenize

logger = logging.getLogger(__name__)


def check_plagiarism(
    answer: str,
    evidence: Sequence[EvidenceItem],
    library_chunks: Sequence[Chunk] | None = None,
    settings: Settings | None = None,
) -> PlagiarismReport:
    settings = settings or get_settings()
    min_n = max(3, int(settings.plagiarism_min_ngram))
    clean = strip_citations(answer)
    answer_tokens = tokenize(clean)
    if len(answer_tokens) < min_n:
        return PlagiarismReport(originality=1.0, overlap_ratio=0.0, risk="low")

    cited_ids = {item.document_id for item in evidence if item.document_id}
    sources = _sources(evidence, library_chunks or [], cited_ids, min_n)
    if not sources:
        return PlagiarismReport(originality=1.0, overlap_ratio=0.0, risk="low")

    covered = [False] * len(answer_tokens)
    cited_cover = [False] * len(answer_tokens)
    uncited_cover = [False] * len(answer_tokens)
    matches: list[PlagiarismMatch] = []

    i = 0
    while i <= len(answer_tokens) - min_n:
        hit = _longest_hit(answer_tokens, i, min_n, sources)
        if hit is None:
            i += 1
            continue
        start, end, source = hit
        for pos in range(start, end):
            covered[pos] = True
            if source["cited"]:
                cited_cover[pos] = True
            else:
                uncited_cover[pos] = True
        snippet = " ".join(answer_tokens[start:end])
        matches.append(
            PlagiarismMatch(
                text=snippet,
                document_id=source["document_id"],
                document_name=source["document_name"],
                page=source["page"],
                cited=source["cited"],
                n_words=end - start,
            )
        )
        i = end

    n = len(answer_tokens)
    overlap_ratio = sum(covered) / n
    originality = max(0.0, 1.0 - overlap_ratio)
    uncited_ratio = sum(uncited_cover) / n
    quoted_ratio = sum(cited_cover) / n

    if uncited_ratio >= settings.plagiarism_uncited_ratio:
        risk = "high"
    elif quoted_ratio >= settings.plagiarism_quoted_ratio:
        risk = "medium"
    else:
        risk = "low"

    flags: list[str] = []
    if uncited_ratio > 0:
        flags.append(
            f"Uncited wording overlaps {uncited_ratio:.0%} of the answer with the library."
        )
    if quoted_ratio >= settings.plagiarism_quoted_ratio:
        flags.append(
            f"Cited sources supply {quoted_ratio:.0%} of the answer as a long quote."
        )

    flagged = _flag_sentences(clean, matches)
    return PlagiarismReport(
        originality=round(originality, 4),
        overlap_ratio=round(overlap_ratio, 4),
        risk=risk,
        matches=matches,
        flagged_sentences=flagged,
        flags=flags,
    )


def _sources(
    evidence: Sequence[EvidenceItem],
    chunks: Sequence[Chunk],
    cited_ids: set[str],
    min_n: int,
) -> list[dict]:
    seen: set[tuple[str, str]] = set()
    out: list[dict] = []
    for item in evidence:
        key = (item.document_id, item.text)
        if key in seen:
            continue
        seen.add(key)
        tokens = tokenize(item.text)
        if len(tokens) < min_n:
            continue
        out.append(
            {
                "document_id": item.document_id,
                "document_name": item.title or item.document_name,
                "page": item.page,
                "cited": True,
                "grams": _gram_set(tokens, min_n),
            }
        )
    for chunk in chunks:
        key = (chunk.metadata.document_id, chunk.text)
        if key in seen:
            continue
        seen.add(key)
        tokens = tokenize(chunk.text)
        if len(tokens) < min_n:
            continue
        out.append(
            {
                "document_id": chunk.metadata.document_id,
                "document_name": chunk.metadata.document_name,
                "page": chunk.metadata.page,
                "cited": chunk.metadata.document_id in cited_ids,
                "grams": _gram_set(tokens, min_n),
            }
        )
    return out


def _gram_set(tokens: list[str], min_n: int) -> set[tuple[str, ...]]:
    return {tuple(tokens[i : i + min_n]) for i in range(0, len(tokens) - min_n + 1)}


def _longest_hit(
    answer_tokens: list[str],
    start: int,
    min_n: int,
    sources: list[dict],
) -> tuple[int, int, dict] | None:
    needle = tuple(answer_tokens[start : start + min_n])
    candidates = [src for src in sources if needle in src["grams"]]
    if not candidates:
        return None
    end = start + min_n
    while end < len(answer_tokens):
        nxt = tuple(answer_tokens[end - min_n + 1 : end + 1])
        still = [src for src in candidates if nxt in src["grams"]]
        if not still:
            break
        candidates = still
        end += 1
    source = sorted(candidates, key=lambda src: (not src["cited"], src["document_name"]))[0]
    return start, end, source


def _flag_sentences(answer: str, matches: Sequence[PlagiarismMatch]) -> list[FlaggedSentence]:
    if not matches:
        return []
    sentences = split_sentences(answer, min_chars=12) or [answer]
    flagged: list[FlaggedSentence] = []
    for sentence in sentences:
        sent_tokens = tokenize(sentence)
        blob = " ".join(sent_tokens)
        hits = [match for match in matches if match.text and match.text in blob]
        if hits:
            uncited = any(not match.cited for match in hits)
            flagged.append(
                FlaggedSentence(
                    text=sentence,
                    matches=hits,
                    corrections=_correction_hints(hits),
                    kind="verbatim" if uncited else "paraphrased",
                )
            )
    return flagged


def _correction_hints(hits: Sequence[PlagiarismMatch]) -> list[str]:
    lines = ["Reword this passage so it is not a long verbatim copy."]
    seen: set[str] = set()
    for match in hits:
        name = match.document_name or match.document_id or "this source"
        page = f", p. {match.page}" if match.page is not None else ""
        line = f"Cite {name}{page}."
        if line in seen:
            continue
        seen.add(line)
        lines.append(line)
    return lines


def _originality_spans(
    text: str,
    matches: Sequence[PlagiarismMatch],
    chunks: Sequence[Chunk],
) -> tuple[list[FlaggedSentence], list[FlaggedSentence]]:
    source_sets = _source_token_sets(chunks)
    sentences = split_sentences(text, min_chars=12) or ([text] if text.strip() else [])
    spans: list[FlaggedSentence] = []
    for sentence in sentences:
        sent_tokens = tokenize(sentence)
        blob = " ".join(sent_tokens)
        hits = [match for match in matches if match.text and match.text in blob]
        kind, used, corrections = _classify_sentence(sentence, hits, source_sets)
        spans.append(
            FlaggedSentence(
                text=sentence,
                matches=used,
                corrections=corrections,
                kind=kind,
            )
        )
    flagged = [span for span in spans if span.kind != "original"]
    return spans, flagged


def _source_token_sets(chunks: Sequence[Chunk]) -> list[tuple[set[str], str, str, int | None]]:
    out: list[tuple[set[str], str, str, int | None]] = []
    for chunk in chunks:
        parts = split_sentences(chunk.text, min_chars=12) or [chunk.text]
        for part in parts:
            tokens = set(tokenize(part))
            if len(tokens) < 8:
                continue
            out.append(
                (
                    tokens,
                    chunk.metadata.document_name,
                    chunk.metadata.document_id,
                    chunk.metadata.page,
                )
            )
    return out


def _classify_sentence(
    sentence: str,
    hits: Sequence[PlagiarismMatch],
    source_sets: Sequence[tuple[set[str], str, str, int | None]],
) -> tuple[str, list[PlagiarismMatch], list[str]]:
    if hits:
        uncited = any(not match.cited for match in hits)
        kind = "verbatim" if uncited else "paraphrased"
        return kind, list(hits), _correction_hints(hits)
    sent = set(tokenize(sentence))
    if len(sent) < 8 or not source_sets:
        return "original", [], []
    best_j = 0.0
    best: tuple[str, str, int | None] | None = None
    for tokens, name, doc_id, page in source_sets:
        union = sent | tokens
        if not union:
            continue
        score = len(sent & tokens) / len(union)
        if score > best_j:
            best_j = score
            best = (name, doc_id, page)
    if best is not None and best_j >= 0.55:
        name, doc_id, page = best
        match = PlagiarismMatch(
            text=" ".join(tokenize(sentence)[:12]),
            document_id=doc_id,
            document_name=name,
            page=page,
            cited=False,
            n_words=len(sent),
        )
        hints = [
            "Reword this passage; it is close to another source without being a long verbatim copy.",
            f"Cite {name}" + (f", p. {page}." if page is not None else "."),
        ]
        return "paraphrased", [match], hints
    return "original", [], []


def _section_originality(text: str, spans: Sequence[FlaggedSentence]) -> list[SectionOriginality]:
    blocks = split_into_sections(text)
    by_text = {span.text: span for span in spans}
    out = []
    for title, body in blocks:
        sentences = split_sentences(body, min_chars=12) or ([body] if body.strip() else [])
        if not sentences:
            continue
        flagged = 0
        for sentence in sentences:
            span = by_text.get(sentence)
            if span and span.kind != "original":
                flagged += 1
        n = len(sentences)
        out.append(
            SectionOriginality(
                title=title or "Untitled",
                originality=round(1.0 - (flagged / n), 4),
                n_sentences=n,
                n_flagged=flagged,
            )
        )
    return out


def score_uploaded_paper(
    filename: str,
    pages: Sequence[LoadedPage],
    library_chunks: Sequence[Chunk],
    settings: Settings | None = None,
    library_documents: Sequence[Document] | None = None,
) -> "PaperPlagiarismCheck":
    from ..models.query import PaperPlagiarismCheck, PlagiarismSourceShare

    settings = settings or get_settings()
    raw_text = "\n\n".join(page.text for page in pages if getattr(page, "text", "").strip())
    body, stripped_refs = _body_without_references(raw_text)
    words = tokenize(body)
    n_words = len(words)
    title = _paper_title(filename, raw_text)
    doi = _extract_doi(raw_text)
    library_empty = len(library_chunks) == 0
    min_n = max(3, int(settings.plagiarism_min_ngram))
    flags: list[str] = []
    if stripped_refs:
        flags.append("Bibliography / references were not scored.")

    filtered_library, skipped_library = _exclude_self_chunks(
        library_chunks, title, doi, library_documents
    )
    academic_chunks, search_flags, oa_full_texts, abstracts_compared, skipped_hits = (
        _academic_index_chunks(filename, body, words, min_n, settings, title=title, doi=doi)
    )
    flags.extend(search_flags)
    self_skipped = skipped_library + skipped_hits
    if self_skipped:
        flags.append("The same paper (matching title or DOI) was excluded from the score.")

    combined = list(filtered_library) + academic_chunks
    report = check_plagiarism(body, evidence=[], library_chunks=combined, settings=settings)
    denom = max(1, n_words)
    meta = {chunk.metadata.document_id: chunk.metadata for chunk in combined}
    grouped: dict[str, PlagiarismSourceShare] = {}
    for match in report.matches:
        row = grouped.get(match.document_id)
        if row is None:
            extra = meta.get(match.document_id)
            payload = extra.extra if extra else {}
            origin = payload.get("origin") or (
                "academic" if str(match.document_id).startswith("academic:") else "library"
            )
            row = PlagiarismSourceShare(
                document_id=match.document_id,
                document_name=match.document_name or match.document_id,
                origin=origin if origin in {"library", "academic"} else "library",
                url=payload.get("url"),
                doi=payload.get("doi"),
            )
            grouped[match.document_id] = row
        row.n_words += match.n_words
        row.share = round(row.n_words / denom, 4)
    sources = sorted(grouped.values(), key=lambda item: item.share, reverse=True)
    flags = flags + list(report.flags)
    if library_empty:
        flags.insert(
            0,
            "Library is empty. Matching uses academic indexes and any open-access PDFs they list.",
        )
    spans, flagged = _originality_spans(body, report.matches, combined)
    sections = _section_originality(body, spans)
    return PaperPlagiarismCheck(
        filename=filename,
        n_words=n_words,
        n_pages=len(pages),
        similarity=report.overlap_ratio,
        originality=report.originality,
        risk=report.risk,
        library_empty=library_empty,
        matches=report.matches,
        flagged_sentences=flagged,
        sources=sources,
        flags=flags,
        spans=spans,
        sections=sections,
        oa_full_texts=oa_full_texts,
        abstracts_compared=abstracts_compared,
        self_matches_skipped=self_skipped,
    )


_DOI_RE = re.compile(r"\b(10\.\d{4,9}/[-._;()/:A-Z0-9]+)", re.I)
_REF_HEADING = re.compile(
    r"(?im)^\s*(?:\d+(?:\.\d+)*[.\)]\s*)?(references|bibliography|works cited|literature cited)\s*$"
)


def _norm_doi(doi: str | None) -> str:
    if not doi:
        return ""
    text = str(doi).strip().lower()
    for prefix in ("https://doi.org/", "http://doi.org/", "doi:"):
        if text.startswith(prefix):
            text = text[len(prefix) :]
    return text.strip().rstrip(".")


def _extract_doi(text: str) -> str | None:
    match = _DOI_RE.search(text or "")
    return _norm_doi(match.group(1)) if match else None


def _paper_title(filename: str, text: str) -> str:
    file_title = title_from_filename(filename)
    heading = ""
    for line in (text or "").splitlines():
        stripped = line.strip().lstrip("#").strip()
        if stripped:
            heading = stripped
            break
    if heading and len(heading.split()) >= 3:
        return heading[:240]
    if len(file_title.split()) >= 3:
        return file_title
    return heading or file_title


def _norm_title(title: str | None) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (title or "").lower()).strip()


def _same_work(
    title_a: str | None,
    doi_a: str | None,
    title_b: str | None,
    doi_b: str | None,
) -> bool:
    da, db = _norm_doi(doi_a), _norm_doi(doi_b)
    if da and db and da == db:
        return True
    ta, tb = _norm_title(title_a), _norm_title(title_b)
    if not ta or not tb:
        return False
    if ta == tb:
        return True
    wa, wb = set(ta.split()), set(tb.split())
    if min(len(wa), len(wb)) < 5:
        return False
    return len(wa & wb) / len(wa | wb) >= 0.85


def _body_without_references(text: str) -> tuple[str, bool]:
    lines = (text or "").splitlines()
    cut = None
    for index, line in enumerate(lines):
        if _REF_HEADING.match(line):
            cut = index
    if cut is None or cut < 8:
        return text, False
    if cut < int(len(lines) * 0.45):
        return text, False
    body = "\n".join(lines[:cut]).strip()
    return (body or text), bool(body)


def _exclude_self_chunks(
    chunks: Sequence[Chunk],
    title: str,
    doi: str | None,
    documents: Sequence[Document] | None,
) -> tuple[list[Chunk], int]:
    skip_ids: set[str] = set()
    for doc in documents or []:
        if _same_work(title, doi, doc.title or doc.name, doc.doi):
            skip_ids.add(doc.document_id)
    kept: list[Chunk] = []
    skipped_docs: set[str] = set()
    for chunk in chunks:
        extra = chunk.metadata.extra or {}
        name = extra.get("title") or chunk.metadata.document_name
        chunk_doi = extra.get("doi")
        if chunk.metadata.document_id in skip_ids or _same_work(title, doi, name, chunk_doi):
            skipped_docs.add(chunk.metadata.document_id)
            continue
        kept.append(chunk)
    return kept, len(skipped_docs)


def _academic_index_chunks(
    filename: str,
    text: str,
    words: Sequence[str],
    min_n: int,
    settings: Settings,
    title: str = "",
    doi: str | None = None,
) -> tuple[list[Chunk], list[str], int, int, int]:
    queries = _academic_search_queries(filename, text, words, title=title, doi=doi)
    if not queries:
        return [], [], 0, 0, 0
    hits: list[PaperHit] = []
    providers: list[str] = []
    seen: set[tuple[str, str]] = set()
    search_failed = 0
    for query in queries:
        try:
            batch, provider = search_papers(query, limit=6, settings=settings)
        except Exception:
            logger.exception("Academic plagiarism search failed")
            search_failed += 1
            continue
        if provider and provider != "none":
            providers.append(provider)
        for hit in batch:
            key = (hit.source, hit.paper_id)
            if key in seen:
                continue
            seen.add(key)
            hits.append(hit)
    if search_failed == len(queries) and not hits:
        return [], ["Academic search unavailable; scored Library only."], 0, 0, 0

    chunks: list[Chunk] = []
    oa_full_texts = 0
    abstracts_compared = 0
    skipped_hits = 0
    for hit in hits:
        if _same_work(title, doi, hit.title, hit.doi):
            skipped_hits += 1
            continue
        pages = _fetch_oa_pages(hit, settings)
        if pages:
            oa_full_texts += 1
            chunks.extend(_pages_to_chunks(hit, pages, min_n))
            continue
        abstract_chunks = _chunks_from_hits([hit], min_n)
        if abstract_chunks:
            abstracts_compared += 1
            chunks.extend(abstract_chunks)

    flags: list[str] = []
    if providers:
        unique = ", ".join(dict.fromkeys(p.replace("_", " ") for p in providers))
        flags.append(f"Compared sources from {unique}.")
    if oa_full_texts:
        flags.append(f"Used {oa_full_texts} open-access PDF{'s' if oa_full_texts != 1 else ''} (not paywalled copies).")
    if abstracts_compared:
        flags.append(
            f"Used title/abstract for {abstracts_compared} paper{'s' if abstracts_compared != 1 else ''} without a linked PDF."
        )
    return chunks, flags, oa_full_texts, abstracts_compared, skipped_hits


def _academic_search_queries(
    filename: str,
    text: str,
    words: Sequence[str],
    title: str = "",
    doi: str | None = None,
) -> list[str]:
    queries: list[str] = []
    if doi:
        queries.append(doi)
    title = title or _paper_title(filename, text)
    if title and len(title.split()) >= 3:
        queries.append(title[:180])
    for sentence in split_sentences(text, min_chars=60):
        tokens = tokenize(sentence)
        if len(tokens) < 12:
            continue
        probe = " ".join(tokens[:14])
        if title and _norm_title(probe)[:40] == _norm_title(title)[:40]:
            continue
        queries.append(probe)
        if len(queries) >= 3:
            break
    if not queries and words:
        queries.append(" ".join(list(words)[:20])[:180])
    # Unique while preserving order.
    out: list[str] = []
    seen: set[str] = set()
    for query in queries:
        key = query.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(query)
    return out[:3]


def _academic_search_query(filename: str, text: str, words: Sequence[str]) -> str:
    queries = _academic_search_queries(filename, text, words)
    return queries[0] if queries else ""


def _fetch_oa_pages(hit: PaperHit, settings: Settings) -> list[LoadedPage]:
    """Download a PDF only when the index already exposes a PDF URL. No paywall bypass."""

    from ..services.ingestion.loaders import load_bytes
    from ..services.papers.pdf_resolve import looks_like_pdf_url, publisher_pdf_rewrite
    from ..services.papers.search import default_get_bytes

    urls: list[str] = []
    for raw in (hit.pdf_url, hit.url):
        rewritten = publisher_pdf_rewrite(raw)
        candidate = rewritten or raw
        if candidate and (rewritten or looks_like_pdf_url(candidate)) and candidate not in urls:
            urls.append(candidate)
    for url in urls[:2]:
        try:
            data, _ctype = default_get_bytes(
                url,
                timeout=min(12.0, settings.paper_search_timeout_s),
                max_bytes=settings.paper_pdf_max_bytes,
            )
        except Exception as exc:
            logger.warning("OA PDF fetch failed for %s: %s", hit.paper_id, exc)
            continue
        if data[:5] != b"%PDF-":
            continue
        try:
            pages = load_bytes(f"{hit.paper_id}.pdf", data)
        except Exception as exc:
            logger.warning("OA PDF parse failed for %s: %s", hit.paper_id, exc)
            continue
        if any(getattr(page, "text", "").strip() for page in pages):
            return pages
    return []


def _pages_to_chunks(hit: PaperHit, pages: Sequence[LoadedPage], min_n: int) -> list[Chunk]:
    doc_id = f"academic:{hit.source}:{hit.paper_id}"
    url = hit.url
    if not url and hit.doi:
        url = f"https://doi.org/{str(hit.doi).replace('https://doi.org/', '')}"
    extra = {"origin": "academic", "url": url, "doi": hit.doi, "title": hit.title}
    out: list[Chunk] = []
    for page in pages:
        text = (page.text or "").strip()
        if len(tokenize(text)) < min_n:
            continue
        out.append(
            Chunk(
                chunk_id=f"{doc_id}::p{page.page}",
                text=text,
                metadata=ChunkMetadata(
                    document_id=doc_id,
                    document_name=hit.title or doc_id,
                    page=page.page,
                    source="academic",
                    extra=extra,
                ),
            )
        )
    return out


def _chunks_from_hits(hits: Sequence[PaperHit], min_n: int) -> list[Chunk]:
    out: list[Chunk] = []
    for hit in hits:
        blob = f"{hit.title or ''}\n\n{hit.abstract or ''}".strip()
        if len(tokenize(blob)) < min_n:
            continue
        doc_id = f"academic:{hit.source}:{hit.paper_id}"
        url = hit.url
        if not url and hit.doi:
            url = f"https://doi.org/{str(hit.doi).replace('https://doi.org/', '')}"
        out.append(
            Chunk(
                chunk_id=f"{doc_id}::abs",
                text=blob,
                metadata=ChunkMetadata(
                    document_id=doc_id,
                    document_name=hit.title or doc_id,
                    source="academic",
                    extra={"origin": "academic", "url": url, "doi": hit.doi, "title": hit.title},
                ),
            )
        )
    return out


def score_uploaded_against_store(
    filename: str,
    pages: Sequence[LoadedPage],
    kb,
    settings: Settings | None = None,
    top_k: int = 8,
    max_chunks: int = 48,
) -> "PaperPlagiarismCheck":
    """Embed the upload and compare it to papers already in the vector index.

    Uses the current knowledge-base embedder and vector store (Pinecone when
    configured). The upload is not ingested.
    """

    from ..models.query import PaperPlagiarismCheck, PlagiarismSourceShare

    settings = settings or get_settings()
    texts = _index_query_texts(pages, settings, max_chunks=max_chunks)
    n_words = sum(len(tokenize(text)) for text in texts)
    empty_index = kb.vectors.count() == 0
    if not texts or empty_index:
        return PaperPlagiarismCheck(
            filename=filename,
            n_words=n_words,
            n_pages=len(pages),
            similarity=0.0,
            originality=1.0,
            risk="low",
            library_empty=empty_index or kb.is_empty(),
        )

    vectors = kb.embedder.encode(texts)
    per_chunk_best: list[float] = []
    best_by_doc: dict[str, float] = {}
    names: dict[str, str] = {}
    for vector in vectors:
        hits = kb.vectors.search(vector, top_k=top_k)
        if not hits:
            per_chunk_best.append(0.0)
            continue
        chunk_best = 0.0
        for hit in hits:
            score = _clamp_similarity(hit.score)
            chunk_best = max(chunk_best, score)
            doc_id = str(hit.metadata.get("document_id") or hit.chunk_id)
            if score > best_by_doc.get(doc_id, 0.0):
                best_by_doc[doc_id] = score
                names[doc_id] = _document_label(kb, doc_id, hit.metadata)
        per_chunk_best.append(chunk_best)

    similarity = round(sum(per_chunk_best) / max(1, len(per_chunk_best)), 4)
    originality = round(max(0.0, 1.0 - similarity), 4)
    sources = [
        PlagiarismSourceShare(
            document_id=doc_id,
            document_name=names.get(doc_id, doc_id),
            origin="library",
            share=round(score, 4),
        )
        for doc_id, score in sorted(best_by_doc.items(), key=lambda item: item[1], reverse=True)
        if score > 0
    ]
    if similarity >= 0.70:
        risk = "high"
    elif similarity >= 0.40:
        risk = "medium"
    else:
        risk = "low"
    return PaperPlagiarismCheck(
        filename=filename,
        n_words=n_words,
        n_pages=len(pages),
        similarity=similarity,
        originality=originality,
        risk=risk,
        library_empty=False,
        sources=sources,
    )


def _clamp_similarity(score: float) -> float:
    return float(max(0.0, min(1.0, score)))


def _document_label(kb, document_id: str, metadata: dict) -> str:
    record = kb.store.get_document(document_id)
    if record is not None:
        return record.title or record.name or document_id
    return str(metadata.get("document_name") or document_id)


def _index_query_texts(
    pages: Sequence[LoadedPage],
    settings: Settings,
    max_chunks: int = 48,
) -> list[str]:
    from ..services.ingestion.chunker import chunk_page
    from ..services.ingestion.cleaner import clean_page

    texts: list[str] = []
    index = 0
    for page in pages:
        cleaned = clean_page(getattr(page, "text", "") or "")
        if not cleaned.strip():
            continue
        pieces = chunk_page(
            text=cleaned,
            page=page.page,
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            min_chunk_chars=min(40, settings.min_chunk_chars),
            start_index=index,
        )
        if pieces:
            texts.extend(piece.text for piece in pieces)
            index += len(pieces)
        else:
            texts.append(cleaned)
    if len(texts) > max_chunks:
        step = len(texts) / max_chunks
        texts = [texts[int(i * step)] for i in range(max_chunks)]
    return texts
