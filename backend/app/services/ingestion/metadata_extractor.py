"""Extract paper title, authors, and year from document text."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Sequence

_PREAMBLE_PATTERNS = [
    re.compile(r"^arxiv:\s*[\d\.]+(?:v\d+)?(?:\s*\[[^\]]+\])?", re.IGNORECASE),
    re.compile(r"^under\s+review\s+(?:as\s+a\s+conference\s+paper|at\s+)", re.IGNORECASE),
    re.compile(r"^proceedings\s+of\s+", re.IGNORECASE),
    re.compile(r"^(?:ieee|acm|springer|elsevier|nature|science)\s+", re.IGNORECASE),
    re.compile(r"^(?:vol\.|volume|issue|no\.)\s+\d+", re.IGNORECASE),
    re.compile(r"^doi:\s*10\.\d+", re.IGNORECASE),
    re.compile(r"^https?://", re.IGNORECASE),
    re.compile(r"^published\s+(?:in|as|by)", re.IGNORECASE),
]

_AFFILIATION_KEYWORDS = {
    "university", "department", "dept", "institute", "institution",
    "laboratory", "laboratories", "lab", "labs", "college", "school",
    "faculty", "center", "centre", "corporation", "corp", "inc",
    "research", "technologies", "microsoft", "google", "meta", "facebook",
    "amazon", "apple", "ibm", "deepmind", "openai", "telecom",
    "campus", "hospital", "division", "foundation", "academy",
    "germany", "france", "china", "japan", "korea", "india", "usa", "uk",
    "california", "massachusetts", "london", "beijing", "zurich",
}

_EMAIL_RE = re.compile(r"[\w\.-]+@[\w\.-]+\.\w+")
_YEAR_RE = re.compile(r"\b(19[7-9]\d|20[0-2]\d)\b")
_CLEAN_SUPER_RE = re.compile(r"[\*\†\‡\§\1-9]")


def is_generic_title(title: str | None, filename: str = "") -> bool:
    if not title or not title.strip():
        return True
    t = title.strip().lower()
    fname = Path(filename).name.lower()
    stem = Path(filename).stem.lower()
    if t in {fname, stem, f"{stem}.pdf"}:
        return True
    if re.match(r"^(pap|paper|doc|document|file|untitled|download|scan)[\d_-]*(\.pdf)?$", t):
        return True
    if re.match(r"^\d{4}\.\d{4,5}(v\d+)?(\.pdf)?$", t):
        return True
    if t.startswith("microsoft word - ") or t in {"untitled", "unknown", "document", "draft"}:
        return True
    return False


def is_generic_authors(authors: Sequence[str] | None) -> bool:
    if not authors:
        return True
    cleaned = [a.strip().lower() for a in authors if a and a.strip()]
    if not cleaned:
        return True
    generic = {
        "anonymous", "anonymous submission", "unknown", "unknown author",
        "author unavailable", "author", "authors", "draft", "none"
    }
    return all(a in generic or "anonymous" in a for a in cleaned)


def extract_metadata_from_text(
    text: str,
    filename: str = "",
) -> tuple[str | None, list[str], int | None]:
    """Heuristic extraction of scientific paper title, authors, and publication year from page 1."""
    if not text or not text.strip():
        return None, [], None

    lines = [ln.strip() for ln in text[:4000].splitlines() if ln.strip()]
    if not lines:
        return None, [], None

    # Step 1: Detect year from early text
    extracted_year = None
    for line in lines[:25]:
        m = _YEAR_RE.search(line)
        if m:
            yr = int(m.group(1))
            if 1970 <= yr <= 2030:
                extracted_year = yr
                break

    # Step 2: Find boundary for Abstract / Introduction
    abstract_idx = len(lines)
    for i, line in enumerate(lines[:35]):
        clean_ln = line.strip().lower()
        if clean_ln in {"abstract", "abstract.", "1 abstract"} or clean_ln.startswith("abstract -") or clean_ln.startswith("abstract—") or clean_ln.startswith("abstract:"):
            abstract_idx = i
            break
        if clean_ln.startswith("1. introduction") or clean_ln == "introduction":
            abstract_idx = min(abstract_idx, i)
            break

    header_lines = lines[:abstract_idx]

    # Step 3: Skip preamble lines (arXiv header, conference banners, etc.)
    start_idx = 0
    while start_idx < len(header_lines) and start_idx < 6:
        ln = header_lines[start_idx]
        if any(pat.search(ln) for pat in _PREAMBLE_PATTERNS) or _EMAIL_RE.search(ln):
            start_idx += 1
            continue
        break

    if start_idx >= len(header_lines):
        return None, [], extracted_year

    # Step 4: Extract Title
    # The title is typically 1-3 lines right after preamble
    title_parts: list[str] = []
    author_start_idx = start_idx + 1

    for idx in range(start_idx, min(start_idx + 4, len(header_lines))):
        ln = header_lines[idx]
        words = ln.split()
        # If line contains an email, affiliation keyword, or starts with author-like list: stop
        lower_ln = ln.lower()
        has_affil = any(k in lower_ln.split() for k in _AFFILIATION_KEYWORDS)
        has_email = bool(_EMAIL_RE.search(ln))
        if idx > start_idx and (has_affil or has_email or "@" in ln or "{" in ln):
            author_start_idx = idx
            break
        title_parts.append(ln)
        author_start_idx = idx + 1
        # If the title line ends with a question mark or colon or has 6+ words and next line is distinct, could be complete
        if len(" ".join(title_parts).split()) >= 8 and not ln.endswith(("-", "and", "of", "for", "in", "with", "the")):
            # Check if next line looks like authors
            if idx + 1 < len(header_lines):
                next_ln = header_lines[idx + 1].lower()
                if any(k in next_ln for k in _AFFILIATION_KEYWORDS) or "@" in next_ln or "," in next_ln:
                    break

    candidate_title = " ".join(title_parts).strip()
    candidate_title = re.sub(r"\s+", " ", candidate_title)
    # Clean trailing or leading punctuation
    candidate_title = candidate_title.strip(" -:*#")

    # Step 5: Extract Authors from remaining lines before Abstract
    authors_found: list[str] = []
    author_lines = header_lines[author_start_idx:]

    for ln in author_lines:
        lower_ln = ln.lower()
        if _EMAIL_RE.search(ln) or "@" in ln:
            continue
        # Skip pure affiliation lines
        words = set(re.findall(r"\b[a-z]{3,}\b", lower_ln))
        if words and words.intersection(_AFFILIATION_KEYWORDS):
            continue
        if any(p in lower_ln for p in ["department", "university", "institute", "laboratory", "school of", "center for"]):
            continue

        # Clean footnote symbols and numbers from author names
        cleaned_line = _CLEAN_SUPER_RE.sub("", ln)
        # Split on commas or " and "
        parts = re.split(r",|\band\b|&|;", cleaned_line, flags=re.IGNORECASE)
        for part in parts:
            name = part.strip()
            # Valid author names are usually 2-4 tokens, mostly letters
            tokens = name.split()
            if 2 <= len(tokens) <= 5 and all(t[0].isupper() for t in tokens if t.isalpha()):
                cleaned_name = " ".join(tokens)
                if cleaned_name.lower() not in {"anonymous submission", "author names omitted", "abstract"}:
                    if cleaned_name not in authors_found:
                        authors_found.append(cleaned_name)

    title_res = candidate_title if (candidate_title and len(candidate_title.split()) >= 2 and not is_generic_title(candidate_title, filename)) else None
    return title_res, authors_found[:12], extracted_year
