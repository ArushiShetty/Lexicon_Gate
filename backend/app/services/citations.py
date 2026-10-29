"""APA and BibTeX strings from library bibliography fields.

Missing year becomes n.d. Missing authors become Unknown. Venue and DOI are
omitted when absent. Nothing is invented.
"""

from __future__ import annotations

import re
from typing import Sequence

from ..models.query import EvidenceItem

_SLUG = re.compile(r"[^a-z0-9]+")


def title_from_filename(name: str) -> str:
    stem = name.rsplit("/", 1)[-1]
    if "." in stem:
        stem = stem.rsplit(".", 1)[0]
    cleaned = re.sub(r"[_-]+", " ", stem).strip()
    return cleaned or name


def parse_authors(raw: str | None) -> list[str]:
    if not raw or not str(raw).strip():
        return []
    text = str(raw).strip()
    for sep in (" and ", " AND ", " & ", ";"):
        if sep in text:
            return [part.strip() for part in text.split(sep) if part.strip()]
    return [text]


def year_from_pdf_date(value: object) -> int | None:
    year = getattr(value, "year", None)
    if isinstance(year, int) and 1000 <= year <= 2100:
        return year
    text = str(value or "")
    match = re.search(r"(?:D:)?(1[89]\d{2}|20\d{2})", text)
    if match:
        return int(match.group(1))
    return None


def attach_cite_strings(item: EvidenceItem) -> EvidenceItem:
    item.apa = format_apa(item)
    item.bibtex = format_bibtex(item)
    return item


def format_apa(item: EvidenceItem) -> str:
    authors = _apa_authors(item.authors)
    year = str(item.year) if item.year else "n.d."
    title = (item.title or item.document_name or "Untitled").rstrip(".")
    parts = [f"{authors} ({year}). {title}."]
    if item.venue:
        parts.append(f"{item.venue.rstrip('.')}.")
    if item.page is not None:
        parts.append(f"p. {item.page}.")
    if item.doi:
        doi = item.doi.removeprefix("https://doi.org/").removeprefix("doi:")
        parts.append(f"https://doi.org/{doi}")
    return " ".join(parts)


def format_bibtex(item: EvidenceItem) -> str:
    title = item.title or item.document_name or "Untitled"
    key = _bibtex_key(item.authors, item.year, title)
    fields = [
        f"  author = {{{_bibtex_authors(item.authors)}}}",
        f"  title = {{{title}}}",
        f"  year = {{{item.year if item.year else 'n.d.'}}}",
    ]
    if item.venue:
        fields.append(f"  journal = {{{item.venue}}}")
    if item.page is not None:
        fields.append(f"  pages = {{{item.page}}}")
    if item.doi:
        fields.append(f"  doi = {{{item.doi}}}")
    body = ",\n".join(fields)
    return f"@article{{{key},\n{body}\n}}"


_SPECIAL_NAMES = {
    "anonymous", "anonymous submission", "unknown", "unknown author",
    "author unavailable", "ieee", "acm", "openai", "google", "meta"
}


def _apa_authors(authors: Sequence[str]) -> str:
    names = [_apa_one(name) for name in authors if name.strip()]
    if not names:
        return "Unknown"
    if len(names) == 1:
        return names[0]
    if len(names) == 2:
        return f"{names[0]}, & {names[1]}"
    return f"{', '.join(names[:-1])}, & {names[-1]}"


def _apa_one(name: str) -> str:
    name = name.strip()
    if not name:
        return ""
    if name.lower() in _SPECIAL_NAMES:
        return name
    if "," in name:
        last, rest = name.split(",", 1)
        initials = " ".join(f"{part[0].upper()}." for part in rest.split() if part)
        return f"{last.strip()}, {initials}".rstrip()
    parts = name.split()
    if len(parts) == 1:
        return parts[0]
    last = parts[-1]
    initials = " ".join(f"{part[0].upper()}." for part in parts[:-1] if part)
    return f"{last}, {initials}"


def _bibtex_authors(authors: Sequence[str]) -> str:
    cleaned = [name.strip() for name in authors if name.strip()]
    return " and ".join(cleaned) if cleaned else "Unknown"


def _bibtex_key(authors: Sequence[str], year: int | None, title: str) -> str:
    last = "anon"
    if authors:
        first = authors[0]
        last = first.split(",")[0].split()[-1] if first.strip() else "anon"
    year_part = str(year) if year else "nd"
    words = _SLUG.sub("-", title.lower()).strip("-").split("-")[:2]
    tail = "-".join(w for w in words if w) or "untitled"
    slug = _SLUG.sub("-", f"{last}-{year_part}-{tail}".lower()).strip("-")
    return slug or "untitled"
