"""Document loaders.

Every loader returns a list of ``(page_number, text)`` pairs so page-level
citation provenance survives ingestion.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

TEXT_SUFFIXES = {".txt", ".md", ".markdown", ".rst", ".csv", ".json", ".log"}
PDF_SUFFIXES = {".pdf"}
HTML_SUFFIXES = {".html", ".htm"}
DOCX_SUFFIXES = {".docx"}

SUPPORTED_SUFFIXES = TEXT_SUFFIXES | PDF_SUFFIXES | HTML_SUFFIXES | DOCX_SUFFIXES


@dataclass(slots=True)
class LoadedPage:
    page: int
    text: str


class UnsupportedDocumentError(ValueError):
    pass


@dataclass(slots=True)
class Bibliography:
    title: str = ""
    authors: list[str] = field(default_factory=list)
    year: int | None = None
    venue: str | None = None
    doi: str | None = None


def load_document(path: Path | str) -> list[LoadedPage]:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in PDF_SUFFIXES:
        return _load_pdf(path)
    if suffix in DOCX_SUFFIXES:
        return _load_docx_bytes(path.read_bytes())
    if suffix in HTML_SUFFIXES:
        return _load_html(path)
    if suffix in TEXT_SUFFIXES or suffix == "":
        return _load_text(path)
    raise UnsupportedDocumentError(
        f"Unsupported file type {suffix!r}. Supported: {sorted(SUPPORTED_SUFFIXES)}"
    )


def _pdf_reader():
    try:
        from pypdf import PdfReader  # noqa: PLC0415
    except ImportError as exc:
        raise UnsupportedDocumentError(
            "PDF support requires the pypdf package (pip install pypdf)."
        ) from exc
    return PdfReader


def load_bytes(name: str, data: bytes) -> list[LoadedPage]:
    """Load from raw bytes by writing to a temporary file when needed."""

    suffix = Path(name).suffix.lower()
    if suffix in PDF_SUFFIXES:
        import io

        reader = _pdf_reader()(io.BytesIO(data))
        return _pages_from_reader(reader)
    if suffix in DOCX_SUFFIXES:
        return _load_docx_bytes(data)
    text = data.decode("utf-8", errors="replace")
    if suffix in HTML_SUFFIXES:
        text = _strip_html(text)
    return _paginate_text(text)


def _load_docx_bytes(data: bytes) -> list[LoadedPage]:
    import io

    try:
        from docx import Document  # noqa: PLC0415
    except ImportError as exc:
        raise UnsupportedDocumentError(
            "Word (.docx) support requires the python-docx package (pip install python-docx)."
        ) from exc
    document = Document(io.BytesIO(data))
    text = "\n\n".join(para.text for para in document.paragraphs if para.text.strip())
    if not text.strip():
        return [LoadedPage(page=1, text="")]
    return _paginate_text(text)


def _load_pdf(path: Path) -> list[LoadedPage]:
    reader = _pdf_reader()(str(path))
    return _pages_from_reader(reader)


def _pages_from_reader(reader) -> list[LoadedPage]:
    pages: list[LoadedPage] = []
    for index, page in enumerate(reader.pages, start=1):
        try:
            text = page.extract_text() or ""
        except Exception as exc:  # pragma: no cover - malformed PDFs
            logger.warning("Failed to extract page %s: %s", index, exc)
            text = ""
        if text.strip():
            pages.append(LoadedPage(page=index, text=text))
    return pages


def _load_text(path: Path) -> list[LoadedPage]:
    text = path.read_text(encoding="utf-8", errors="replace")
    return _paginate_text(text)


def _load_html(path: Path) -> list[LoadedPage]:
    return _paginate_text(_strip_html(path.read_text(encoding="utf-8", errors="replace")))


def _strip_html(html: str) -> str:
    import re

    html = re.sub(r"<script.*?</script>|<style.*?</style>", " ", html, flags=re.S | re.I)
    html = re.sub(r"<br\s*/?>|</p>|</div>|</h\d>", "\n", html, flags=re.I)
    html = re.sub(r"<[^>]+>", " ", html)
    return html


PAGE_BREAK_MARKERS = ("\f", "\n--- page", "\n[page")
CHARS_PER_SYNTHETIC_PAGE = 3000


def _paginate_text(text: str) -> list[LoadedPage]:
    """Split plain text into pages.

    Explicit form-feed / ``--- page N ---`` markers win. Otherwise text is
    divided into fixed-size synthetic pages at paragraph boundaries so that
    citations still carry a stable, meaningful page number.
    """

    import re

    if "\f" in text:
        parts = [p for p in text.split("\f")]
    else:
        marker = re.split(r"\n\s*(?:---\s*)?\[?page\s+\d+\]?\s*(?:---)?\s*\n", text, flags=re.I)
        parts = marker if len(marker) > 1 else _chunk_paragraphs(text)

    pages: list[LoadedPage] = []
    for index, part in enumerate(parts, start=1):
        if part.strip():
            pages.append(LoadedPage(page=index, text=part))
    return pages or [LoadedPage(page=1, text=text)]


def _chunk_paragraphs(text: str) -> list[str]:
    paragraphs = text.split("\n\n")
    pages: list[str] = []
    buffer: list[str] = []
    size = 0
    for paragraph in paragraphs:
        buffer.append(paragraph)
        size += len(paragraph) + 2
        if size >= CHARS_PER_SYNTHETIC_PAGE:
            pages.append("\n\n".join(buffer))
            buffer = []
            size = 0
    if buffer:
        pages.append("\n\n".join(buffer))
    return pages or [text]


def extract_bibliography(name: str, data: bytes | None = None, path: Path | str | None = None) -> Bibliography:
    """PDF document info when available; otherwise a title from the filename."""

    from ..citations import parse_authors, title_from_filename, year_from_pdf_date

    suffix = Path(name).suffix.lower()
    biblio = Bibliography(title=title_from_filename(name))
    if suffix not in PDF_SUFFIXES:
        return biblio
    try:
        PdfReader = _pdf_reader()

        if data is not None:
            import io

            reader = PdfReader(io.BytesIO(data))
        elif path is not None:
            reader = PdfReader(str(path))
        else:
            return biblio
        info = reader.metadata
        if info is None:
            return biblio
        title = getattr(info, "title", None)
        if title and str(title).strip():
            biblio.title = str(title).strip()
        authors = parse_authors(getattr(info, "author", None))
        if authors:
            biblio.authors = authors
        biblio.year = year_from_pdf_date(getattr(info, "creation_date", None))
        subject = getattr(info, "subject", None)
        if subject and str(subject).strip():
            biblio.venue = str(subject).strip()
    except Exception as exc:  # pragma: no cover - malformed PDFs
        logger.warning("Could not read PDF bibliography from %s: %s", name, exc)
    return biblio
