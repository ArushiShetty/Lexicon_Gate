"""Ingestion: loaders, cleaning, chunking, document store."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from app.services.ingestion.chunker import chunk_page, detect_heading, split_into_sections
from app.services.ingestion.cleaner import clean_page, strip_repeated_lines
from app.services.ingestion.loaders import UnsupportedDocumentError, load_bytes, load_document
from app.services.store.document_store import DocumentStore
from app.services.ingestion.loaders import LoadedPage


def test_clean_page_unwraps_hyphenation_and_quotes():
    raw = "regular-\nization is “useful”\n\n\n42"
    cleaned = clean_page(raw)
    assert "regularization" in cleaned
    assert "useful" in cleaned
    assert "42" not in cleaned or "\n42" not in cleaned


def test_strip_repeated_headers():
    pages = [
        "CONFIDENTIAL\nDropout reduces overfitting.\n1",
        "CONFIDENTIAL\nDropout is applied at training time.\n2",
        "CONFIDENTIAL\nBatch normalization is different.\n3",
    ]
    cleaned = strip_repeated_lines(pages, min_ratio=0.6)
    assert all("CONFIDENTIAL" not in page for page in cleaned)
    assert "overfitting" in cleaned[0]


def test_heading_detection_and_sections():
    text = """# Dropout

Dropout reduces overfitting.

## Limitations of dropout

Dropout increases training time.
"""
    assert detect_heading("# Dropout") == "Dropout"
    sections = split_into_sections(text)
    titles = [title for title, _body in sections]
    assert "Dropout" in titles
    assert any(title and "Limitations" in title for title in titles)


def test_chunker_does_not_cross_pages_and_keeps_overlap():
    page_one = chunk_page(
        "Sentence one is about dropout. " * 40,
        page=1,
        chunk_size=180,
        chunk_overlap=40,
        min_chunk_chars=40,
    )
    assert page_one
    assert all(c.page == 1 for c in page_one)
    assert all(len(c.text) >= 40 for c in page_one)


def test_load_text_and_bytes(tmp_path: Path):
    path = tmp_path / "note.md"
    path.write_text("# Title\n\nHello world.\n")
    pages = load_document(path)
    assert pages[0].text
    loaded = load_bytes("note.md", b"# Title\n\nHello world.\n")
    assert loaded[0].text


def test_load_pdf_without_pypdf(monkeypatch):
    monkeypatch.setitem(sys.modules, "pypdf", None)
    with pytest.raises(UnsupportedDocumentError, match="pypdf"):
        load_bytes("paper.pdf", b"%PDF-1.4\n")


def test_load_docx_bytes():
    pytest.importorskip("docx")
    from io import BytesIO

    from docx import Document

    document = Document()
    document.add_paragraph("Dropout reduces overfitting in neural networks.")
    buf = BytesIO()
    document.save(buf)
    pages = load_bytes("notes.docx", buf.getvalue())
    assert pages
    assert "Dropout" in pages[0].text


def test_document_store_roundtrip(settings):
    store = DocumentStore(settings)
    doc, chunks = store.add_document(
        name="Guide.md",
        pages=[LoadedPage(page=1, text="# Dropout\n\nDropout reduces overfitting by preventing co-adaptation.")],
        source="test",
        source_quality=0.9,
    )
    assert doc.n_chunks == len(chunks) >= 1
    assert store.get_chunk(chunks[0].chunk_id) is not None
    assert store.get_document(doc.document_id) is not None
    assert store.delete_document(doc.document_id) is True
    assert store.get_document(doc.document_id) is None
