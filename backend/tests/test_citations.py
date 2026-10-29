"""Bibliography extraction and APA / BibTeX formatting."""

from __future__ import annotations

from io import BytesIO

from pypdf import PdfWriter

from app.models.query import EvidenceItem
from app.services.citations import attach_cite_strings, format_apa, format_bibtex, title_from_filename
from app.services.ingestion.loaders import extract_bibliography
from app.services.store.document_store import DocumentStore
from app.services.ingestion.loaders import LoadedPage


def test_title_from_filename():
    assert title_from_filename("Neural_Regularization_Handbook.md") == "Neural Regularization Handbook"


def test_filename_fallback_bibliography():
    biblio = extract_bibliography("dropout-notes.txt")
    assert biblio.title == "dropout notes"
    assert biblio.authors == []
    assert biblio.year is None


def test_pdf_metadata_bibliography():
    buffer = BytesIO()
    writer = PdfWriter()
    writer.add_blank_page(width=72, height=72)
    writer.add_metadata(
        {
            "/Title": "Dropout Paper",
            "/Author": "Jane Smith and John Doe",
            "/CreationDate": "D:20200101000000",
        }
    )
    writer.write(buffer)
    biblio = extract_bibliography("ignored.pdf", data=buffer.getvalue())
    assert biblio.title == "Dropout Paper"
    assert biblio.authors == ["Jane Smith", "John Doe"]
    assert biblio.year == 2020


def test_apa_missing_year_and_authors():
    item = EvidenceItem(
        citation_id=1,
        chunk_id="c1",
        document_id="d1",
        document_name="notes.md",
        title="A study of dropout",
        page=3,
        text="x",
    )
    apa = format_apa(item)
    assert "Unknown (n.d.). A study of dropout." in apa
    assert "p. 3." in apa
    bib = format_bibtex(item)
    assert "@article{" in bib
    assert "year = {n.d.}" in bib


def test_apa_authors_and_attach():
    item = EvidenceItem(
        citation_id=1,
        chunk_id="c1",
        document_id="d1",
        document_name="paper.pdf",
        title="Regularization",
        authors=["Ada Lovelace", "Alan Turing"],
        year=2021,
        venue="J. ML",
        page=12,
        text="x",
        doi="10.1/xyz",
    )
    attach_cite_strings(item)
    assert "Lovelace, A., & Turing, A. (2021)." in item.apa
    assert "https://doi.org/10.1/xyz" in item.apa
    assert "author = {Ada Lovelace and Alan Turing}" in item.bibtex


def test_patch_bibliography(settings):
    store = DocumentStore(settings)
    doc, _chunks = store.add_document(
        name="Guide.md",
        pages=[LoadedPage(page=1, text="# Dropout\n\nDropout reduces overfitting by preventing co-adaptation.")],
        source="test",
    )
    assert doc.title == "Guide"
    updated = store.update_bibliography(
        doc.document_id,
        title="Dropout handbook",
        authors=["Jane Smith"],
        year=2019,
        venue="Notes",
    )
    assert updated is not None
    assert updated.title == "Dropout handbook"
    assert updated.authors == ["Jane Smith"]
    assert updated.year == 2019
    assert updated.venue == "Notes"
