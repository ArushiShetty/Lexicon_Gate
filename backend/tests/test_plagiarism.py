"""Local originality / verbatim-overlap checks."""

from __future__ import annotations

from app.models.documents import Chunk, ChunkMetadata
from app.models.query import EvidenceItem
from app.verification.plagiarism import check_plagiarism


LONG_SOURCE = (
    "Dropout randomly disables units in a neural network during training so that "
    "hidden units cannot co-adapt and the model generalises better on unseen data."
)


def _chunk(doc: str, text: str) -> Chunk:
    return Chunk(
        chunk_id=f"{doc}::c0",
        text=text,
        metadata=ChunkMetadata(document_id=doc, document_name=doc, page=2),
    )


def _evidence(doc: str, text: str) -> EvidenceItem:
    return EvidenceItem(
        citation_id=1,
        chunk_id=f"{doc}::c0",
        document_id=doc,
        document_name=doc,
        page=2,
        text=text,
        title=doc,
    )


def test_paraphrase_is_high_originality(settings):
    answer = (
        "Regularisation helps by reducing the chance that neurons rely on each other, "
        "which often improves test accuracy in deep networks."
    )
    report = check_plagiarism(answer, [_evidence("handbook", LONG_SOURCE)], settings=settings)
    assert report.risk == "low"
    assert report.originality >= 0.85
    assert report.flagged_sentences == []


def test_uncited_library_copy_is_high_risk(settings):
    report = check_plagiarism(
        LONG_SOURCE,
        evidence=[],
        library_chunks=[_chunk("handbook", LONG_SOURCE)],
        settings=settings,
    )
    assert report.risk == "high"
    assert report.originality < 0.5
    assert report.matches
    assert report.matches[0].cited is False
    assert report.flagged_sentences
    hints = report.flagged_sentences[0].corrections
    assert any("Reword" in line for line in hints)
    assert any("handbook" in line.lower() for line in hints)


def test_cited_copy_is_not_uncited_risk(settings):
    report = check_plagiarism(
        LONG_SOURCE,
        evidence=[_evidence("handbook", LONG_SOURCE)],
        library_chunks=[_chunk("handbook", LONG_SOURCE)],
        settings=settings,
    )
    assert all(match.cited for match in report.matches)
    assert report.risk in {"low", "medium"}
    assert report.risk != "high"
    assert report.flagged_sentences


def test_uploaded_paper_spans_and_sections(settings, monkeypatch):
    from app.services.ingestion.loaders import LoadedPage
    from app.verification.plagiarism import score_uploaded_paper

    monkeypatch.setattr("app.verification.plagiarism.search_papers", lambda *a, **k: ([], "none"))
    text = (
        "# Methods\n\n"
        f"{LONG_SOURCE}\n\n"
        "# Notes\n\n"
        "This closing remark is my own wording about homework deadlines.\n"
    )
    report = score_uploaded_paper(
        "draft.md",
        [LoadedPage(page=1, text=text)],
        [_chunk("handbook", LONG_SOURCE)],
        settings=settings,
    )
    assert report.spans
    assert any(span.kind == "verbatim" for span in report.spans)
    assert any(span.kind == "original" for span in report.spans)
    assert report.sections
    titles = [row.title.lower() for row in report.sections]
    assert any("method" in title for title in titles)


def test_uploaded_pdf_is_scored_against_vector_index(kb, settings):
    from app.services.ingestion.loaders import LoadedPage
    from app.verification.plagiarism import score_uploaded_against_store

    kb.ingest_pages(
        "handbook.md",
        [LoadedPage(page=1, text=f"Handbook\n\n{LONG_SOURCE}")],
        title="Handbook",
    )
    report = score_uploaded_against_store(
        "draft.pdf",
        [LoadedPage(page=1, text=f"Class notes. {LONG_SOURCE}")],
        kb,
        settings=settings,
    )
    assert report.library_empty is False
    assert report.similarity > 0
    assert report.originality < 1
    assert report.sources
    assert "handbook" in report.sources[0].document_name.lower()


def test_uploaded_paper_skips_same_title_in_index(settings, monkeypatch):
    from app.models.papers import PaperHit
    from app.services.ingestion.loaders import LoadedPage
    from app.verification.plagiarism import score_uploaded_paper

    def fake_search(query, limit=10, **kwargs):
        return (
            [
                PaperHit(
                    paper_id="self-1",
                    title="Dropout randomly disables units in a neural network",
                    abstract=LONG_SOURCE,
                    source="semantic_scholar",
                )
            ],
            "semantic_scholar",
        )

    monkeypatch.setattr("app.verification.plagiarism.search_papers", fake_search)
    monkeypatch.setattr("app.verification.plagiarism._fetch_oa_pages", lambda *a, **k: [])
    text = (
        "Dropout randomly disables units in a neural network\n\n"
        f"{LONG_SOURCE}\n"
    )
    report = score_uploaded_paper(
        "dropout.md",
        [LoadedPage(page=1, text=text)],
        [],
        settings=settings,
    )
    assert report.self_matches_skipped >= 1
    assert report.similarity == 0
    assert report.sources == []


def test_uploaded_paper_uses_oa_pdf_body(settings, monkeypatch):
    from app.models.papers import PaperHit
    from app.services.ingestion.loaders import LoadedPage
    from app.verification.plagiarism import score_uploaded_paper

    def fake_search(query, limit=10, **kwargs):
        return (
            [
                PaperHit(
                    paper_id="other-1",
                    title="A different study of regularisation tricks",
                    abstract="Short abstract without the copied paragraph.",
                    source="semantic_scholar",
                    pdf_url="https://example.org/paper.pdf",
                )
            ],
            "semantic_scholar",
        )

    monkeypatch.setattr("app.verification.plagiarism.search_papers", fake_search)
    monkeypatch.setattr(
        "app.verification.plagiarism._fetch_oa_pages",
        lambda *a, **k: [LoadedPage(page=3, text=LONG_SOURCE)],
    )
    copied = (
        "Notes on another topic in vision.\n\n"
        f"{LONG_SOURCE}\n"
    )
    report = score_uploaded_paper(
        "notes.md",
        [LoadedPage(page=1, text=copied)],
        [],
        settings=settings,
    )
    assert report.oa_full_texts == 1
    assert report.similarity > 0
    assert report.sources
    assert report.sources[0].origin == "academic"


def test_references_are_not_scored(settings, monkeypatch):
    from app.services.ingestion.loaders import LoadedPage
    from app.verification.plagiarism import score_uploaded_paper

    monkeypatch.setattr("app.verification.plagiarism.search_papers", lambda *a, **k: ([], "none"))
    body = "\n".join(
        [
            "Original remarks about soil chemistry and garden pH for homework.",
            "Further original sentences about watering schedules and compost bins.",
            "Still more original wording so the references heading is late enough.",
            "Another original paragraph about sunlight hours in winter gardens.",
            "Keep adding original text so bibliography sits in the last half.",
            "This section stays unique and should not match the handbook copy.",
            "Closing original notes about mulch layers and frost dates.",
            "Final original sentence before the bibliography heading appears.",
            "References",
            LONG_SOURCE,
        ]
    )
    report = score_uploaded_paper(
        "essay.md",
        [LoadedPage(page=1, text=body)],
        [_chunk("handbook", LONG_SOURCE)],
        settings=settings,
    )
    assert report.similarity == 0
    assert any("references" in flag.lower() or "bibliography" in flag.lower() for flag in report.flags)
