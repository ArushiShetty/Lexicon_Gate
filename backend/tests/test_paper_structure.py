"""Per-paper structured extraction at ingest."""

from __future__ import annotations

from app.models.documents import PAPER_STRUCTURE_FIELDS, Chunk, ChunkMetadata, Document
from app.services.extraction.paper_structure import extract_paper_structure
from app.verification.nli_verifier import reset_nli_runtime, set_nli_score_override


def _chunk(text: str, chunk_id: str = "c1") -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        text=text,
        metadata=ChunkMetadata(document_id="doc1", document_name="paper.md"),
    )


def test_verbatim_extract_is_not_low_confidence(llm):
    reset_nli_runtime()
    set_nli_score_override(lambda claim, evidence: ("unsupported", 0.9))
    document = Document(document_id="doc1", name="paper.md", title="Dropout paper")
    chunks = [
        _chunk(
            "This paper we propose dropout as a regularization method. "
            "A limitation is that training time increases.",
            "c1",
        )
    ]
    record = extract_paper_structure(document, chunks, llm=llm)
    assert record.fields["objective"].value
    assert record.fields["objective"].low_confidence is False
    reset_nli_runtime()


def test_unrelated_claim_is_flagged_not_dropped():
    from app.config import get_settings
    from app.services.extraction.paper_structure import _validate_field

    reset_nli_runtime()
    set_nli_score_override(lambda claim, evidence: ("unsupported", 0.95))
    field = _validate_field(
        "The moon is made of green cheese according to this study.",
        [_chunk("Dropout reduces overfitting during training of neural networks.")],
        get_settings(),
    )
    assert field.value.startswith("The moon")
    assert field.low_confidence is True
    reset_nli_runtime()


def test_extract_does_not_drop_empty_fields(llm):
    reset_nli_runtime()
    document = Document(document_id="doc1", name="note.md", title="Note")
    chunks = [_chunk("The cat sat on the mat and then took a nap in the sun.")]
    record = extract_paper_structure(document, chunks, llm=llm)
    for key in PAPER_STRUCTURE_FIELDS:
        assert key in record.fields
    assert record.fields["dataset"].value == ""
    assert record.fields["dataset"].low_confidence is True
    reset_nli_runtime()
