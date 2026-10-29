"""Per-paper structured extraction (ingestion time, once per document).

Fields: objective, method, dataset, metric, result, limitation.
Each field is validated lightly against that paper's chunks (NLI when available)
and flagged ``low_confidence`` instead of being dropped.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Sequence

from ...config import Settings, get_settings
from ...models.documents import (
    PAPER_STRUCTURE_FIELDS,
    Chunk,
    Document,
    PaperStructure,
    StructuredField,
)
from ...text_utils import content_tokens, idf_weighted_containment, split_sentences, stem, stem_set, truncate
from ..llm.base import LLMProvider, LLMRequest, LLMTask
from ..llm.registry import get_llm_provider

logger = logging.getLogger(__name__)

_FIELD_CUES: dict[str, tuple[str, ...]] = {
    "objective": (
        "this paper",
        "we propose",
        "we present",
        "we introduce",
        "aim",
        "objective",
        "goal of this",
        "in this work",
        "this work",
        "contribution",
        "abstract",
    ),
    "method": (
        "method",
        "approach",
        "architecture",
        "algorithm",
        "we use",
        "we train",
        "pipeline",
        "framework",
    ),
    "dataset": (
        "dataset",
        "corpus",
        "benchmark",
        "trained on",
        "evaluated on",
        "imagenet",
        "mnist",
        "cifar",
    ),
    "metric": (
        "accuracy",
        "f1",
        "bleu",
        "rouge",
        "rmse",
        "auc",
        "precision",
        "recall",
        "metric",
    ),
    "result": (
        "achieves",
        "outperforms",
        "improves",
        "results show",
        "we obtain",
        "state-of-the-art",
        "sota",
    ),
    "limitation": (
        "limitation",
        "future work",
        "however",
        "drawback",
        "cannot",
        "fails to",
        "restricted to",
    ),
}

_SYSTEM = """You extract a structured summary of one scientific paper.
Return JSON only with exactly these keys:
{"objective":"","method":"","dataset":"","metric":"","result":"","limitation":""}
Each value is one short sentence copied or tightly paraphrased from the passages.
If a field is not present in the passages, use an empty string. Do not invent facts."""


def extract_paper_structure(
    document: Document,
    chunks: Sequence[Chunk],
    llm: LLMProvider | None = None,
    settings: Settings | None = None,
) -> PaperStructure:
    settings = settings or get_settings()
    llm = llm or get_llm_provider()
    raw = _llm_fields(document, chunks, llm)
    fallback = _heuristic_fields(chunks)
    merged: dict[str, str] = {}
    for key in PAPER_STRUCTURE_FIELDS:
        value = str(raw.get(key) or fallback.get(key) or "").strip()
        merged[key] = truncate(value, 280)

    fields: dict[str, StructuredField] = {}
    for key, value in merged.items():
        fields[key] = _validate_field(value, chunks, settings)

    return PaperStructure(
        document_id=document.document_id,
        document_name=document.name,
        title=document.title or document.name,
        fields=fields,
        extractor=getattr(llm, "name", "unknown"),
    )


def extract_and_store(store, document: Document, chunks: Sequence[Chunk]) -> PaperStructure | None:
    try:
        record = extract_paper_structure(document, chunks)
        store.upsert_extraction(record)
        return record
    except Exception:
        logger.exception("Structured extraction failed for %s", document.document_id)
        return None


def refresh_document(store, document_id: str) -> PaperStructure | None:
    document = store.get_document(document_id)
    if document is None:
        return None
    chunks = store.all_chunks([document_id])
    return extract_and_store(store, document, chunks)


def _is_bib(chunk: Chunk) -> bool:
    if getattr(chunk.metadata, "extra", {}).get("is_bibliography") == "true":
        return True
    from ..ingestion.chunker import is_bibliography_text
    return is_bibliography_text(chunk.text or "", chunk.metadata.section)


def _llm_fields(document: Document, chunks: Sequence[Chunk], llm: LLMProvider) -> dict[str, str]:
    substantive_chunks = [c for c in chunks if not _is_bib(c)] or list(chunks)
    passages = []
    used = 0
    for chunk in substantive_chunks:
        text = (chunk.text or "").strip()
        if not text:
            continue
        if used + len(text) > 8000:
            text = text[: max(0, 8000 - used)]
        passages.append(f"[{chunk.chunk_id}] {text}")
        used += len(text)
        if used >= 8000:
            break
    prompt = (
        f"Paper title: {document.title or document.name}\n\n"
        "Passages:\n"
        + "\n\n".join(passages)
        + "\n\nReturn JSON with keys objective, method, dataset, metric, result, limitation."
    )
    response = llm.generate(
        LLMRequest(
            task=LLMTask.extract_paper_structure,
            prompt=prompt,
            system=_SYSTEM,
            payload={
                "title": document.title or document.name,
                "chunks": [{"chunk_id": c.chunk_id, "text": c.text} for c in substantive_chunks],
            },
        )
    )
    structured = response.structured or _parse_json_object(response.text)
    if not isinstance(structured, dict):
        return {}
    return {key: str(structured.get(key) or "").strip() for key in PAPER_STRUCTURE_FIELDS}


def _heuristic_fields(chunks: Sequence[Chunk]) -> dict[str, str]:
    substantive_chunks = [c for c in chunks if not _is_bib(c)] or list(chunks)
    sentences: list[tuple[str, str]] = []
    for chunk in substantive_chunks:
        for sentence in split_sentences(chunk.text or "", min_chars=20) or [chunk.text or ""]:
            text = sentence.strip()
            if text:
                sentences.append((chunk.chunk_id, text))
    out: dict[str, str] = {key: "" for key in PAPER_STRUCTURE_FIELDS}
    for key, cues in _FIELD_CUES.items():
        best = ""
        best_score = 0.0
        for _cid, sentence in sentences:
            lowered = sentence.lower()
            score = sum(1.0 for cue in cues if cue in lowered)
            if score > best_score:
                best_score = score
                best = sentence
        if best_score > 0:
            out[key] = truncate(best, 280)
    if not out["objective"]:
        for _cid, sentence in sentences[:8]:
            if len(sentence) >= 40:
                out["objective"] = truncate(sentence, 280)
                break
    if not out["method"]:
        for _cid, sentence in sentences:
            if re.search(r"\b(?:we use|we train|we propose|we design|architecture|framework|model|algorithm|pipeline)\b", sentence, re.I):
                out["method"] = truncate(sentence, 280)
                break
    if not out["dataset"]:
        for _cid, sentence in sentences:
            if re.search(r"\b(?:trained on|evaluated on|benchmark|dataset|corpus|mnist|cifar|imagenet|squad|glue)\b", sentence, re.I):
                out["dataset"] = truncate(sentence, 280)
                break
    if not out["result"]:
        for _cid, sentence in sentences:
            if re.search(r"\b(?:achieve|obtain|reach|yield|show|improve|outperform|accuracy|f1|score|bleu|%)\b", sentence, re.I) and re.search(r"\d+", sentence):
                out["result"] = truncate(sentence, 280)
                break
    return out


def _validate_field(
    value: str, chunks: Sequence[Chunk], settings: Settings
) -> StructuredField:
    if not value.strip():
        return StructuredField(value="", low_confidence=True, nli_label="unsupported")

    compact = " ".join(value.split())
    for chunk in chunks:
        hay = " ".join((chunk.text or "").split())
        if compact and compact in hay:
            return StructuredField(
                value=value,
                low_confidence=False,
                nli_label="supported",
                nli_confidence=1.0,
                source_chunk_id=chunk.chunk_id,
            )

    sample = [c for c in chunks if compact[:40] in " ".join((c.text or "").split())] or list(chunks)[:12]
    nli_label: str | None = None
    nli_conf = 0.0
    source_id: str | None = None
    used_nli = False
    try:
        from ...verification.nli_verifier import nli_is_available, verify_claim

        if nli_is_available():
            used_nli = True
            for chunk in sample:
                verdict = verify_claim(value, chunk.text or "")
                conf = float(verdict.get("confidence") or 0.0)
                label = str(verdict.get("label") or "unsupported")
                rank = {"supported": 2, "contradicted": 1, "unsupported": 0}.get(label, 0)
                best_rank = {"supported": 2, "contradicted": 1, "unsupported": 0}.get(nli_label or "", -1)
                if rank > best_rank or (rank == best_rank and conf > nli_conf):
                    nli_label, nli_conf, source_id = label, conf, chunk.chunk_id
    except Exception:
        logger.debug("NLI unavailable for paper-field validation", exc_info=True)
        used_nli = False

    if used_nli and nli_label:
        low = nli_label != "supported" or nli_conf < settings.claim_support_threshold
        return StructuredField(
            value=value,
            low_confidence=low,
            nli_label=nli_label,
            nli_confidence=round(nli_conf, 4),
            source_chunk_id=source_id,
        )

    best_overlap = 0.0
    best_id: str | None = None
    claim_tokens = [stem(tok) for tok in content_tokens(value)]
    for chunk in sample:
        stems = stem_set(chunk.text or "")
        overlap = idf_weighted_containment(claim_tokens, stems, {})
        if overlap > best_overlap:
            best_overlap = overlap
            best_id = chunk.chunk_id
    supported = best_overlap >= settings.claim_partial_threshold
    return StructuredField(
        value=value,
        low_confidence=not supported,
        nli_label="supported" if supported else "unsupported",
        nli_confidence=round(best_overlap, 4),
        source_chunk_id=best_id,
    )


def _parse_json_object(text: str) -> dict:
    match = re.search(r"\{.*\}", text or "", re.S)
    if not match:
        return {}
    try:
        parsed = json.loads(match.group(0))
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}
