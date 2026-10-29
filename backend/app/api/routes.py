"""HTTP API."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from fastapi import APIRouter, File, HTTPException, Query, Request, UploadFile
from pydantic import BaseModel, Field

from ..config import get_settings
from ..evaluation.dataset import benchmark_dataset, load_demo_corpus
from ..evaluation.harness import (
    EvaluationHarness,
    comparison_table,
    headline_table,
    list_runs,
    load_latest_run,
    summarise,
)
from ..models.documents import (
    ClusterResponse,
    Document,
    DocumentBiblioUpdate,
    DocumentListResponse,
    PAPER_STRUCTURE_FIELDS,
    PaperStructure,
    PaperStructureListResponse,
    UploadResponse,
)
from ..services.clustering import cluster_documents
from ..models.papers import PaperImportRequest, PaperImportResponse, PaperSearchResponse
from ..services.papers.import_paper import import_paper
from ..models.evaluation import EvaluateRequest, EvaluationRun
from ..models.query import (
    CompareRequest,
    CompareResponse,
    ComparisonRow,
    PipelineName,
    PipelineResult,
    QueryRequest,
)
from ..pipelines.runner import (
    PipelineRunner,
    enhanced_config,
    no_rag_config,
    self_rag_config,
    traditional_config,
    verify_only_config,
)
from ..services.embeddings.registry import available_embedding_providers, get_embedding_provider
from ..services.llm.registry import available_llm_providers, get_llm_provider
from ..services.store.history import history
from ..services.store.knowledge_base import get_knowledge_base
from ..services.vectorstore.registry import available_vector_stores
from ..services.workspace import ask_chats, ask_queries
from ..models.workspace import AskChatTurn
from .auth import current_user_from_request

logger = logging.getLogger(__name__)
router = APIRouter()

_CONFIGS = {
    PipelineName.traditional: traditional_config,
    PipelineName.self_rag: self_rag_config,
    PipelineName.enhanced: enhanced_config,
    PipelineName.no_rag: no_rag_config,
    PipelineName.rag_verify: verify_only_config,
}


def _runner(pipeline: PipelineName, threshold: float | None = None) -> PipelineRunner:
    return PipelineRunner(
        kb=get_knowledge_base(),
        llm=get_llm_provider(),
        config=_CONFIGS[pipeline](),
        evidence_threshold=threshold,
    )


def _to_row(result: PipelineResult) -> ComparisonRow:
    n_claims = result.confidence.claims_total
    n_verified = result.confidence.claims_verified
    last_gate = result.gate_decisions[-1].evidence_score if result.gate_decisions else 0.0
    unsupported_rate = (
        result.confidence.unsupported_claims / n_claims if n_claims else 0.0
    )
    return ComparisonRow(
        pipeline=result.pipeline,
        pipeline_label=result.pipeline_label,
        answer=result.answer,
        status=result.status,
        abstained=result.abstained,
        confidence=result.confidence.confidence,
        evidence_score=last_gate,
        evidence_coverage=result.confidence.evidence_coverage,
        claims_verified=n_verified,
        claims_total=n_claims,
        claim_support_rate=(n_verified / n_claims) if n_claims else 0.0,
        unsupported_claim_rate=unsupported_rate,
        hallucination_detected=result.hallucination.hallucination_detected,
        hallucination_flags=result.hallucination.flags,
        contradictions_found=len(result.contradictions),
        n_sources=len(result.evidence),
        citations=[item.citation_label for item in result.evidence],
        retrieval_attempts=result.metrics.retrieval_attempts,
        correction_loops=result.metrics.correction_loops,
        latency_ms=result.metrics.latency_ms,
    )


def _pick_winner(rows: list[ComparisonRow]) -> tuple[str, str]:
    """Pick a winner from observable quality signals, not a hard-coded ranking."""

    def score(row: ComparisonRow) -> float:
        hall = 1.0 if row.hallucination_detected else 0.0
        return (
            0.35 * row.claim_support_rate
            + 0.25 * row.confidence
            + 0.20 * row.evidence_score
            + 0.20 * (1.0 - hall)
            - (0.15 if row.abstained and row.evidence_score >= 0.55 else 0.0)
        )

    ranked = sorted(rows, key=score, reverse=True)
    winner = ranked[0]
    reason = (
        f"{winner.pipeline_label} ranked highest on combined claim support "
        f"({winner.claim_support_rate:.0%}), evidence score ({winner.evidence_score:.2f}) "
        f"and confidence ({winner.confidence:.0%})."
    )
    return winner.pipeline.value, reason


# --------------------------------------------------------------------------- health
@router.get("/health", tags=["system"])
def health() -> dict[str, Any]:
    settings = get_settings()
    kb = get_knowledge_base()
    return {
        "status": "ok",
        "app": settings.app_name,
        "llm": get_llm_provider().describe(),
        "embeddings": get_embedding_provider().describe(),
        "vector_store": kb.vectors.name,
        "knowledge_base": kb.stats(),
        "available_llm_providers": available_llm_providers(),
        "available_embedding_providers": available_embedding_providers(),
        "available_vector_stores": available_vector_stores(),
    }


@router.get("/config", tags=["system"])
def public_config() -> dict[str, Any]:
    from ..services.tts import server_tts_available

    settings = get_settings()
    return {
        "evidence_threshold": settings.evidence_threshold,
        "initial_top_k": settings.initial_top_k,
        "max_top_k": settings.max_top_k,
        "max_retrieval_attempts": settings.max_retrieval_attempts,
        "max_correction_loops": settings.max_correction_loops,
        "llm_provider": get_llm_provider().describe(),
        "embedding_provider": get_embedding_provider().describe(),
        "server_tts": server_tts_available(),
        "google_client_id": (settings.google_client_id or "").strip(),
        "supabase_configured": bool(
            (settings.supabase_url or "").strip()
            and (settings.supabase_service_key or "").strip()
        ),
    }


class _TtsRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=8000)


@router.post("/tts", tags=["system"])
def synthesize_speech(payload: _TtsRequest):
    """Return WAV audio for Read aloud. Uses macOS say when available."""

    from fastapi.responses import Response

    from ..services.tts import clean_tts_text, synthesize_wav

    text = clean_tts_text(payload.text)
    if not text:
        raise HTTPException(status_code=400, detail="Nothing to speak.")
    try:
        audio = synthesize_wav(text)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=501, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover - defensive
        logger.exception("TTS failed")
        raise HTTPException(status_code=500, detail="Speech synthesis failed") from exc

    media = "audio/wav" if audio[:4] == b"RIFF" else "audio/aiff"
    return Response(content=audio, media_type=media, headers={"Cache-Control": "no-store"})


# ----------------------------------------------------------------------- documents
MAX_UPLOAD_FILES = 20


@router.post("/documents/upload", response_model=UploadResponse, tags=["documents"])
async def upload_documents(files: list[UploadFile] = File(...)) -> UploadResponse:
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded.")
    if len(files) > MAX_UPLOAD_FILES:
        raise HTTPException(
            status_code=400,
            detail=f"Upload at most {MAX_UPLOAD_FILES} files at once (received {len(files)}).",
        )
    kb = get_knowledge_base()
    documents = []
    warnings: list[str] = []
    total_chunks = 0
    for upload in files:
        data = await upload.read()
        name = Path(upload.filename or "untitled.txt").name
        if not name or name in {".", ".."}:
            name = "untitled.txt"
        if not data:
            warnings.append(f"{name}: empty file skipped")
            continue
        try:
            document, chunks = kb.ingest_bytes(name, data, source="upload", rebuild=False)
        except Exception as exc:
            logger.exception("Failed to ingest %s", name)
            warnings.append(f"{name}: {exc}")
            continue
        if len(chunks) == 0:
            warnings.append(f"{name}: no readable text extracted (scanned/image-only PDFs require OCR which is not supported)")
        documents.append(document)
        total_chunks += len(chunks)
    if documents:
        kb.rebuild_indexes()
    if not documents:
        raise HTTPException(status_code=400, detail={"message": "No files ingested", "warnings": warnings})
    return UploadResponse(documents=documents, total_chunks=total_chunks, warnings=warnings)


@router.get("/documents", response_model=DocumentListResponse, tags=["documents"])
def list_documents() -> DocumentListResponse:
    from ..services.ingestion.metadata_extractor import (
        extract_metadata_from_text,
        is_generic_authors,
        is_generic_title,
    )

    kb = get_knowledge_base()
    docs = kb.store.list_documents()
    updated_any = False
    for doc in docs:
        if is_generic_title(doc.title, doc.name) or is_generic_authors(doc.authors):
            chunks = kb.store.all_chunks([doc.document_id])
            if chunks and chunks[0].text:
                t, a, y = extract_metadata_from_text(chunks[0].text, filename=doc.name)
                patch = {}
                if t and is_generic_title(doc.title, doc.name):
                    patch["title"] = t
                if a and is_generic_authors(doc.authors):
                    patch["authors"] = a
                if y and doc.year is None:
                    patch["year"] = y
                if patch:
                    kb.update_bibliography(doc.document_id, **patch)
                    updated_any = True
    if updated_any:
        docs = kb.store.list_documents()

    return DocumentListResponse(
        documents=docs,
        total_documents=len(docs),
        total_chunks=kb.store.chunk_count(),
        embedding_provider=kb.embedder.name,
        vector_store=kb.vectors.name,
    )


@router.get("/documents/extractions", response_model=PaperStructureListResponse, tags=["documents"])
def list_paper_extractions() -> PaperStructureListResponse:
    papers = get_knowledge_base().store.list_extractions()
    return PaperStructureListResponse(papers=papers, fields=list(PAPER_STRUCTURE_FIELDS), total=len(papers))


@router.post("/documents/extractions/refresh", response_model=PaperStructureListResponse, tags=["documents"])
def refresh_paper_extractions() -> PaperStructureListResponse:
    from ..services.extraction.paper_structure import refresh_document

    kb = get_knowledge_base()
    for document in kb.store.list_documents():
        refresh_document(kb.store, document.document_id)
    papers = kb.store.list_extractions()
    return PaperStructureListResponse(papers=papers, fields=list(PAPER_STRUCTURE_FIELDS), total=len(papers))


@router.get("/documents/{document_id}/extraction", response_model=PaperStructure, tags=["documents"])
def get_paper_extraction(document_id: str) -> PaperStructure:
    record = get_knowledge_base().store.get_extraction(document_id)
    if record is None:
        raise HTTPException(status_code=404, detail="No structured extraction for this paper")
    return record


@router.post("/documents/{document_id}/extract", response_model=PaperStructure, tags=["documents"])
def extract_one_paper(document_id: str) -> PaperStructure:
    from ..services.extraction.paper_structure import refresh_document

    kb = get_knowledge_base()
    if kb.store.get_document(document_id) is None:
        raise HTTPException(status_code=404, detail="Document not found")
    record = refresh_document(kb.store, document_id)
    if record is None:
        raise HTTPException(status_code=500, detail="Extraction failed")
    return record


@router.get("/documents/clusters", response_model=ClusterResponse, tags=["documents"])
def document_clusters() -> ClusterResponse:
    return cluster_documents(get_knowledge_base())


@router.patch("/documents/{document_id}", response_model=Document, tags=["documents"])
def update_document(document_id: str, payload: DocumentBiblioUpdate) -> Document:
    kb = get_knowledge_base()
    updated = kb.update_bibliography(document_id, **payload.model_dump(exclude_unset=True))
    if updated is None:
        raise HTTPException(status_code=404, detail="Document not found")
    return updated


@router.delete("/documents/{document_id}", tags=["documents"])
def delete_document(document_id: str) -> dict[str, Any]:
    kb = get_knowledge_base()
    if not kb.delete_document(document_id):
        raise HTTPException(status_code=404, detail="Document not found")
    return {"deleted": document_id}


@router.post("/documents/demo", tags=["documents"])
def reload_demo_corpus() -> dict[str, Any]:
    kb = get_knowledge_base()
    stats = load_demo_corpus(kb, reset=True)
    return {"status": "ok", **stats}


# --------------------------------------------------------------------------- papers
@router.get("/papers/search", response_model=PaperSearchResponse, tags=["papers"])
def papers_search(
    q: str = Query(default="", min_length=0),
    limit: int = Query(default=10, ge=1, le=25),
    filter: str = Query(default="all", alias="filter"),
) -> PaperSearchResponse:
    from ..services.papers.search import search_papers_detailed

    query = q.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Search query is required.")
    filt = (filter or "all").strip().lower()
    if filt not in {"all", "academic", "research_web", "open_access", "ieee"}:
        raise HTTPException(
            status_code=400,
            detail="filter must be one of: all, academic, research_web, open_access, ieee",
        )
    try:
        papers, provider, providers_used, notes = search_papers_detailed(
            query, limit=limit, search_filter=filt
        )
    except Exception as exc:
        logger.exception("Paper search failed")
        raise HTTPException(
            status_code=502,
            detail="Paper search is temporarily unavailable. Try again in a few seconds.",
        ) from exc
    return PaperSearchResponse(
        query=query,
        provider=provider,
        filter=filt,  # type: ignore[arg-type]
        providers_used=providers_used,
        notes=notes,
        papers=papers,
    )


@router.post("/papers/import", response_model=PaperImportResponse, tags=["papers"])
def papers_import(request: PaperImportRequest) -> PaperImportResponse:
    kb = get_knowledge_base()
    try:
        result = import_paper(request, kb)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Paper import failed")
        raise HTTPException(status_code=502, detail=f"Could not import paper: {exc}") from exc
    return PaperImportResponse(
        document=result.document.model_dump(mode="json"),
        ingested=result.ingested,
        warnings=result.warnings,
        paper_url=result.paper_url,
        pdf_url=result.pdf_url,
    )


# --------------------------------------------------------------------------- query
def _persist_ask(
    result: PipelineResult,
    http_request: Request | None,
    *,
    chat_id: str | None = None,
) -> str | None:
    history.add(result)
    if http_request is None:
        return None
    user = current_user_from_request(http_request)
    user_id = str((user or {}).get("user_id") or "").strip()
    if not user_id:
        return None
    ask_queries.save_ask_query(user_id, result)

    turn = AskChatTurn(
        query=result.query,
        result=result.model_dump(mode="json"),
        error="",
    )
    cid = str(chat_id or "").strip() or None
    existing = ask_chats.get_chat(user_id, cid) if cid else None
    turns = list(existing.turns) if existing else []
    # Replace same query_id turn if re-asked; otherwise append.
    qid = str(result.query_id or "")
    replaced = False
    next_turns = []
    for prev in turns:
        prev_qid = ""
        if isinstance(prev.result, dict):
            prev_qid = str(prev.result.get("query_id") or "")
        if prev_qid and prev_qid == qid:
            next_turns.append(turn)
            replaced = True
        else:
            next_turns.append(prev)
    if not replaced:
        next_turns.append(turn)
    saved = ask_chats.save_chat(
        user_id,
        chat_id=cid,
        title=(existing.title if existing else "") or result.query,
        turns=next_turns,
    )
    return saved.chat_id


def _load_ask(query_id: str, http_request: Request | None = None) -> PipelineResult | None:
    result = history.get(query_id)
    if result is not None:
        return result
    if http_request is None:
        return None
    user = current_user_from_request(http_request)
    user_id = str((user or {}).get("user_id") or "").strip()
    if not user_id:
        return None
    return ask_queries.get_ask_query(user_id, query_id)


@router.post("/query", response_model=PipelineResult, tags=["query"])
def query(payload: QueryRequest, request: Request) -> PipelineResult:
    runner = _runner(PipelineName.enhanced, threshold=payload.evidence_threshold)
    result = runner.run(
        payload.query,
        top_k=payload.top_k,
        document_ids=payload.document_ids,
        mode=payload.mode,
    )
    chat_id = _persist_ask(result, request, chat_id=payload.chat_id)
    if chat_id:
        # Stash for clients that read config_snapshot extras; keep response model stable.
        result.config_snapshot = {
            **(result.config_snapshot or {}),
            "chat_id": chat_id,
        }
    if not payload.include_trace:
        result.trace = []
    return result


@router.post("/query/compare", response_model=CompareResponse, tags=["query"])
def compare(payload: CompareRequest, request: Request) -> CompareResponse:
    results: dict[str, PipelineResult] = {}
    rows: list[ComparisonRow] = []
    query_id = ""
    for pipeline in payload.pipelines:
        result = _runner(pipeline).run(
            payload.query,
            top_k=payload.top_k,
            document_ids=payload.document_ids,
        )
        _persist_ask(result, request)
        results[pipeline.value] = result
        rows.append(_to_row(result))
        query_id = result.query_id
    winner, reason = _pick_winner(rows)
    return CompareResponse(
        query_id=query_id,
        query=payload.query,
        rows=rows,
        results=results,
        winner=winner,
        winner_reason=reason,
    )


@router.get("/query/{query_id}/trace", tags=["query"])
def get_trace(query_id: str, request: Request) -> dict[str, Any]:
    result = _load_ask(query_id, request)
    if result is None:
        raise HTTPException(status_code=404, detail="Query id not found in recent history")
    return {
        "query_id": result.query_id,
        "pipeline": result.pipeline,
        "query": result.query,
        "trace": [event.model_dump() for event in result.trace],
        "metrics": result.metrics.model_dump(),
        "confidence": result.confidence.model_dump(),
    }


@router.get("/query/recent", tags=["query"])
def recent_queries(request: Request, limit: int = Query(default=12, ge=1, le=50)) -> list[dict[str, Any]]:
    user = current_user_from_request(request)
    user_id = str((user or {}).get("user_id") or "").strip()
    out = []
    if user_id:
        entries = ask_queries.list_ask_queries(user_id, limit=limit)
        source = [entry["result"] for entry in entries]
    else:
        source = history.recent(limit)
    for result in source:
        out.append(
            {
                "query_id": result.query_id,
                "pipeline": result.pipeline,
                "pipeline_label": result.pipeline_label,
                "query": result.query,
                "status": result.status,
                "confidence": result.confidence.confidence,
                "latency_ms": result.metrics.latency_ms,
            }
        )
    return out


@router.get("/query/history/{query_id}", response_model=PipelineResult, tags=["query"])
def get_history_item(query_id: str, request: Request) -> PipelineResult:
    result = _load_ask(query_id, request)
    if result is None:
        raise HTTPException(status_code=404, detail="Query id not found in recent history")
    return result


# ----------------------------------------------------------------------- evaluation
@router.post("/evaluate", response_model=EvaluationRun, tags=["evaluation"])
def evaluate(request: EvaluateRequest) -> EvaluationRun:
    harness = EvaluationHarness(get_knowledge_base(), get_llm_provider())
    if request.include_headline:
        return harness.run_headline(
            limit=request.limit,
            categories=request.categories,
            k=request.k,
            persist=request.persist,
        )
    return harness.run(
        pipelines=request.pipelines,
        include_ablation=request.include_ablation,
        limit=request.limit,
        categories=request.categories,
        k=request.k,
        persist=request.persist,
    )


@router.get("/evaluation/results", tags=["evaluation"])
def evaluation_results() -> dict[str, Any]:
    run = load_latest_run()
    if run is None:
        return {
            "run": None,
            "summary": None,
            "table": [],
            "headline_table": [],
            "message": "No evaluation has been run yet.",
        }
    return {
        "run": run.model_dump(mode="json"),
        "summary": summarise(run).model_dump(mode="json"),
        "table": comparison_table(run),
        "headline_table": headline_table(run),
    }


@router.get("/evaluation/runs", tags=["evaluation"])
def evaluation_runs() -> list[dict[str, Any]]:
    return [item.model_dump(mode="json") for item in list_runs()]


@router.get("/evaluation/benchmark", tags=["evaluation"])
def get_benchmark() -> dict[str, Any]:
    dataset = benchmark_dataset()
    return {
        "name": dataset.name,
        "description": dataset.description,
        "n_questions": len(dataset.questions),
        "questions": [q.model_dump(mode="json") for q in dataset.questions],
    }


@router.get("/evaluation/demo-cases", tags=["evaluation"])
def demo_cases() -> list[dict[str, str]]:
    """Curated questions that demonstrate each architectural behaviour."""

    return [
        {
            "id": "normal",
            "title": "Normal factual question",
            "query": "What is the main advantage of dropout?",
            "expect": "Answered with citations from the handbook.",
        },
        {
            "id": "multi_document",
            "title": "Multi-document question",
            "query": "How does hybrid retrieval using RRF relate to the remaining gaps in standard Self-RAG?",
            "expect": "Evidence combined from the survey and the hybrid-retrieval notes.",
        },
        {
            "id": "rewrite",
            "title": "Rare identifier (may trigger rewrite)",
            "query": "What k does RRF-7F3 use?",
            "expect": "Hybrid retrieval / rewrite recovers the lexical identifier; k = 60.",
        },
        {
            "id": "refuse",
            "title": "No supporting evidence",
            "query": "How does dropout affect Titan's nitrogen atmosphere?",
            "expect": "Enhanced Self-RAG refuses; Traditional RAG may hallucinate a fluent mix.",
        },
        {
            "id": "conflict",
            "title": "Contradictory sources",
            "query": "Does dropout reduce overfitting?",
            "expect": "Conflict between the handbook and the internal memo is reported, not collapsed.",
        },
        {
            "id": "architecture",
            "title": "Proposed-architecture advantage",
            "query": "What is the main advantage of the proposed architecture?",
            "expect": "Answers only when evidence is sufficient; claim-level verification.",
        },
    ]
