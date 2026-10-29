"""Offline, deterministic LLM provider.

Dispatches each :class:`LLMTask` to the corresponding routine in
:mod:`app.services.llm.extractive_engine`. Requires no network and no API key,
so the full evaluation harness is reproducible on any machine.
"""

from __future__ import annotations

import json
from typing import Any

from . import extractive_engine as engine
from .base import LLMProvider, LLMRequest, LLMResponse, LLMTask, estimate_tokens


class ExtractiveProvider(LLMProvider):
    name = "extractive"
    model = "deterministic-grounded-v1"
    deterministic = True

    def generate(self, request: LLMRequest) -> LLMResponse:
        payload = request.payload or {}
        handler = {
            LLMTask.answer: self._answer,
            LLMTask.revise: self._answer,
            LLMTask.extract_claims: self._extract_claims,
            LLMTask.extract_paper_structure: self._extract_paper_structure,
            LLMTask.rewrite_query: self._rewrite_query,
            LLMTask.reflect: self._reflect,
            LLMTask.retrieval_decision: self._retrieval_decision,
            LLMTask.manuscript_assist: self._manuscript_assist,
        }.get(request.task)

        if handler is None:  # pragma: no cover - defensive
            raise ValueError(f"Unsupported task for ExtractiveProvider: {request.task}")

        structured = handler(payload)
        if request.task in (LLMTask.answer, LLMTask.revise):
            text = str(structured.get("answer") or "")
        elif request.task is LLMTask.manuscript_assist:
            text = str(structured.get("result_text") or "")
        else:
            text = structured.get("answer") or json.dumps(structured, ensure_ascii=False)
        prompt_chars = sum(
            len(str(item.get("text", ""))) for item in payload.get("evidence", [])
        ) + len(str(payload.get("query", "")))

        return LLMResponse(
            text=text,
            prompt_tokens=estimate_tokens("x" * prompt_chars),
            completion_tokens=estimate_tokens(text),
            provider=self.name,
            model=self.model,
            structured=structured,
        )

    # ------------------------------------------------------------------ tasks
    def _answer(self, payload: dict[str, Any]) -> dict[str, Any]:
        return engine.compose_answer(
            query=payload.get("query", ""),
            evidence=payload.get("evidence", []),
            focus_terms=payload.get("focus_terms", ()),
            avoid_claims=payload.get("avoid_claims", ()),
            include_lead=payload.get("include_lead", True),
            max_sentences=payload.get("max_sentences", engine.MAX_SUPPORTING_SENTENCES),
            question_type=str(payload.get("question_type") or ""),
        )

    def _extract_claims(self, payload: dict[str, Any]) -> dict[str, Any]:
        claims = engine.extract_claims(
            payload.get("answer", ""), max_claims=payload.get("max_claims", 12)
        )
        return {"claims": claims}

    def _extract_paper_structure(self, payload: dict[str, Any]) -> dict[str, Any]:
        from ...models.documents import Chunk, ChunkMetadata
        from ..extraction.paper_structure import _heuristic_fields

        chunks: list[Chunk] = []
        for item in payload.get("chunks") or []:
            chunks.append(
                Chunk(
                    chunk_id=str(item.get("chunk_id") or f"c{len(chunks)+1}"),
                    text=str(item.get("text") or ""),
                    metadata=ChunkMetadata(document_id="doc", document_name="paper"),
                )
            )
        return _heuristic_fields(chunks)

    def _rewrite_query(self, payload: dict[str, Any]) -> dict[str, Any]:
        rewrites = engine.rewrite_queries(
            query=payload.get("query", ""),
            keywords=payload.get("keywords", ()),
            entities=payload.get("entities", ()),
            uncovered_terms=payload.get("uncovered_terms", ()),
            corpus_terms=payload.get("corpus_terms", ()),
            limit=payload.get("limit", 3),
        )
        return {"rewrites": rewrites}

    def _reflect(self, payload: dict[str, Any]) -> dict[str, Any]:
        return engine.reflect(
            answer=payload.get("answer", ""),
            evidence=payload.get("evidence", []),
            query=payload.get("query", ""),
        )

    def _retrieval_decision(self, payload: dict[str, Any]) -> dict[str, Any]:
        # The heuristic analyser owns this decision; the provider simply echoes
        # it so hosted providers can override the behaviour later.
        return {
            "needs_retrieval": bool(payload.get("needs_retrieval", True)),
            "reason": payload.get("reason", "document-grounded question"),
        }

    def _manuscript_assist(self, payload: dict[str, Any]) -> dict[str, Any]:
        action = str(payload.get("action") or "")
        text = str(payload.get("text") or "")
        evidence = payload.get("evidence") or []
        if action in ("suggest_evidence", "summarize_evidence") and not evidence:
            return {
                "result_text": "",
                "suggestions": [],
                "warnings": ["No project evidence available."],
                "missing_evidence": True,
            }
        if action == "suggest_outline":
            titles = payload.get("section_titles") or []
            lines = [f"{i}. {t}" for i, t in enumerate(titles, start=1)] or ["1. Abstract", "2. Introduction"]
            return {
                "result_text": "\n".join(lines),
                "suggestions": lines,
                "warnings": ["Offline outline from existing section titles."],
                "missing_evidence": False,
            }
        if action == "summarize_evidence":
            quotes = [str(e.get("quote") or "").strip() for e in evidence if e.get("quote")]
            return {
                "result_text": "\n".join(f"- {q[:300]}" for q in quotes[:8]),
                "suggestions": quotes[:8],
                "warnings": ["Offline evidence summary uses stored quotes only."],
                "missing_evidence": not bool(quotes),
            }
        return {
            "result_text": text,
            "suggestions": [],
            "warnings": ["Limited offline manuscript assist."],
            "missing_evidence": False,
        }

