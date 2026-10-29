"""User-triggered manuscript AI assists — never invent evidence or full papers."""

from __future__ import annotations

import json
import re
from typing import Any

from ...models.projects import (
    MANUSCRIPT_TEMPLATES,
    ManuscriptAiAction,
    ManuscriptAiRequest,
    ManuscriptAiResponse,
    ProjectEvidence,
    ProjectManuscript,
)
from ...services.llm.base import LLMRequest, LLMTask
from ...services.llm.registry import get_llm_provider
from . import store as project_store

_SYSTEM = (
    "You assist with academic manuscript editing inside a research project. "
    "Do not invent references, bibliographic metadata, scientific findings, data, "
    "or evidence quotes. Do not generate a complete paper. "
    "If evidence is missing, say so clearly. Preserve uncertainty."
)


def run_manuscript_ai(project_id: str, payload: ManuscriptAiRequest) -> ManuscriptAiResponse:
    action: ManuscriptAiAction = payload.action
    project = project_store.get_project(project_id)
    if project is None:
        raise ValueError("Project not found.")
    ms = project_store.get_manuscript(project_id)
    if ms is None:
        raise ValueError("Manuscript not found.")

    section = None
    if payload.section_id:
        section = next(
            (s for s in ms.sections if s.section_id == payload.section_id), None
        )
        if section is None:
            raise ValueError("Section not found.")

    evidence = _select_evidence(project.evidence or [], payload.evidence_ids)
    text = (payload.selection or "").strip() or (section.body if section else "")

    if action in ("suggest_evidence", "summarize_evidence"):
        if not evidence:
            return ManuscriptAiResponse(
                action=action,
                result_text="",
                suggestions=[],
                warnings=[
                    "No project evidence is available for this action. "
                    "Link sources and add evidence quotes first."
                ],
                missing_evidence=True,
                provider="none",
            )

    provider = get_llm_provider()
    if getattr(provider, "deterministic", False) or provider.name == "extractive":
        return _extractive_assist(action, text=text, ms=ms, evidence=evidence, provider=provider.name)

    prompt = _build_prompt(action, text=text, ms=ms, evidence=evidence, section_title=section.title if section else "")
    try:
        response = provider.generate(
            LLMRequest(
                task=LLMTask.manuscript_assist,
                system=_SYSTEM,
                prompt=prompt,
                payload={
                    "action": action,
                    "text": text,
                    "evidence": [
                        {"evidence_id": e.evidence_id, "quote": e.quote, "title": e.title}
                        for e in evidence
                    ],
                    "section_titles": [s.title for s in ms.sections],
                },
                temperature=0.2,
                max_tokens=1200,
            )
        )
    except Exception as exc:  # pragma: no cover - network
        fallback = _extractive_assist(
            action, text=text, ms=ms, evidence=evidence, provider="fallback"
        )
        fallback.warnings = [
            *fallback.warnings,
            f"AI provider failed ({exc.__class__.__name__}); used limited offline assist.",
        ]
        return fallback

    structured = response.structured or {}
    result_text = str(structured.get("result_text") or response.text or "").strip()
    suggestions = [
        str(s).strip()
        for s in (structured.get("suggestions") or [])
        if str(s).strip()
    ]
    warnings = [
        str(w).strip()
        for w in (structured.get("warnings") or [])
        if str(w).strip()
    ]
    warnings.append(
        "AI output is not plagiarism-free and is not official IEEE compliance. Review before applying."
    )
    missing = bool(structured.get("missing_evidence"))
    if action in ("suggest_evidence", "summarize_evidence") and not evidence:
        missing = True
    return ManuscriptAiResponse(
        action=action,
        result_text=result_text,
        suggestions=suggestions,
        warnings=warnings,
        missing_evidence=missing,
        provider=response.provider or provider.name,
    )


def _select_evidence(
    all_evidence: list[ProjectEvidence], evidence_ids: list[str]
) -> list[ProjectEvidence]:
    if not evidence_ids:
        return list(all_evidence)
    wanted = set(evidence_ids)
    return [e for e in all_evidence if e.evidence_id in wanted]


def _build_prompt(
    action: ManuscriptAiAction,
    *,
    text: str,
    ms: ProjectManuscript,
    evidence: list[ProjectEvidence],
    section_title: str,
) -> str:
    ev_block = "\n".join(
        f"- [{e.evidence_id}] ({e.title or e.document_id}): {e.quote[:500]}"
        for e in evidence[:12]
    ) or "(none)"
    headings = ", ".join(s.title for s in ms.sections)
    instructions = {
        "improve_grammar": "Improve grammar and spelling only. Do not add new claims or citations.",
        "improve_clarity": "Improve clarity and flow. Do not invent facts or references.",
        "suggest_outline": f"Suggest a concise outline for this {ms.document_type} manuscript using existing headings where possible: {headings}.",
        "identify_citation_gaps": "List sentences/claims that likely need citations. Do not invent references.",
        "suggest_evidence": "From the PROJECT EVIDENCE quotes only, suggest which quotes may support the text. Do not invent quotes.",
        "summarize_evidence": "Summarize only the provided PROJECT EVIDENCE quotes. If insufficient, say so.",
    }[action]
    return (
        f"Action: {action}\n"
        f"Section: {section_title or '(n/a)'}\n"
        f"Instructions: {instructions}\n\n"
        f"TEXT:\n{text or '(empty)'}\n\n"
        f"PROJECT EVIDENCE:\n{ev_block}\n\n"
        "Respond as JSON with keys: result_text (string), suggestions (string array), "
        "warnings (string array), missing_evidence (boolean)."
    )


def _extractive_assist(
    action: ManuscriptAiAction,
    *,
    text: str,
    ms: ProjectManuscript,
    evidence: list[ProjectEvidence],
    provider: str,
) -> ManuscriptAiResponse:
    warnings = [
        "Limited offline assist (deterministic provider). Results are heuristic, not model-generated prose."
    ]
    if action in ("improve_grammar", "improve_clarity"):
        cleaned = re.sub(r"\s+", " ", text or "").strip()
        return ManuscriptAiResponse(
            action=action,
            result_text=cleaned or text,
            suggestions=[],
            warnings=[
                *warnings,
                "No rewriting applied offline — review the text manually or configure an LLM provider.",
            ],
            missing_evidence=False,
            provider=provider,
        )
    if action == "suggest_outline":
        tpl = MANUSCRIPT_TEMPLATES.get(ms.document_type) or MANUSCRIPT_TEMPLATES["ieee_research"]
        lines = [f"{i}. {title}" for i, (_, title) in enumerate(tpl, start=1)]
        return ManuscriptAiResponse(
            action=action,
            result_text="\n".join(lines),
            suggestions=lines,
            warnings=warnings,
            missing_evidence=False,
            provider=provider,
        )
    if action == "identify_citation_gaps":
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text or "") if s.strip()]
        gaps = [
            s
            for s in sentences
            if len(s.split()) >= 8
            and not re.search(r"\[\d+\]|\(\d{4}\)|et al\.", s, re.I)
        ]
        if not gaps:
            return ManuscriptAiResponse(
                action=action,
                result_text="No obvious uncited claims detected with the offline heuristic.",
                suggestions=[],
                warnings=warnings,
                missing_evidence=False,
                provider=provider,
            )
        return ManuscriptAiResponse(
            action=action,
            result_text="Possible claims needing citations:\n" + "\n".join(f"- {g}" for g in gaps[:12]),
            suggestions=gaps[:12],
            warnings=warnings,
            missing_evidence=False,
            provider=provider,
        )
    if action == "suggest_evidence":
        suggestions = [
            f"{e.title or e.document_id}: {e.quote[:240]}"
            for e in evidence[:10]
            if e.quote
        ]
        return ManuscriptAiResponse(
            action=action,
            result_text="\n\n".join(suggestions) if suggestions else "",
            suggestions=suggestions,
            warnings=warnings,
            missing_evidence=not bool(suggestions),
            provider=provider,
        )
    # summarize_evidence
    quotes = [e.quote.strip() for e in evidence if e.quote.strip()]
    if not quotes:
        return ManuscriptAiResponse(
            action=action,
            result_text="",
            suggestions=[],
            warnings=[
                *warnings,
                "No evidence quotes to summarize.",
            ],
            missing_evidence=True,
            provider=provider,
        )
    summary = "Evidence summary (verbatim excerpts only):\n" + "\n".join(
        f"- {q[:300]}" for q in quotes[:8]
    )
    return ManuscriptAiResponse(
        action=action,
        result_text=summary,
        suggestions=quotes[:8],
        warnings=warnings,
        missing_evidence=False,
        provider=provider,
    )
