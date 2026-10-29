"""Adaptive retrieval controller and query rewriting.

Two related decisions live here:

* **How many documents to retrieve next** — start small, expand if the gate
  says the evidence is thin, stop early if the evidence is already strong.
* **How to rewrite the query** — if the gate says the ranking is about the
  wrong thing, produce a small set of reformulations targeted at the gaps.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..config import Settings, get_settings
from ..models.query import EvidenceGateDecision, QueryAnalysis
from ..services.llm.base import LLMProvider, LLMRequest, LLMTask
from ..services.llm import extractive_engine as engine
from ..text_utils import definition_subject


REWRITE_PROMPT = """Rewrite the user question into up to {limit} alternative search queries
that are more likely to retrieve supporting evidence.

Focus on the uncovered terms: {uncovered}
Keep the original intent. Do not answer the question.

Original question: {query}

Return JSON: {{"rewrites": ["...", "..."]}}
"""


@dataclass(slots=True)
class AdaptivePlan:
    next_top_k: int
    rewrite: str | None
    stop: bool
    reason: str


class AdaptiveController:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()

    def initial_top_k(self, analysis: QueryAnalysis, override: int | None = None) -> int:
        if override is not None:
            return max(1, min(override, self.settings.max_top_k))
        return max(1, min(analysis.suggested_top_k, self.settings.max_top_k))

    def plan(
        self,
        decision: EvidenceGateDecision,
        current_top_k: int,
        analysis: QueryAnalysis,
        attempts_used: int,
        rewrites: list[str],
        llm: LLMProvider | None = None,
        corpus_terms: list[str] | None = None,
    ) -> AdaptivePlan:
        settings = self.settings
        remaining = settings.max_retrieval_attempts - attempts_used
        if remaining <= 0 or decision.action in {"proceed", "conflict"}:
            return AdaptivePlan(
                next_top_k=current_top_k,
                rewrite=None,
                stop=True,
                reason="Evidence is sufficient or no attempts remain.",
            )
        if decision.action == "abstain":
            return AdaptivePlan(
                next_top_k=current_top_k,
                rewrite=None,
                stop=True,
                reason=decision.rationale,
            )

        if decision.action == "expand":
            next_k = min(settings.max_top_k, current_top_k + settings.top_k_increment)
            if next_k <= current_top_k:
                # Depth is already maxed; fall through to a rewrite instead of looping.
                rewrite = self._next_rewrite(
                    analysis, decision, rewrites, llm, corpus_terms
                )
                return AdaptivePlan(
                    next_top_k=current_top_k,
                    rewrite=rewrite,
                    stop=rewrite is None,
                    reason="Retrieval depth is already at the configured maximum; rewriting.",
                )
            return AdaptivePlan(
                next_top_k=next_k,
                rewrite=None,
                stop=False,
                reason=f"Increasing retrieval depth from {current_top_k} to {next_k}.",
            )

        rewrite = self._next_rewrite(analysis, decision, rewrites, llm, corpus_terms)
        return AdaptivePlan(
            next_top_k=current_top_k,
            rewrite=rewrite,
            stop=rewrite is None,
            reason=(
                f"Trying rewritten query {rewrite!r}."
                if rewrite
                else "No unused rewrite remaining."
            ),
        )

    def _next_rewrite(
        self,
        analysis: QueryAnalysis,
        decision: EvidenceGateDecision,
        already: list[str],
        llm: LLMProvider | None,
        corpus_terms: list[str] | None,
    ) -> str | None:
        used = {analysis.original_query.lower(), *(r.lower() for r in already)}
        candidates = list(already)
        extra = self._generate_rewrites(analysis, decision, llm, corpus_terms)
        for candidate in extra:
            if candidate.lower() not in used:
                candidates.append(candidate)
        for candidate in candidates:
            if candidate.lower() not in used:
                return candidate
        return None

    def _generate_rewrites(
        self,
        analysis: QueryAnalysis,
        decision: EvidenceGateDecision,
        llm: LLMProvider | None,
        corpus_terms: list[str] | None,
    ) -> list[str]:
        prioritized: list[str] = []
        if analysis.question_type == "definition":
            subject = definition_subject(analysis.original_query)
            if subject and subject.lower() != analysis.original_query.lower().strip():
                prioritized.append(subject)

        payload = {
            "query": analysis.original_query,
            "keywords": analysis.keywords,
            "entities": analysis.entities,
            "uncovered_terms": decision.uncovered_terms,
            "corpus_terms": corpus_terms or [],
            "limit": 3,
        }
        generated: list[str] = []
        if llm is not None:
            try:
                response = llm.generate(
                    LLMRequest(
                        task=LLMTask.rewrite_query,
                        prompt=REWRITE_PROMPT.format(
                            limit=3,
                            uncovered=", ".join(decision.uncovered_terms) or "(none)",
                            query=analysis.original_query,
                        ),
                        system="You rewrite search queries. Return JSON only.",
                        payload=payload,
                    )
                )
                rewrites = (response.structured or {}).get("rewrites") or []
                if rewrites:
                    generated = [str(r).strip() for r in rewrites if str(r).strip()]
            except Exception:
                pass
        if not generated:
            generated = engine.rewrite_queries(
                query=analysis.original_query,
                keywords=analysis.keywords,
                entities=analysis.entities,
                uncovered_terms=decision.uncovered_terms,
                corpus_terms=corpus_terms or [],
                limit=3,
                question_type=analysis.question_type,
            )
        # Subject noun phrase first so definition questions retry the core term.
        return [*prioritized, *generated]
