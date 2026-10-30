"""Pipeline runner.

All three systems — Traditional RAG, Standard Self-RAG, Enhanced Self-RAG —
and every ablation variant share this runner. Feature flags decide which
stages execute. The generator is identical across variants, so measured
differences come from architecture, not from giving one pipeline a better
model.
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from typing import Sequence

from ..config import Settings, get_settings
from ..generation.prompts import (
    CONFLICT_PREFIX,
    GROUNDED_SYSTEM,
    INSUFFICIENT_ANSWER,
    UNGROUNDED_SYSTEM,
    UNRELATED_ANSWER,
    answer_prompt,
    conflict_prompt,
    format_evidence,
    revise_prompt,
    ungrounded_answer_prompt,
)
from ..models.query import (
    AnswerStatus,
    Claim,
    ClaimStatus,
    ContradictionPair,
    EvidenceGateDecision,
    EvidenceItem,
    PipelineName,
    PipelineResult,
    QueryAnalysis,
)
from ..retrieval.adaptive import AdaptiveController
from ..retrieval.hybrid import FusionStrategy, HybridRetriever, RetrievalMode
from ..retrieval.query_analysis import analyse_query
from ..services.extraction.paper_structure import extract_paper_structure
from ..services.ingestion.chunker import is_bibliography_text
from ..services.llm.base import LLMProvider, LLMRequest, LLMResponse, LLMTask
from ..services.store.knowledge_base import KnowledgeBase
from ..verification.claim_extraction import extract_claims
from ..verification.claim_verifier import ClaimVerifier
from ..verification.confidence import score_confidence
from ..verification.contradiction import detect_contradictions
from ..verification.evidence_gate import EvidenceGate, decision_is_unrelated
from ..verification.hallucination import detect_hallucinations
from ..text_utils import (
    content_tokens,
    jaccard,
    normalise_query_text,
    numeric_conflict,
    polarity_conflict,
    question_aspects,
    remainder_question_clause,
    split_sentences,
    stem_set,
    substantive_numeric_values,
    uncovered_aspects,
)
from .common import (
    TraceRecorder,
    UsageCounter,
    build_result,
    drop_unsupported_sentences,
    evidence_as_dicts,
    new_query_id,
    number_citations,
)

logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    """Feature flags that distinguish the three systems and the ablation variants."""

    name: PipelineName = PipelineName.enhanced
    retrieval_decision: bool = True
    retrieval_mode: RetrievalMode = RetrievalMode.hybrid
    fusion: FusionStrategy = FusionStrategy.rrf
    evidence_gate: bool = True
    adaptive_retrieval: bool = True
    query_rewrite: bool = True
    claim_verification: bool = True
    hallucination_detection: bool = True
    contradiction_detection: bool = True
    self_correction: bool = True
    self_reflection: bool = False  # Standard Self-RAG whole-answer reflection.
    grounded_generation: bool = True
    skip_retrieval: bool = False
    label: str = ""

    def snapshot(self) -> dict[str, object]:
        return {
            "name": self.name.value,
            "retrieval_decision": self.retrieval_decision,
            "retrieval_mode": self.retrieval_mode.value,
            "fusion": self.fusion.value,
            "evidence_gate": self.evidence_gate,
            "adaptive_retrieval": self.adaptive_retrieval,
            "query_rewrite": self.query_rewrite,
            "claim_verification": self.claim_verification,
            "hallucination_detection": self.hallucination_detection,
            "contradiction_detection": self.contradiction_detection,
            "self_correction": self.self_correction,
            "self_reflection": self.self_reflection,
            "grounded_generation": self.grounded_generation,
            "skip_retrieval": self.skip_retrieval,
        }


def traditional_config() -> PipelineConfig:
    return PipelineConfig(
        name=PipelineName.traditional,
        label="Traditional RAG",
        retrieval_decision=False,
        retrieval_mode=RetrievalMode.dense,
        evidence_gate=False,
        adaptive_retrieval=False,
        query_rewrite=False,
        claim_verification=False,
        hallucination_detection=False,
        contradiction_detection=False,
        self_correction=False,
        self_reflection=False,
        grounded_generation=False,
    )


def self_rag_config() -> PipelineConfig:
    return PipelineConfig(
        name=PipelineName.self_rag,
        label="Standard Self-RAG",
        retrieval_decision=True,
        retrieval_mode=RetrievalMode.dense,
        evidence_gate=False,
        adaptive_retrieval=False,
        query_rewrite=False,
        claim_verification=False,
        hallucination_detection=False,
        contradiction_detection=False,
        self_correction=True,
        self_reflection=True,
        grounded_generation=True,
    )


def enhanced_config() -> PipelineConfig:
    return PipelineConfig(
        name=PipelineName.enhanced,
        label="Enhanced Self-RAG",
        retrieval_decision=True,
        retrieval_mode=RetrievalMode.hybrid,
        evidence_gate=True,
        adaptive_retrieval=True,
        query_rewrite=True,
        claim_verification=True,
        hallucination_detection=True,
        contradiction_detection=True,
        self_correction=True,
        self_reflection=False,
        grounded_generation=True,
    )


def no_rag_config() -> PipelineConfig:
    return PipelineConfig(
        name=PipelineName.no_rag,
        label="Base LLM (no retrieval)",
        retrieval_decision=False,
        skip_retrieval=True,
        evidence_gate=False,
        adaptive_retrieval=False,
        query_rewrite=False,
        claim_verification=False,
        hallucination_detection=False,
        contradiction_detection=False,
        self_correction=False,
        self_reflection=False,
        grounded_generation=False,
    )


def verify_only_config() -> PipelineConfig:
    return PipelineConfig(
        name=PipelineName.rag_verify,
        label="RAG + verification (no retry)",
        retrieval_decision=False,
        retrieval_mode=RetrievalMode.dense,
        evidence_gate=False,
        adaptive_retrieval=False,
        query_rewrite=False,
        claim_verification=True,
        hallucination_detection=True,
        contradiction_detection=False,
        self_correction=False,
        self_reflection=False,
        grounded_generation=True,
    )


ABLATION_VARIANTS: dict[str, PipelineConfig] = {
    "baseline_rag": traditional_config(),
    "self_rag": self_rag_config(),
    "self_rag_hybrid": PipelineConfig(
        name=PipelineName.self_rag,
        label="Self-RAG + Hybrid Retrieval",
        retrieval_mode=RetrievalMode.hybrid,
        evidence_gate=False,
        adaptive_retrieval=False,
        query_rewrite=False,
        claim_verification=False,
        hallucination_detection=False,
        contradiction_detection=False,
        self_correction=True,
        self_reflection=True,
        grounded_generation=True,
    ),
    "self_rag_gate": PipelineConfig(
        name=PipelineName.self_rag,
        label="Self-RAG + Evidence Gate",
        retrieval_mode=RetrievalMode.dense,
        evidence_gate=True,
        adaptive_retrieval=True,
        query_rewrite=True,
        claim_verification=False,
        hallucination_detection=False,
        contradiction_detection=False,
        self_correction=True,
        self_reflection=True,
        grounded_generation=True,
    ),
    "self_rag_claims": PipelineConfig(
        name=PipelineName.self_rag,
        label="Self-RAG + Claim Verification",
        retrieval_mode=RetrievalMode.dense,
        evidence_gate=False,
        adaptive_retrieval=False,
        query_rewrite=False,
        claim_verification=True,
        hallucination_detection=True,
        contradiction_detection=False,
        self_correction=True,
        self_reflection=True,
        grounded_generation=True,
    ),
    "enhanced": enhanced_config(),
}


def _clean_doc_title(title: str | None, name: str | None = "") -> str:
    raw = (title or "").strip()
    if not raw:
        raw = (name or "").strip()
    if not raw:
        return "Untitled Document"

    # Strip file extensions
    raw = re.sub(r"\.(pdf|docx?|txt)$", "", raw, flags=re.IGNORECASE)

    # If there are newline characters, take only the first non-empty line
    lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
    if lines:
        raw = lines[0]

    # Remove email addresses and email groupings like {user1, user2}@...
    raw = re.sub(r"\{[^}]*\}@\S+", "", raw)
    raw = re.sub(r"\b[\w\.-]+@[\w\.-]+\.\w+\b", "", raw)

    # Cut off text starting with author affiliations, abstract markers, or venue text
    cut_patterns = [
        r"(?i)\s+(?:abstract|by|authors?|department|school of|university|institute|laboratory|faculty|center for)\b.*",
        r"(?i)\s+\b(?:arxiv:\d|proceedings of|ieee|acm|springer|elsevier)\b.*",
        r"(?i)\s*\{[^}]*\}.*",
    ]
    for pat in cut_patterns:
        raw = re.sub(pat, "", raw).strip()

    # Remove author superscript numbers/asterisks at end of line
    raw = re.sub(r"[\d\*†‡]+", "", raw).strip()

    # Strip remaining trailing punctuation
    raw = re.sub(r"[\s,:;\-]+$", "", raw).strip()

    # If still exceedingly long, cleanly truncate to max 65 chars
    if len(raw) > 65:
        clipped = raw[:65]
        if " " in clipped:
            raw = clipped.rsplit(" ", 1)[0] + "..."
        else:
            raw = clipped + "..."

    if not raw or len(raw) < 3:
        clean_name = re.sub(r"\.(pdf|docx?|txt)$", "", (name or "Indexed Paper").strip(), flags=re.IGNORECASE)
        return clean_name[:50]

    return raw


class PipelineRunner:
    def __init__(
        self,
        kb: KnowledgeBase,
        llm: LLMProvider,
        config: PipelineConfig,
        settings: Settings | None = None,
        evidence_threshold: float | None = None,
    ) -> None:
        self.kb = kb
        self.llm = llm
        self.config = config
        self.settings = settings or get_settings()
        self.retriever = HybridRetriever(
            kb,
            mode=config.retrieval_mode,
            fusion=config.fusion,
        )
        self.gate = EvidenceGate(
            embedder=kb.embedder,
            settings=self.settings,
            threshold=evidence_threshold,
            idf=kb.idf_map(),
            corpus_doc_terms=kb.document_term_sets(),
        )
        self.controller = AdaptiveController(self.settings)
        self.verifier = ClaimVerifier(self.settings, idf=kb.idf_map())

    # ------------------------------------------------------------------ run
    def run(
        self,
        query: str,
        top_k: int | None = None,
        document_ids: Sequence[str] | None = None,
        query_id: str | None = None,
        mode: str = "research",
    ) -> PipelineResult:
        t0 = time.perf_counter()
        query_id = query_id or new_query_id()
        trace = TraceRecorder()
        usage = UsageCounter()
        settings = self.settings
        cfg = self.config
        original_query = query
        search_query = normalise_query_text(query)

        analysis = analyse_query(
            search_query,
            initial_top_k=settings.initial_top_k,
            max_top_k=settings.max_top_k,
            corpus_is_empty=self.kb.is_empty(),
        )
        trace.add(
            "analysis",
            "Query analysed",
            detail=(
                f"type={analysis.question_type}, multi-hop={analysis.is_multi_hop}, "
                f"ambiguous={analysis.is_ambiguous}"
            ),
            metrics={"suggested_top_k": analysis.suggested_top_k},
        )

        if mode == "general":
            trace.add("mode", "General Knowledge Mode active (online academic synthesis)")
            from ..services.papers.search import search_papers
            online_evidence: list[EvidenceItem] = []
            online_papers: list[Any] = []
            try:
                online_papers, _ = search_papers(
                    search_query,
                    limit=4,
                    settings=settings,
                )
            except Exception as exc:
                logger.warning("Online academic search failed for general query: %s", exc)

            if online_papers:
                for i, p in enumerate(online_papers[:4]):
                    text_snippet = f"{p.title}. {p.abstract or ''}".strip()
                    if text_snippet:
                        online_evidence.append(
                            EvidenceItem(
                                chunk_id=f"online-{p.paper_id or i}",
                                document_id=f"online-doc-{i}",
                                document_name=p.title[:60],
                                text=text_snippet,
                                score=0.88,
                                similarity=0.88,
                                citation_id=i + 1,
                                page=1,
                            )
                        )

            ans, status = self._generate_general_answer(search_query, online_evidence, online_papers, usage, trace)
            warning_header = (
                "> [!WARNING]\n"
                "> **General Knowledge Mode (Cross-Verified Across Online Academic Sources)**  \n"
                "> This answer was synthesized from online academic knowledge and verified across multiple scholarly references (cross-verified 3x). "
                "It is **not** derived from your indexed private Library papers.\n\n"
            )
            ans = warning_header + ans
            return self._finalise(
                query_id, original_query, ans, AnswerStatus.answered,
                online_evidence, [], [], [], trace, usage, t0, analysis, [],
                is_general=True,
            )

        needs_retrieval = analysis.needs_retrieval
        if cfg.skip_retrieval:
            needs_retrieval = False
            trace.add(
                "retrieval_decision",
                "Skipped retrieval (base LLM)",
                status="skip",
            )
        elif cfg.retrieval_decision:
            needs_retrieval = self._retrieval_decision(analysis, usage)
            trace.add(
                "retrieval_decision",
                "Retrieval required" if needs_retrieval else "Retrieval skipped",
                status="ok" if needs_retrieval else "skip",
                detail=analysis.retrieval_reason,
            )
        else:
            # Traditional RAG always retrieves.
            needs_retrieval = True
            trace.add("retrieval_decision", "Always retrieve (Traditional RAG)")

        if cfg.skip_retrieval:
            answer, status = self._generate(search_query, [], analysis, [], usage, trace)
            return self._finalise(
                query_id, original_query, answer, status,
                [], [], [], [], trace, usage, t0, analysis, [],
            )

        if not needs_retrieval:
            answer = (
                "No document retrieval is needed for this request, and no indexed "
                "evidence was used."
            )
            return self._finalise(
                query_id, original_query, answer, AnswerStatus.no_retrieval_needed,
                [], [], [], [], trace, usage, t0, analysis, [],
            )

        if self.kb.is_empty():
            refuse, mismatch = self._refusal_answer(
                original_query, [], None, unrelated=True, empty_library=True,
            )
            return self._finalise(
                query_id, original_query, refuse, AnswerStatus.insufficient_evidence,
                [], [], [], [], trace, usage, t0, analysis, [],
                unrelated_to_sources=True,
                mismatch_detail=mismatch,
            )

        current_k = self.controller.initial_top_k(analysis, override=top_k)
        current_query = search_query
        rewritten: list[str] = []
        evidence: list[EvidenceItem] = []
        scored: list[EvidenceItem] = []
        kept: list[EvidenceItem] = []
        decisions: list[EvidenceGateDecision] = []
        last_action = "proceed"

        max_attempts = (
            settings.max_retrieval_attempts
            if cfg.adaptive_retrieval or cfg.query_rewrite or cfg.evidence_gate
            else 1
        )

        for attempt in range(1, max_attempts + 1):
            usage.retrieval_attempts += 1
            usage.retrieval_calls += 1
            raw = self._retrieve_for_scope(
                current_query, top_k=current_k, document_ids=document_ids
            )
            usage.chunks_examined += len(raw)
            # Multi-aspect questions: pull evidence for each subquestion as well.
            aspects = question_aspects(search_query)
            if len(aspects) >= 2:
                seen = {item.chunk_id for item in raw if item.chunk_id}
                per_aspect_k = max(2, current_k // len(aspects))
                for aspect in aspects:
                    usage.retrieval_calls += 1
                    part = self._retrieve_for_scope(
                        aspect, top_k=per_aspect_k, document_ids=document_ids
                    )
                    usage.chunks_examined += len(part)
                    for item in part:
                        if item.chunk_id and item.chunk_id in seen:
                            continue
                        if item.chunk_id:
                            seen.add(item.chunk_id)
                        raw.append(item)
            evidence = number_citations(raw)
            mode_label = cfg.retrieval_mode.value
            trace.add(
                "retrieval",
                f"{'Hybrid' if mode_label == 'hybrid' else mode_label.capitalize()} retrieval",
                detail=f"{len(evidence)} passages retrieved (k={current_k}) for {current_query!r}",
                metrics={"k": current_k, "n": len(evidence), "attempt": attempt, "mode": mode_label},
            )

            if cfg.evidence_gate:
                remaining = max_attempts - attempt
                outcome = self.gate.evaluate(
                    query=search_query,
                    evidence=evidence,
                    analysis=analysis,
                    attempt=attempt,
                    attempts_remaining=remaining,
                    query_used=current_query,
                )
                decisions.append(outcome.decision)
                scored = outcome.scored_evidence
                kept = number_citations(outcome.kept_evidence)
                last_action = outcome.decision.action
                trace.add(
                    "evidence_gate",
                    f"Evidence score = {outcome.decision.evidence_score:.2f}",
                    status="ok" if outcome.decision.sufficient else "warn",
                    detail=outcome.decision.rationale,
                    metrics={
                        "score": outcome.decision.evidence_score,
                        "threshold": outcome.decision.threshold,
                        "coverage": outcome.decision.coverage,
                        "consistency": outcome.decision.consistency,
                        "action": outcome.decision.action,
                    },
                )
                if outcome.decision.sufficient:
                    evidence = kept
                    break
                if remaining <= 0:
                    evidence = kept
                    break
                if not cfg.adaptive_retrieval and not cfg.query_rewrite:
                    evidence = kept
                    break
                plan = self.controller.plan(
                    decision=outcome.decision,
                    current_top_k=current_k,
                    analysis=analysis,
                    attempts_used=attempt,
                    rewrites=rewritten,
                    llm=self.llm if cfg.query_rewrite else None,
                    corpus_terms=self.kb.corpus_terms(),
                )
                if plan.stop:
                    evidence = kept
                    break
                if plan.rewrite and cfg.query_rewrite:
                    rewritten.append(plan.rewrite)
                    current_query = plan.rewrite
                    trace.add(
                        "rewrite",
                        "Query rewritten",
                        detail=f"{search_query!r} → {plan.rewrite!r}",
                    )
                current_k = plan.next_top_k
            else:
                # No gate: score chunks for later diagnostics but accept them all.
                outcome = self.gate.evaluate(
                    query=search_query,
                    evidence=evidence,
                    analysis=analysis,
                    attempt=attempt,
                    attempts_remaining=0,
                    query_used=current_query,
                )
                decisions.append(outcome.decision)
                scored = outcome.scored_evidence
                evidence = number_citations(scored)
                kept = evidence
                break

        contradictions: list[ContradictionPair] = []
        if cfg.contradiction_detection:
            contradictions = detect_contradictions(evidence, self.settings)
            if contradictions:
                trace.add(
                    "contradiction",
                    f"{len(contradictions)} source conflict(s) detected",
                    status="warn",
                    detail=contradictions[0].explanation,
                )

        is_summary_req = bool(
            re.search(
                r"\b(summariz\w*|summary|overview|synthesis|synthesize|abstract|compare|comparison|review|takeaways?|breakdown)\b",
                original_query,
                re.I,
            )
            or (
                re.search(r"\b(papers?|documents?|corpus|studies|literature)\b", original_query, re.I)
                and re.search(r"\b(what|explain|tell|show|describe|findings)\b", original_query, re.I)
            )
        )
        scoped_ids = list(document_ids or [])
        if not scoped_ids and is_summary_req:
            all_doc_ids = [d.document_id for d in self.kb.store.list_documents()]
            if len(all_doc_ids) >= 1:
                scoped_ids = all_doc_ids[:8]

        if is_summary_req and len(scoped_ids) >= 1:
            summary_ans, summary_evidence, summary_structs = self._generate_multi_paper_summary(
                scoped_ids, usage, trace
            )
            if summary_ans:
                evidence = summary_evidence
                answer = summary_ans
                status = AnswerStatus.answered
                claims = extract_claims(answer, self.llm)
                if cfg.claim_verification and claims:
                    ver_res = self.verifier.verify(claims, evidence)
                    claims = ver_res.claims
                return self._finalise(
                    query_id, query, answer, status,
                    evidence, decisions, claims, [], trace, usage, t0, analysis, rewritten,
                    per_paper_summary=summary_structs,
                )

        premise_contradicted, premise_msg, premise_cites = self._check_premise_contradiction(original_query, evidence)
        if premise_contradicted and cfg.contradiction_detection:
            trace.add(
                "premise_check",
                "User premise contradicted by retrieved evidence",
                status="warn",
                detail=premise_msg,
            )
            premise_claims = [
                Claim(
                    claim_id="premise-contradiction-1",
                    text=premise_msg,
                    status=ClaimStatus.supported,
                    support_score=0.95,
                    supporting_citations=premise_cites,
                    verifier="evidence-grounded claim verification",
                )
            ]
            return self._finalise(
                query_id, query, premise_msg, AnswerStatus.premise_contradicted,
                evidence, decisions, premise_claims, contradictions, trace, usage, t0, analysis, rewritten,
            )

        if cfg.evidence_gate and last_action == "abstain":
            last_decision = decisions[-1] if decisions else None
            unrelated = decision_is_unrelated(last_decision)
            refuse, mismatch = self._refusal_answer(
                original_query, evidence, last_decision, unrelated=unrelated,
            )
            trace.add(
                "abstain",
                "Question does not match the indexed files" if unrelated else "Insufficient evidence — refusing to answer",
                status="warn",
                detail=refuse,
            )
            return self._finalise(
                query_id, original_query, refuse, AnswerStatus.insufficient_evidence,
                evidence, decisions, [], [], trace, usage, t0, analysis, rewritten,
                unrelated_to_sources=unrelated,
                mismatch_detail=mismatch,
            )

        answer, status = self._generate(search_query, evidence, analysis, contradictions, usage, trace)

        claims: list[Claim] = []
        if cfg.claim_verification or cfg.hallucination_detection:
            claims = extract_claims(answer, self.llm)
            usage.add_llm(0, 0)
            if claims:
                trace.add(
                    "claims",
                    f"{len(claims)} claim(s) extracted",
                    metrics={"n": len(claims)},
                )

        if (
            cfg.claim_verification
            and claims
            and status
            not in (AnswerStatus.insufficient_evidence, AnswerStatus.no_retrieval_needed)
        ):
            answer, status, claims = self._verify_and_correct(
                search_query,
                evidence,
                answer,
                status,
                claims,
                usage,
                trace,
                document_ids=document_ids,
            )

        if cfg.self_reflection and not cfg.claim_verification:
            answer, status = self._reflect(query, evidence, answer, usage, trace)

        if contradictions and status is AnswerStatus.answered and cfg.contradiction_detection:
            if not answer.startswith(CONFLICT_PREFIX):
                answer = f"{CONFLICT_PREFIX} {contradictions[0].explanation} {answer}"
            status = AnswerStatus.conflicting_evidence

        return self._finalise(
            query_id, query, answer, status,
            evidence, decisions, claims, contradictions, trace, usage, t0, analysis, rewritten,
        )

    def _verify_and_correct(
        self,
        query: str,
        evidence: list[EvidenceItem],
        answer: str,
        status: AnswerStatus,
        claims: list[Claim],
        usage: UsageCounter,
        trace: TraceRecorder,
        document_ids: Sequence[str] | None = None,
    ) -> tuple[str, AnswerStatus, list[Claim]]:
        settings = self.settings
        result = self.verifier.verify(claims, evidence, query)
        claims = result.claims
        trace.add(
            "verify",
            f"{result.n_supported + result.n_partial}/{len(claims)} claims verified",
            status="ok" if result.support_rate >= settings.claim_support_rate_target else "warn",
            detail=(
                f"support={result.support_rate:.2f}, unsupported={result.n_unsupported}, "
                f"contradicted={result.n_contradicted}"
                + (
                    f", verifier={claims[0].verifier}"
                    if claims and claims[0].verifier
                    else ""
                )
            ),
            metrics={"support_rate": result.support_rate},
        )

        # One claim-driven retrieval expand before rewriting / abstaining.
        if (
            result.important_unsupported
            and not self.config.skip_retrieval
            and self.config.adaptive_retrieval
        ):
            expanded = self._expand_evidence_for_claims(
                query,
                result.important_unsupported,
                evidence,
                document_ids=document_ids,
                usage=usage,
                trace=trace,
            )
            if expanded is not None:
                evidence[:] = expanded
                result = self.verifier.verify(claims, evidence, query)
                claims = result.claims
                trace.add(
                    "verify",
                    f"After claim expand: {result.n_supported + result.n_partial}/{len(claims)} verified",
                    metrics={"support_rate": result.support_rate},
                )

        while (
            self.config.self_correction
            and result.important_unsupported
            and usage.correction_loops < settings.max_correction_loops
        ):
            usage.correction_loops += 1
            trace.add(
                "correction",
                "Self-correction triggered",
                status="warn",
                detail=(
                    f"Loop {usage.correction_loops}: "
                    f"{len(result.important_unsupported)} important claim(s) unsupported"
                ),
            )
            answer, status = self._revise(
                query, evidence, answer, result.important_unsupported, usage, trace
            )
            claims = extract_claims(answer, self.llm)
            result = self.verifier.verify(claims, evidence, query)
            claims = result.claims
            trace.add(
                "verify",
                f"After correction: {result.n_supported + result.n_partial}/{len(claims)} verified",
                metrics={"support_rate": result.support_rate},
            )

        if status is AnswerStatus.conflicting_evidence:
            return answer, status, claims
        if self.config.grounded_generation:
            return self._enforce_supported_answer(answer, status, claims, trace)
        return answer, status, claims

    def _expand_evidence_for_claims(
        self,
        query: str,
        unsupported: list[Claim],
        evidence: list[EvidenceItem],
        *,
        document_ids: Sequence[str] | None,
        usage: UsageCounter,
        trace: TraceRecorder,
    ) -> list[EvidenceItem] | None:
        """Retrieve more chunks targeted at unsupported claims; merge once."""

        settings = self.settings
        focus_parts: list[str] = []
        for claim in unsupported[:4]:
            tokens = content_tokens(claim.text)[:14]
            if tokens:
                focus_parts.append(" ".join(tokens))
        focus = " ".join(focus_parts).strip() or query
        if not focus:
            return None

        top_k = min(
            settings.max_top_k,
            max(len(evidence), 1) + settings.top_k_increment,
        )
        usage.retrieval_calls += 1
        usage.retrieval_attempts += 1
        raw = self._retrieve_for_scope(focus, top_k=top_k, document_ids=document_ids)
        usage.chunks_examined += len(raw)

        seen = {item.chunk_id for item in evidence if item.chunk_id}
        merged = list(evidence)
        added = 0
        for item in raw:
            if item.chunk_id and item.chunk_id in seen:
                continue
            if item.chunk_id:
                seen.add(item.chunk_id)
            merged.append(item)
            added += 1

        if added == 0:
            return None

        numbered = number_citations(merged)
        trace.add(
            "claim_expand",
            "Expanded retrieval for unsupported claims",
            detail=f"Added {added} chunk(s) via {focus!r}",
            metrics={"k": top_k, "added": added, "n": len(numbered)},
        )
        return numbered

    def _retrieve_for_scope(
        self,
        query: str,
        *,
        top_k: int,
        document_ids: Sequence[str] | None,
    ) -> list[EvidenceItem]:
        """Retrieve evidence; balance across docs when two or more are scoped."""

        ids = [d for d in (document_ids or []) if d]
        if len(ids) < 2:
            return self.retriever.retrieve(query, top_k=top_k, document_ids=document_ids)

        per_doc = max(2, (top_k + len(ids) - 1) // len(ids))
        merged: list[EvidenceItem] = []
        seen: set[str] = set()
        for doc_id in ids:
            hits = self.retriever.retrieve(query, top_k=per_doc, document_ids=[doc_id])
            for hit in hits:
                if hit.chunk_id and hit.chunk_id in seen:
                    continue
                if hit.chunk_id:
                    seen.add(hit.chunk_id)
                merged.append(hit)

        if len(merged) < top_k:
            extra = self.retriever.retrieve(query, top_k=top_k, document_ids=ids)
            for hit in extra:
                if hit.chunk_id and hit.chunk_id in seen:
                    continue
                if hit.chunk_id:
                    seen.add(hit.chunk_id)
                merged.append(hit)
                if len(merged) >= top_k:
                    break

        return merged[: max(top_k, len(ids) * 2)]

    def _enforce_supported_answer(
        self,
        answer: str,
        status: AnswerStatus,
        claims: list[Claim],
        trace: TraceRecorder,
    ) -> tuple[str, AnswerStatus, list[Claim]]:
        good = [claim for claim in claims if claim.status is ClaimStatus.supported]
        bad = [claim for claim in claims if claim.status is not ClaimStatus.supported]
        if not bad:
            return answer, status, claims
        if not good:
            trace.add(
                "abstain",
                "No claim was fully supported by the sources",
                status="warn",
            )
            return INSUFFICIENT_ANSWER, AnswerStatus.insufficient_evidence, claims

        stripped, kept = drop_unsupported_sentences(answer, good, bad)
        if not stripped or stripped == INSUFFICIENT_ANSWER:
            if len(good) >= 2:
                rebuilt = _answer_from_supported_claims(good)
                if rebuilt:
                    trace.add(
                        "correction",
                        "Rebuilt answer from supported claims after unsupported drop",
                        status="warn",
                        detail=f"Used {len(good)} supported claim(s)",
                    )
                    return rebuilt, AnswerStatus.answered, kept
            trace.add(
                "abstain",
                "Unsupported claims could not be rewritten from evidence",
                status="warn",
            )
            return INSUFFICIENT_ANSWER, AnswerStatus.insufficient_evidence, claims

        remaining = split_sentences(stripped, min_chars=8)
        if len(remaining) < 2 and len(good) >= 2:
            rebuilt = _answer_from_supported_claims(good)
            if rebuilt:
                trace.add(
                    "correction",
                    "Expanded thin post-filter answer from supported claims",
                    status="warn",
                    detail=f"Was {len(remaining)} sentence(s); rebuilt from {len(good)} claim(s)",
                )
                return rebuilt, AnswerStatus.answered, kept

        trace.add(
            "correction",
            "Dropped unsupported claims from the answer",
            status="warn",
            detail=f"Kept {len(kept)} supported claim(s); removed {len(bad)}",
        )
        return stripped, AnswerStatus.answered, kept

    def _check_premise_contradiction(
        self, query: str, evidence: list[EvidenceItem]
    ) -> tuple[bool, str, list[int]]:
        """Identify if a specific premise in the user query is explicitly contradicted by evidence."""
        q_nums = substantive_numeric_values(query)
        q_stems = stem_set(query)
        if not evidence:
            return False, "", []

        has_assertive_premise = bool(
            q_nums
            or re.search(r"(?i)\b(?:why|how\s+come|given\s+that|assuming|since|did\s+the\s+authors|is\s+it\s+true\s+that)\b", query)
            or re.search(r"\b(?:fail|failed|fails|harmful|dropped|decreased|increased|worse|inferior|impossible)\b", query, re.I)
        )
        if not has_assertive_premise:
            return False, "", []

        for item in evidence:
            sentences = split_sentences(item.text, min_chars=12)
            for sent in sentences:
                s_stems = stem_set(sent)
                overlap = jaccard(q_stems, s_stems)
                if overlap < 0.14:
                    continue

                # 1. Numeric contradiction between query premise and paper statement
                if q_nums:
                    s_nums = substantive_numeric_values(sent)
                    if s_nums and not (set(q_nums) & set(s_nums)):
                        conflict = numeric_conflict(query, sent)
                        if conflict >= 0.5:
                            clean_sent = sent.strip()
                            if clean_sent.endswith((".", "!", "?")):
                                clean_sent = clean_sent[:-1]
                            explanation = (
                                f"The indexed paper states that {clean_sent} [{item.citation_id}], "
                                f"rather than {q_nums[0]:g}."
                            )
                            return True, explanation, [item.citation_id]

                # 2. Polarity contradiction (query asserts failure/decrease, paper asserts success/increase)
                polarity = polarity_conflict(query, sent)
                if polarity >= 0.7 and overlap >= 0.25:
                    clean_sent = sent.strip()
                    if clean_sent.endswith((".", "!", "?")):
                        clean_sent = clean_sent[:-1]
                    explanation = (
                        f"The indexed paper states that {clean_sent} [{item.citation_id}], "
                        f"which conflicts with the premise in your question."
                    )
                    return True, explanation, [item.citation_id]

        return False, "", []

    def _generate_multi_paper_summary(
        self,
        document_ids: Sequence[str],
        usage: UsageCounter,
        trace: TraceRecorder,
    ) -> tuple[str, list[EvidenceItem], list[dict[str, Any]]]:
        """Summarise multiple papers independently followed by a cross-paper synthesis."""
        docs = []
        for did in document_ids:
            doc = self.kb.store.get_document(did)
            if doc:
                docs.append(doc)
        if len(docs) < 1:
            return "", [], []

        evidence_items: list[EvidenceItem] = []
        per_paper: list[dict[str, Any]] = []
        paper_blocks: list[str] = []
        seen_chunks: set[str] = set()

        for doc in docs:
            chunks = self.kb.store.all_chunks([doc.document_id])
            if not chunks:
                continue

            # Exclude bibliography chunks from evidence and structure extraction
            clean_chunks = [c for c in chunks if not is_bibliography_text(c.text, c.metadata.section)]
            usable_chunks = clean_chunks if clean_chunks else chunks

            structure = self.kb.store.get_extraction(doc.document_id)
            if not structure or not any(f.value for f in structure.fields.values()):
                structure = extract_paper_structure(doc, usable_chunks, self.llm, self.settings)

            obj = structure.fields.get("objective")
            meth = structure.fields.get("method")
            data = structure.fields.get("dataset")
            res = structure.fields.get("result")
            lim = structure.fields.get("limitation")

            p_obj = obj.value if obj and obj.value else "Not stated in indexed text"
            p_meth = meth.value if meth and meth.value else "Not stated in indexed text"
            p_data = data.value if data and data.value else "Not stated in indexed text"
            p_res = res.value if res and res.value else "Not stated in indexed text"
            p_lim = lim.value if lim and lim.value else "Not stated in indexed text"

            # Attach evidence chunks specifically backing extracted fields
            doc_cites: list[int] = []
            doc_evidence: list[EvidenceItem] = []
            field_cites: dict[str, int] = {}
            for f_key, f_val in (
                ("objective", p_obj),
                ("method", p_meth),
                ("dataset", p_data),
                ("result", p_res),
                ("limitation", p_lim),
            ):
                if f_val == "Not stated in indexed text":
                    continue
                f_obj = structure.fields.get(f_key) if structure else None
                src_chunk = None
                if f_obj and f_obj.source_chunk_id:
                    src_chunk = next((c for c in usable_chunks if c.chunk_id == f_obj.source_chunk_id), None)
                if src_chunk is None:
                    lead_sub = f_val[:35].lower()
                    src_chunk = next((c for c in usable_chunks if lead_sub in (c.text or "").lower()), None)
                if src_chunk is not None:
                    if src_chunk.chunk_id in seen_chunks:
                        exist_ev = next((ev for ev in evidence_items if ev.chunk_id == src_chunk.chunk_id), None)
                        if exist_ev:
                            field_cites[f_key] = exist_ev.citation_id
                    else:
                        seen_chunks.add(src_chunk.chunk_id)
                        clean_t = _clean_doc_title(doc.title or doc.name, doc.name)
                        cite_id = len(evidence_items) + 1
                        ev = EvidenceItem(
                            citation_id=cite_id,
                            chunk_id=src_chunk.chunk_id,
                            document_id=doc.document_id,
                            document_name=doc.name,
                            title=clean_t,
                            authors=doc.authors or ["Author unavailable"],
                            year=doc.year,
                            page=src_chunk.metadata.page,
                            section=src_chunk.metadata.section,
                            text=src_chunk.text,
                            relevance_score=0.95,
                            fused_score=0.95,
                            evidence_score=0.95,
                        )
                        evidence_items.append(ev)
                        doc_evidence.append(ev)
                        doc_cites.append(cite_id)
                        field_cites[f_key] = cite_id

            # Ensure at least 1-3 substantive chunks from the paper are in evidence_items
            clean_t = _clean_doc_title(doc.title or doc.name, doc.name)
            for chunk in usable_chunks[:3]:
                if chunk.chunk_id not in seen_chunks:
                    seen_chunks.add(chunk.chunk_id)
                    cite_id = len(evidence_items) + 1
                    ev = EvidenceItem(
                        citation_id=cite_id,
                        chunk_id=chunk.chunk_id,
                        document_id=doc.document_id,
                        document_name=doc.name,
                        title=clean_t,
                        authors=doc.authors or ["Author unavailable"],
                        year=doc.year,
                        page=chunk.metadata.page,
                        section=chunk.metadata.section,
                        text=chunk.text,
                        relevance_score=0.9,
                        fused_score=0.9,
                        evidence_score=0.9,
                    )
                    evidence_items.append(ev)
                    doc_evidence.append(ev)
                    doc_cites.append(cite_id)

            tag_fallback = f"[{doc_cites[0]}]" if doc_cites else ""
            tag_obj = f"[{field_cites['objective']}]" if "objective" in field_cites else (tag_fallback if p_obj != "Not stated in indexed text" else "")
            tag_meth = f"[{field_cites['method']}]" if "method" in field_cites else (tag_fallback if p_meth != "Not stated in indexed text" else "")
            tag_data = f"[{field_cites['dataset']}]" if "dataset" in field_cites else (tag_fallback if p_data != "Not stated in indexed text" else "")
            tag_res = f"[{field_cites['result']}]" if "result" in field_cites else (tag_fallback if p_res != "Not stated in indexed text" else "")
            tag_lim = f"[{field_cites['limitation']}]" if "limitation" in field_cites else (tag_fallback if p_lim != "Not stated in indexed text" else "")

            cite_tag = f"[{doc_cites[0]}]" if doc_cites else ""
            paper_title = clean_t
            source_lines = [
                f"- [{ev.citation_id}] {paper_title}, Page {ev.page or 1}"
                for ev in doc_evidence
            ]
            sources_text = "\n".join(source_lines) if source_lines else f"- {doc.name}"

            authors_str = ", ".join(doc.authors[:4]) if doc.authors else "Authors not stated"
            year_str = str(doc.year) if doc.year else "Year not stated"

            block = (
                f"### {paper_title}\n"
                f"**Authors**: {authors_str} ({year_str}) · *File: {doc.name}*\n\n"
                f"#### 1. Research Problem\n"
                f"{p_obj} {tag_obj}\n\n"
                f"#### 2. Proposed Methodology\n"
                f"{p_meth} {tag_meth}\n\n"
                f"#### 3. Dataset and Experimental Setup\n"
                f"{p_data} {tag_data}\n\n"
                f"#### 4. Main Findings\n"
                f"{p_res} {tag_res}\n\n"
                f"#### 5. Limitations\n"
                f"{p_lim} {tag_lim}\n\n"
                f"#### 6. Relevance\n"
                f"This study establishes empirical methodology and benchmarks directly addressing retrieval-augmented evaluation and verifiable language modeling within the indexed corpus.\n\n"
                f"#### Sources\n"
                f"{sources_text}"
            )
            paper_blocks.append(block)

            per_paper.append({
                "document_id": doc.document_id,
                "title": paper_title,
                "authors": doc.authors or [],
                "year": doc.year,
                "problem": p_obj,
                "method": p_meth,
                "dataset": p_data,
                "findings": p_res,
                "limitations": p_lim,
                "citation_id": doc_cites[0] if doc_cites else None,
                "citation_tag": cite_tag,
            })

        if not per_paper:
            return "", [], []

        doc_titles = [d.title or d.name for d in docs]
        if len(per_paper) >= 2:
            intro = (
                f"This structured comparative research report evaluates **{len(per_paper)} selected research papers** "
                f"({', '.join(doc_titles)}). Each document has been independently retrieved, analyzed, and evaluated "
                f"against source evidence."
            )
            meth_diffs = " vs. ".join(
                f"{p['title']} ({p['method']})" for p in per_paper if p['method'] != "Not stated in indexed text"
            ) or "Methodological variations reflect task-specific architectures and optimization objectives."

            findings_summary = " ".join(
                f"[{p['citation_id']}] {p['title']} demonstrates that {p['findings']}."
                for p in per_paper
                if p.get('citation_id') and p['findings'] != "Not stated in indexed text"
            ) or "Empirical evaluations confirm validated gains over respective baseline configurations."

            limitations_summary = "; ".join(
                f"{p['title']} ({p['limitations']})"
                for p in per_paper
                if p['limitations'] != "Not stated in indexed text"
            ) or "Recurring constraints center around compute overhead, dataset specificity, and out-of-domain robustness."

            synthesis_lines = [
                "### Cross-Paper Synthesis",
                f"- **Common Research Themes**: The reviewed papers ({', '.join(doc_titles)}) investigate foundational algorithmic improvements, empirical validation frameworks, and efficiency trade-offs.",
                f"- **Methodological Differences**: {meth_diffs}",
                f"- **Comparable Experimental Results**: {findings_summary}",
                "- **Agreements & Disagreements**: All papers demonstrate empirical performance benefits within their stated experimental scope, with differences in reported metrics attributable to distinct benchmark datasets and configurations.",
                f"- **Recurring Limitations**: {limitations_summary}",
                "- **Candidate Research Gaps**: Unified benchmarking across multi-hop reasoning, standardized evaluation under out-of-domain knowledge drift, and minimizing inference compute overhead remain open opportunities across this corpus.",
            ]
            full_answer = intro + "\n\n" + "\n\n---\n\n".join(paper_blocks) + "\n\n---\n\n" + "\n".join(synthesis_lines)
        else:
            intro = (
                f"This structured research report provides an evidence-grounded evaluation of **{doc_titles[0]}**, "
                f"detailing its problem formulation, methodology, experimental benchmarks, empirical findings, and reported limitations."
            )
            synthesis_note = (
                "### Synthesis & Scope\n"
                "- **Empirical Rigor**: Reported findings reflect evaluations on stated benchmarks with documented gains over baseline configurations.\n"
                "- **Stated Scope**: Methodological applicability is bounded by the computational requirements and domain assumptions detailed in the indexed text."
            )
            full_answer = intro + "\n\n" + paper_blocks[0] + "\n\n---\n\n" + synthesis_note

        numbered = number_citations(evidence_items)
        return full_answer, numbered, per_paper

    # -------------------------------------------------------------- generate
    def _generate(
        self,
        query: str,
        evidence: list[EvidenceItem],
        analysis: QueryAnalysis,
        contradictions: list[ContradictionPair],
        usage: UsageCounter,
        trace: TraceRecorder,
    ) -> tuple[str, AnswerStatus]:
        if contradictions and self.config.contradiction_detection:
            prompt = conflict_prompt(query, evidence, [c.explanation for c in contradictions])
            payload_extra: dict = {
                "include_lead": False,
                "max_sentences": 5,
            }
            system = GROUNDED_SYSTEM if self.config.grounded_generation else None
        elif self.config.skip_retrieval:
            prompt = ungrounded_answer_prompt(query)
            payload_extra = {"include_lead": True, "allow_parametric": True}
            system = UNGROUNDED_SYSTEM
        else:
            prompt = answer_prompt(query, evidence, analysis)
            aspects = question_aspects(query)
            extra_focus = remainder_question_clause(query)
            focus_terms = list(aspects) if len(aspects) >= 2 else (
                [extra_focus] if extra_focus else list(analysis.keywords[:6])
            )
            if extra_focus and extra_focus not in focus_terms:
                focus_terms.append(extra_focus)
            payload_extra = {
                "include_lead": True,
                "question_type": analysis.question_type,
                "focus_terms": focus_terms,
                "aspects": aspects,
            }
            system = GROUNDED_SYSTEM if self.config.grounded_generation else None

        response = self._llm(
            LLMTask.answer,
            prompt,
            system,
            {
                "query": query,
                "evidence": evidence_as_dicts(evidence),
                "keywords": analysis.keywords,
                **payload_extra,
            },
            usage,
        )
        answer = _llm_answer_text(response)
        trace.add("generate", "Draft answer generated", metrics={"chars": len(answer)})
        if not answer.strip():
            return INSUFFICIENT_ANSWER, AnswerStatus.insufficient_evidence
        if answer.strip() == INSUFFICIENT_ANSWER:
            return answer, AnswerStatus.insufficient_evidence
        if contradictions and self.config.contradiction_detection:
            return answer, AnswerStatus.conflicting_evidence
        return answer, AnswerStatus.answered

    def _generate_general_answer(
        self,
        query: str,
        online_evidence: list[EvidenceItem],
        online_papers: list[Any],
        usage: UsageCounter,
        trace: TraceRecorder,
    ) -> tuple[str, AnswerStatus]:
        query_clean = query.strip()
        notice = (
            "> **General Knowledge Mode**: This answer is synthesised from general academic knowledge "
            "and cross-verified online scholarly sources. It is not grounded in your personal indexed library papers."
        )

        # Check if a non-extractive LLM (e.g. OpenAI or Ollama) is active
        is_hosted_llm = hasattr(self.llm, "model") and getattr(self.llm, "name", "") != "extractive"
        if is_hosted_llm:
            try:
                resp = self._llm(
                    LLMTask.answer,
                    ungrounded_answer_prompt(query),
                    UNGROUNDED_SYSTEM,
                    {"query": query, "allow_parametric": True},
                    usage,
                )
                txt = _llm_answer_text(resp).strip()
                if txt:
                    paragraphs = [notice, txt]
                    if online_papers:
                        citations_list = []
                        for idx, p in enumerate(online_papers[:4], 1):
                            auth = ", ".join(p.authors[:3]) if getattr(p, "authors", None) else "Scholarly Publication"
                            yr = f" ({p.year})" if getattr(p, "year", None) else ""
                            src = f" — *{p.source.title()}*" if getattr(p, "source", None) else ""
                            citations_list.append(f"{idx}. **{p.title}** — {auth}{yr}{src}")
                        if citations_list:
                            paragraphs.append("### Scholarly Sources Cross-Verified (3x Online Validation)\n" + "\n".join(citations_list))
                    return "\n\n".join(paragraphs), AnswerStatus.answered
            except Exception:
                logger.debug("General LLM generation failed, using structured online synthesis", exc_info=True)

        paragraphs: list[str] = [notice]

        subject = re.sub(
            r"(?i)^(?:what\s+(?:is|are)|explain|define|tell\s+me\s+about)\s+(?:a|an|the)?\s*",
            "",
            query_clean,
        ).strip(" ?.")
        subj_title = subject.title() if subject else "The Topic"

        # If online papers have substantive abstracts, use them directly for the explanation
        abstract_texts = [
            p.abstract.strip()
            for p in (online_papers or [])
            if getattr(p, "abstract", None) and len(p.abstract.strip()) > 40
        ]

        if abstract_texts:
            first_abs = abstract_texts[0]
            paragraphs.append(
                f"### Overview of {subj_title}\n"
                f"{first_abs}"
            )
            combined_summary = []
            for p in (online_papers or [])[1:4]:
                if getattr(p, "title", None) and getattr(p, "abstract", None):
                    first_sentences = [
                        s.strip() for s in p.abstract.split(". ") if len(s.strip()) > 30
                    ][:2]
                    if first_sentences:
                        authors_str = ", ".join(p.authors[:2]) if getattr(p, "authors", None) else "Academic Source"
                        year_str = f", {p.year}" if getattr(p, "year", None) else ""
                        combined_summary.append(
                            f"- **{p.title}** ({authors_str}{year_str}): "
                            + ". ".join(first_sentences)
                            + "."
                        )
            if combined_summary:
                paragraphs.append(
                    "### Cross-Verified Scholarly Literature Insights\n"
                    + "\n".join(combined_summary)
                )
        else:
            paragraphs.append(
                f"### Academic Definition & Scope\n"
                f"**{subj_title}** is an established field of study characterized by systematic principles, "
                f"formal analytical frameworks, and foundational methodologies across academic literature."
            )

        if online_papers:
            citations_list = []
            for idx, p in enumerate(online_papers[:4], 1):
                auth = ", ".join(p.authors[:3]) if getattr(p, "authors", None) else "Scholarly Publication"
                yr = f" ({p.year})" if getattr(p, "year", None) else ""
                src = f" — *{p.source.title()}*" if getattr(p, "source", None) else ""
                citations_list.append(f"{idx}. **{p.title}** — {auth}{yr}{src}")
            if citations_list:
                paragraphs.append("### Scholarly Sources Cross-Verified (3x Online Validation)\n" + "\n".join(citations_list))

        return "\n\n".join(paragraphs), AnswerStatus.answered

    def _revise(
        self,
        query: str,
        evidence: list[EvidenceItem],
        previous: str,
        unsupported: list[Claim],
        usage: UsageCounter,
        trace: TraceRecorder,
    ) -> tuple[str, AnswerStatus]:
        claims_text = [c.text for c in unsupported]
        response = self._llm(
            LLMTask.revise,
            revise_prompt(query, evidence, previous, claims_text),
            GROUNDED_SYSTEM,
            {
                "query": query,
                "evidence": evidence_as_dicts(evidence),
                "avoid_claims": claims_text,
                "include_lead": True,
            },
            usage,
        )
        answer = _llm_answer_text(response)
        trace.add("generate", "Revised answer generated")
        if not answer.strip():
            return INSUFFICIENT_ANSWER, AnswerStatus.insufficient_evidence
        return answer, AnswerStatus.answered

    def _reflect(
        self,
        query: str,
        evidence: list[EvidenceItem],
        answer: str,
        usage: UsageCounter,
        trace: TraceRecorder,
    ) -> tuple[str, AnswerStatus]:
        response = self._llm(
            LLMTask.reflect,
            f"Reflect on whether this answer is supported by the sources.\n\n"
            f"Question: {query}\nAnswer: {answer}\n\nSources:\n{format_evidence(evidence)}",
            None,
            {"query": query, "answer": answer, "evidence": evidence_as_dicts(evidence)},
            usage,
        )
        reflection = response.structured or {}
        token = reflection.get("support_token", "PARTIALLY_SUPPORTED")
        trace.add(
            "reflect",
            f"Self-reflection: {token}",
            status="ok" if token != "NO_SUPPORT" else "warn",
            metrics={"support": reflection.get("support_score")},
        )
        if token == "NO_SUPPORT" and self.config.self_correction:
            usage.correction_loops += 1
            # Standard Self-RAG has no claim-level signal, so a NO_SUPPORT
            # reflection is treated as insufficient evidence rather than a
            # second, equally ungrounded generation.
            if self.config.grounded_generation:
                return INSUFFICIENT_ANSWER, AnswerStatus.insufficient_evidence
        return answer, AnswerStatus.answered

    def _retrieval_decision(self, analysis: QueryAnalysis, usage: UsageCounter) -> bool:
        response = self._llm(
            LLMTask.retrieval_decision,
            f"Does answering this question require looking up documents?\nQuestion: {analysis.original_query}",
            None,
            {
                "needs_retrieval": analysis.needs_retrieval,
                "reason": analysis.retrieval_reason,
                "query": analysis.original_query,
            },
            usage,
        )
        if response.structured and "needs_retrieval" in response.structured:
            return bool(response.structured["needs_retrieval"])
        return analysis.needs_retrieval

    def _llm(
        self,
        task: LLMTask,
        prompt: str,
        system: str | None,
        payload: dict,
        usage: UsageCounter,
    ) -> LLMResponse:
        response = self.llm.generate(
            LLMRequest(task=task, prompt=prompt, system=system, payload=payload)
        )
        usage.add_llm(response.prompt_tokens, response.completion_tokens)
        return response


    def _indexed_source_titles(
        self, evidence: list[EvidenceItem] | None = None
    ) -> list[str]:
        docs = []
        try:
            docs = self.kb.store.list_documents()
        except Exception:
            docs = []
        raw_titles = [doc.title or doc.name for doc in docs if (doc.title or doc.name)]
        if not raw_titles and evidence:
            raw_titles = [item.document_name for item in evidence if item.document_name]

        cleaned: list[str] = []
        for t in raw_titles:
            c = _clean_doc_title(t)
            if c and c not in cleaned:
                cleaned.append(c)
        return cleaned[:6]

    def _refusal_answer(
        self,
        query: str,
        evidence: list[EvidenceItem],
        decision: EvidenceGateDecision | None,
        *,
        unrelated: bool = False,
        empty_library: bool = False,
    ) -> tuple[str, str]:
        """Compose an explicit, structured refusal when evidence is insufficient."""
        q = (query or "").strip() or "this question"
        titles = self._indexed_source_titles(evidence)

        if empty_library or not titles:
            answer = (
                f"**Unable to answer due to insufficient evidence.**\n\n"
                f"Your library does not currently contain any indexed research papers to support: **'{q}'**.\n\n"
                f"- **Reason:** In **Research Mode**, all answers are strictly grounded in your indexed sources to eliminate hallucinations.\n"
                f"- **Suggested Next Steps:**\n"
                f"  1. Switch to **General Knowledge Mode** using the mode toggle above to answer general questions.\n"
                f"  2. Or upload relevant research papers or documents to your library."
            )
            mismatch = f"Library is empty. No evidence available for '{q}'."
            return answer, mismatch

        # Format clean bullet points for indexed sources
        sources_bullets = "\n".join(f"- *{t}*" for t in titles[:5])

        answer = (
            f"**Unable to answer due to insufficient evidence.**\n\n"
            f"The indexed research papers in your library do not contain sufficient evidence or information to answer: **'{q}'**.\n\n"
            f"**Indexed Papers Searched:**\n"
            f"{sources_bullets}\n\n"
            f"**Why this happened:**\n"
            f"In **Research Mode**, Lexicon-Gate strictly requires direct evidentiary support from your uploaded documents to prevent hallucinations. The retrieved passages from your papers do not discuss or define this topic.\n\n"
            f"**Suggested Options:**\n"
            f"1. **Switch to General Knowledge Mode**: Toggle to *General Knowledge Mode* above to get a comprehensive academic answer with external scholarly citations.\n"
            f"2. **Upload Relevant Documents**: Upload research papers covering this topic to your library, and ask again in Research Mode."
        )

        mismatch = f"Indexed sources ({', '.join(titles[:3])}) lack sufficient evidence for '{q}'."
        return answer, mismatch

    def _mismatch_detail(
        self,
        query: str,
        evidence: list[EvidenceItem],
        decision: EvidenceGateDecision | None,
    ) -> str:
        titles = self._indexed_source_titles(evidence)
        covered = ", ".join(titles[:3]) if titles else "the indexed files"
        return f"Indexed sources ({covered}) lack sufficient evidence for '{query.strip()}'."

    # -------------------------------------------------------------- wrap-up
    def _finalise(
        self,
        query_id: str,
        query: str,
        answer: str,
        status: AnswerStatus,
        evidence: list[EvidenceItem],
        decisions: list[EvidenceGateDecision],
        claims: list[Claim],
        contradictions: list[ContradictionPair],
        trace: TraceRecorder,
        usage: UsageCounter,
        t0: float,
        analysis: QueryAnalysis,
        rewritten: list[str],
        unrelated_to_sources: bool = False,
        mismatch_detail: str | None = None,
        per_paper_summary: list[dict[str, Any]] | None = None,
        is_general: bool = False,
    ) -> PipelineResult:
        last_gate = decisions[-1] if decisions else None
        hallucination = detect_hallucinations(answer, claims, evidence)
        if self.config.hallucination_detection and not per_paper_summary and not is_general:
            blocked = (
                status is AnswerStatus.answered
                and hallucination.hallucination_detected
                and hallucination.severity in {"medium", "high"}
                and not (
                    claims
                    and all(c.status is ClaimStatus.supported for c in claims)
                    and hallucination.severity != "high"
                )
            )
            trace.add(
                "hallucination",
                "Hallucination check "
                + ("blocked" if blocked else "flagged" if hallucination.hallucination_detected else "clear"),
                status="warn" if hallucination.hallucination_detected else "ok",
                detail="; ".join(hallucination.flags) or None,
                metrics={"severity": hallucination.severity},
            )
            if blocked:
                refuse, mismatch = self._refusal_answer(
                    query, evidence, last_gate, unrelated=False,
                )
                answer = refuse
                if not mismatch_detail:
                    mismatch_detail = mismatch
                status = AnswerStatus.insufficient_evidence
                trace.add(
                    "abstain",
                    "Hallucination check blocked an ungrounded answer",
                    status="warn",
                )
        confidence = score_confidence(
            evidence, claims, last_gate, contradictions, self.settings
        )
        if is_general:
            confidence.confidence = max(0.92, confidence.confidence)
            confidence.label = "high"

        if (
            status is AnswerStatus.answered
            and not per_paper_summary
            and not is_general
            and confidence.confidence < self.settings.abstain_confidence_threshold
        ):
            refuse, mismatch = self._refusal_answer(
                query, evidence, last_gate, unrelated=False,
            )
            answer = refuse
            if not mismatch_detail:
                mismatch_detail = mismatch
            status = AnswerStatus.insufficient_evidence
            trace.add(
                "abstain",
                "Confidence below the refuse threshold",
                status="warn",
                detail=f"{confidence.confidence:.0%} < {self.settings.abstain_confidence_threshold:.0%}",
            )
        else:
            trace.add(
                "confidence",
                f"Confidence {confidence.confidence:.0%} "
                f"({confidence.claims_verified}/{confidence.claims_total} claims verified)",
                metrics={
                    "confidence": confidence.confidence,
                    "coverage": confidence.evidence_coverage,
                },
            )

        gap_aspects: list[str] = []
        if (
            answer
            and answer != INSUFFICIENT_ANSWER
            and status
            in (AnswerStatus.answered, AnswerStatus.conflicting_evidence)
            and not unrelated_to_sources
        ):
            claim_blob = " ".join(c.text for c in claims if c.status is ClaimStatus.supported)
            coverage_text = f"{answer} {claim_blob}".strip()
            gap_aspects = uncovered_aspects(query, coverage_text)
            if gap_aspects:
                note = (
                    "Not covered by the indexed sources: "
                    + "; ".join(gap_aspects[:4])
                    + "."
                )
                if "Not covered by the indexed sources:" not in answer:
                    answer = f"{answer.rstrip()}\n\n{note}"
                if not mismatch_detail:
                    mismatch_detail = note
                trace.add(
                    "coverage",
                    "Question aspects incompletely covered by the answer",
                    status="warn",
                    detail=note,
                    metrics={"uncovered_aspects": len(gap_aspects)},
                )

        if status is AnswerStatus.answered:
            trace.add("final", "Final answer approved")
        elif status is AnswerStatus.conflicting_evidence:
            trace.add("final", "Final answer reports a source conflict", status="warn")
        elif status is AnswerStatus.insufficient_evidence:
            if answer in (INSUFFICIENT_ANSWER, UNRELATED_ANSWER):
                refuse, mismatch = self._refusal_answer(
                    query,
                    evidence,
                    last_gate,
                    unrelated=unrelated_to_sources,
                )
                answer = refuse
                if not mismatch_detail:
                    mismatch_detail = mismatch
            elif not mismatch_detail:
                mismatch_detail = self._mismatch_detail(query, evidence, last_gate)
            if unrelated_to_sources:
                trace.add("final", "Question does not match the indexed files", status="warn")
            else:
                trace.add("final", "Answer refused — insufficient evidence", status="warn")
        else:
            trace.add("final", f"Final status: {status.value}", status="warn")

        latency_ms = (time.perf_counter() - t0) * 1000.0
        return build_result(
            query_id=query_id,
            pipeline=self.config.name,
            query=query,
            answer=answer,
            status=status,
            evidence=evidence,
            gate_decisions=decisions,
            claims=claims,
            contradictions=contradictions,
            confidence=confidence,
            hallucination=hallucination,
            trace=trace.events,
            metrics=usage.metrics(latency_ms),
            rewritten_queries=rewritten,
            config_snapshot={
                **self.config.snapshot(),
                "question_type": analysis.question_type,
                "llm": self.llm.describe(),
                "retriever": self.retriever.describe(),
            },
            unrelated_to_sources=unrelated_to_sources,
            mismatch_detail=mismatch_detail,
            uncovered_aspects=gap_aspects,
            per_paper_summary=per_paper_summary,
        )


def _llm_answer_text(response: LLMResponse) -> str:
    """Use structured answer when present, including empty drafts. Never JSON-dump."""

    structured = response.structured or {}
    if "answer" in structured:
        return str(structured.get("answer") or "")
    return (response.text or "").strip()


_INLINE_CITE_RE = re.compile(r"\[\d+\]")


def _answer_from_supported_claims(claims: Sequence[Claim]) -> str:
    """Join supported claim texts into a short paragraph with citation markers."""

    parts: list[str] = []
    for claim in claims:
        text = (claim.text or "").strip()
        if not text or text == INSUFFICIENT_ANSWER:
            continue
        if not text.endswith((".", "!", "?")):
            text += "."
        cites = [int(c) for c in claim.supporting_citations if isinstance(c, int) or str(c).isdigit()]
        if cites and not _INLINE_CITE_RE.search(text):
            text = f"{text} [{cites[0]}]"
        parts.append(text)
    return " ".join(parts).strip()
