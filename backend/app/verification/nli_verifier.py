"""NLI-based claim verification (cross-encoder entailment).

``verify_claim`` scores (claim, evidence chunk) with a pretrained MNLI-style
classifier and returns ``supported`` / ``unsupported`` / ``contradicted``.

The default checkpoint is DeBERTa-v3 xsmall fine-tuned for NLI. It runs on CPU.
If that checkpoint cannot be loaded, DistilBERT-MNLI is tried next.
"""

from __future__ import annotations

import logging
import math
import threading
from typing import Any, Callable, Sequence

from ..config import Settings, get_settings
from ..models.query import Claim, ClaimStatus, EvidenceItem
from ..text_utils import normalise_whitespace, split_sentences, stem_set, truncate
from .claim_verifier import ClaimVerificationResult, _is_important

logger = logging.getLogger(__name__)

# Primary: small DeBERTa-v3 NLI (MNLI). Fallback: DistilBERT MNLI for lighter CPU.
DEFAULT_NLI_MODEL = "cross-encoder/nli-deberta-v3-xsmall"
FALLBACK_NLI_MODEL = "typeform/distilbert-base-uncased-mnli"

_LABEL_SUPPORTED = "supported"
_LABEL_UNSUPPORTED = "unsupported"
_LABEL_CONTRADICTED = "contradicted"

_lock = threading.Lock()
_loaded: _LoadedNLI | None = None
_load_failed = False
# Tests inject (claim, chunk) -> (mnli_label, confidence)
_score_override: Callable[[str, str], tuple[str, float]] | None = None


class _LoadedNLI:
    __slots__ = ("name", "tokenizer", "model", "id2label", "device")

    def __init__(self, name: str, tokenizer: Any, model: Any, id2label: dict, device: str) -> None:
        self.name = name
        self.tokenizer = tokenizer
        self.model = model
        self.id2label = {int(k): str(v).lower() for k, v in id2label.items()}
        self.device = device


def set_nli_score_override(fn: Callable[[str, str], tuple[str, float]] | None) -> None:
    """Replace the classifier (tests). Pass ``None`` to restore the real model."""

    global _score_override
    _score_override = fn


def reset_nli_runtime() -> None:
    global _loaded, _load_failed, _score_override
    _loaded = None
    _load_failed = False
    _score_override = None


def nli_backend_name() -> str | None:
    return None if _loaded is None else _loaded.name


def verify_claim(claim: str, evidence_chunk: str) -> dict[str, Any]:
    """Score one claim against one evidence chunk.

    Returns ``{"label": "supported"|"unsupported"|"contradicted", "confidence": float}``.
    """

    mnli, confidence = _score_pair(claim or "", evidence_chunk or "")
    return {"label": _map_mnli_label(mnli), "confidence": float(confidence)}


def nli_verify(
    claims: list[Claim],
    evidence: list[EvidenceItem],
    query: str = "",
    settings: Settings | None = None,
) -> ClaimVerificationResult:
    """Fill each claim from the best (claim, chunk) NLI score among retrieved evidence."""

    settings = settings or get_settings()
    if not claims:
        return ClaimVerificationResult(
            claims=[],
            support_rate=1.0 if evidence else 0.0,
            n_supported=0,
            n_partial=0,
            n_unsupported=0,
            n_contradicted=0,
            important_unsupported=[],
        )

    query_stems = stem_set(query)
    for claim in claims:
        _verify_claim_against_evidence(claim, evidence, settings)

    n_supported = sum(1 for c in claims if c.status is ClaimStatus.supported)
    n_partial = sum(1 for c in claims if c.status is ClaimStatus.partially_supported)
    n_unsupported = sum(1 for c in claims if c.status is ClaimStatus.unsupported)
    n_contradicted = sum(1 for c in claims if c.status is ClaimStatus.contradicted)
    support_rate = (n_supported + 0.5 * n_partial) / len(claims) if claims else 0.0
    important = [
        c
        for c in claims
        if c.status in (ClaimStatus.unsupported, ClaimStatus.contradicted)
        and _is_important(c, query_stems)
    ]
    return ClaimVerificationResult(
        claims=claims,
        support_rate=round(support_rate, 4),
        n_supported=n_supported,
        n_partial=n_partial,
        n_unsupported=n_unsupported,
        n_contradicted=n_contradicted,
        important_unsupported=important,
    )


def _verify_claim_against_evidence(
    claim: Claim, evidence: Sequence[EvidenceItem], settings: Settings
) -> None:
    if not evidence:
        claim.status = ClaimStatus.unsupported
        claim.nli_label = _LABEL_UNSUPPORTED
        claim.nli_confidence = 0.0
        claim.verifier = "nli"
        claim.rationale = "No retrieved passages to verify against."
        return

    best_entail = (-1.0, None)
    best_contra = (-1.0, None)
    supporting: list[int] = []
    contradicting: list[int] = []
    hypothesis = _hypothesis_text(claim.text)

    for item in evidence:
        verdict = verify_claim(hypothesis, item.text)
        label = verdict["label"]
        conf = float(verdict["confidence"])
        if label == _LABEL_SUPPORTED:
            if conf > best_entail[0]:
                best_entail = (conf, item)
            if conf >= settings.claim_partial_threshold and item.citation_id not in supporting:
                supporting.append(item.citation_id)
        elif label == _LABEL_CONTRADICTED:
            if conf > best_contra[0]:
                best_contra = (conf, item)
            if conf >= settings.contradiction_threshold and item.citation_id not in contradicting:
                contradicting.append(item.citation_id)

    entail_conf, entail_item = best_entail
    contra_conf, contra_item = best_contra
    claim.verifier = "nli"
    claim.supporting_citations = supporting
    claim.contradicting_citations = contradicting

    if (
        contra_item is not None
        and contra_conf >= settings.contradiction_threshold
        and contra_conf >= entail_conf
    ):
        claim.status = ClaimStatus.contradicted
        claim.nli_label = _LABEL_CONTRADICTED
        claim.nli_confidence = round(contra_conf, 4)
        claim.support_score = round(max(entail_conf, 0.0), 4)
        claim.contradiction_score = round(contra_conf, 4)
        claim.source_chunk_id = contra_item.chunk_id
        claim.best_evidence_span = truncate(contra_item.text, 280)
        claim.rationale = (
            f"NLI contradicted by chunk {contra_item.chunk_id} "
            f"(confidence {contra_conf:.2f})."
        )
        return

    if entail_item is not None and entail_conf >= settings.claim_support_threshold:
        _mark_supported(
            claim,
            entail_item,
            entail_conf,
            contra_conf,
            "nli",
            (
                f"NLI supported by chunk {entail_item.chunk_id} "
                f"(confidence {entail_conf:.2f})."
            ),
        )
        return

    if entail_item is not None and entail_conf >= settings.claim_partial_threshold:
        claim.status = ClaimStatus.partially_supported
        claim.nli_label = _LABEL_SUPPORTED
        claim.nli_confidence = round(entail_conf, 4)
        claim.support_score = round(entail_conf, 4)
        claim.contradiction_score = round(max(contra_conf, 0.0), 4)
        claim.source_chunk_id = entail_item.chunk_id
        claim.best_evidence_span = truncate(entail_item.text, 280)
        claim.rationale = (
            f"NLI weakly supported by chunk {entail_item.chunk_id} "
            f"(confidence {entail_conf:.2f})."
        )
        return

    verbatim = _find_verbatim_chunk(hypothesis, evidence)
    if verbatim is not None:
        _mark_supported(
            claim,
            verbatim,
            1.0,
            contra_conf,
            "verbatim",
            f"Claim text appears in chunk {verbatim.chunk_id}.",
        )
        return

    overlap = _best_stem_overlap(hypothesis, evidence)
    if overlap is not None:
        item, score = overlap
        if score >= settings.claim_support_threshold:
            _mark_supported(
                claim,
                item,
                score,
                contra_conf,
                "lexical",
                f"Claim stems overlap {score:.2f} with chunk {item.chunk_id}.",
            )
            return

    winner = entail_item or (evidence[0] if evidence else None)
    claim.status = ClaimStatus.unsupported
    claim.nli_label = _LABEL_UNSUPPORTED
    claim.nli_confidence = round(max(entail_conf, 0.0), 4)
    claim.support_score = round(max(entail_conf, 0.0), 4)
    claim.contradiction_score = round(max(contra_conf, 0.0), 4)
    claim.source_chunk_id = winner.chunk_id if winner is not None else None
    claim.best_evidence_span = truncate(winner.text, 280) if winner is not None else None
    claim.rationale = (
        f"NLI found no supporting passage (best entailment {max(entail_conf, 0.0):.2f})."
    )


def _mark_supported(
    claim: Claim,
    item: EvidenceItem,
    conf: float,
    contra_conf: float,
    verifier: str,
    rationale: str,
) -> None:
    claim.status = ClaimStatus.supported
    claim.nli_label = _LABEL_SUPPORTED
    claim.nli_confidence = round(conf, 4)
    claim.support_score = round(conf, 4)
    claim.contradiction_score = round(max(contra_conf, 0.0), 4)
    claim.source_chunk_id = item.chunk_id
    claim.best_evidence_span = truncate(item.text, 280)
    claim.verifier = verifier
    if item.citation_id not in claim.supporting_citations:
        claim.supporting_citations = [item.citation_id] + list(claim.supporting_citations)
    claim.rationale = rationale


def _hypothesis_text(claim: str) -> str:
    from ..services.llm.extractive_engine import strip_citations

    clean = normalise_whitespace(strip_citations(claim or ""))
    sentences = split_sentences(clean, min_chars=8)
    text = sentences[0] if sentences else clean
    if len(text) <= 280:
        return text
    return text[:280].rsplit(" ", 1)[0]


def _compact(text: str) -> str:
    return " ".join((text or "").lower().split())


def _find_verbatim_chunk(claim: str, evidence: Sequence[EvidenceItem]) -> EvidenceItem | None:
    compact = _compact(claim)
    if len(compact) < 24:
        return None
    for item in evidence:
        if compact and compact in _compact(item.text):
            return item
    return None


def _best_stem_overlap(
    claim: str, evidence: Sequence[EvidenceItem]
) -> tuple[EvidenceItem, float] | None:
    claim_stems = stem_set(claim)
    if not claim_stems:
        return None
    best: tuple[EvidenceItem, float] | None = None
    for item in evidence:
        score = len(claim_stems & stem_set(item.text)) / len(claim_stems)
        if best is None or score > best[1]:
            best = (item, score)
    return best


def _map_mnli_label(raw: str) -> str:
    token = (raw or "").lower()
    if "entail" in token:
        return _LABEL_SUPPORTED
    if "contradict" in token:
        return _LABEL_CONTRADICTED
    return _LABEL_UNSUPPORTED


def _score_pair(claim: str, evidence_chunk: str) -> tuple[str, float]:
    if _score_override is not None:
        return _score_override(claim, evidence_chunk)
    loaded = _ensure_model()
    if loaded is None:
        raise RuntimeError("NLI model is not available")
    return _predict(loaded, evidence_chunk, claim)


def _ensure_model() -> _LoadedNLI | None:
    global _loaded, _load_failed
    if _loaded is not None:
        return _loaded
    if _load_failed:
        return None
    with _lock:
        if _loaded is not None:
            return _loaded
        if _load_failed:
            return None
        settings = get_settings()
        names = [
            getattr(settings, "nli_model", None) or DEFAULT_NLI_MODEL,
            getattr(settings, "nli_fallback_model", None) or FALLBACK_NLI_MODEL,
        ]
        allow_download = bool(getattr(settings, "nli_allow_download", True))
        for name in names:
            loaded = _try_load(name, allow_download=allow_download)
            if loaded is not None:
                _loaded = loaded
                logger.info("Loaded NLI claim verifier %s on %s", loaded.name, loaded.device)
                return _loaded
        _load_failed = True
        logger.warning(
            "Could not load NLI models %s; claim verification will fall back to the lexical baseline.",
            names,
        )
        return None


def _try_load(name: str, allow_download: bool) -> _LoadedNLI | None:
    try:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except ImportError:
        logger.info("transformers/torch not installed; NLI verifier unavailable")
        return None

    kwargs: dict[str, Any] = {}
    if not allow_download:
        kwargs["local_files_only"] = True

    try:
        tokenizer = AutoTokenizer.from_pretrained(name, **kwargs)
        model = AutoModelForSequenceClassification.from_pretrained(name, **kwargs)
    except Exception as exc:  # network, missing cache, incompatible checkpoint
        logger.info("Failed to load NLI checkpoint %s: %s", name, exc)
        return None

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    id2label = dict(getattr(model.config, "id2label", {}) or {})
    if not id2label:
        id2label = {0: "contradiction", 1: "neutral", 2: "entailment"}
    return _LoadedNLI(name, tokenizer, model, id2label, device)


def _predict(loaded: _LoadedNLI, premise: str, hypothesis: str) -> tuple[str, float]:
    import torch

    encoded = loaded.tokenizer(
        premise[:4000],
        hypothesis[:1000],
        return_tensors="pt",
        truncation=True,
        max_length=512,
        padding=True,
    )
    encoded = {key: value.to(loaded.device) for key, value in encoded.items()}
    with torch.no_grad():
        logits = loaded.model(**encoded).logits[0].detach().cpu().tolist()
    probs = _softmax(logits)
    idx = max(range(len(probs)), key=lambda i: probs[i])
    raw = loaded.id2label.get(idx, "neutral")
    return raw, float(probs[idx])


def _softmax(values: Sequence[float]) -> list[float]:
    peak = max(values)
    exps = [math.exp(v - peak) for v in values]
    total = sum(exps) or 1.0
    return [item / total for item in exps]


def nli_is_available() -> bool:
    if _score_override is not None:
        return True
    return _ensure_model() is not None
