"""Transparent, evidence-derived confidence scoring.

The model is never asked "how confident are you?". Confidence is a weighted
combination of four observables:

* evidence relevance (from the evidence gate)
* claim support rate (from claim-level verification)
* source agreement (1 − contradiction pressure)
* evidence coverage (query-term coverage of the retrieved set)

The display is accompanied by an explicit caveat that this is a *system*
confidence metric, not a guarantee of truth. Calibration against actual
correctness is measured in the evaluation harness (ECE).
"""

from __future__ import annotations

from ..config import Settings, get_settings
from ..models.query import (
    Claim,
    ClaimStatus,
    ConfidenceReport,
    ContradictionPair,
    EvidenceGateDecision,
    EvidenceItem,
)
from ..text_utils import clamp


def score_confidence(
    evidence: list[EvidenceItem],
    claims: list[Claim],
    gate: EvidenceGateDecision | None,
    contradictions: list[ContradictionPair] | None = None,
    settings: Settings | None = None,
) -> ConfidenceReport:
    settings = settings or get_settings()
    contradictions = contradictions or []

    evidence_component = 0.0
    coverage_component = 0.0
    if gate is not None:
        evidence_component = clamp(gate.evidence_score)
        coverage_component = clamp(gate.coverage)
    elif evidence:
        evidence_component = clamp(sum(i.evidence_score for i in evidence) / len(evidence))
        coverage_component = clamp(sum(i.relevance_score for i in evidence) / len(evidence))

    n_claims = len(claims)
    n_supported = sum(1 for c in claims if c.status is ClaimStatus.supported)
    n_partial = sum(1 for c in claims if c.status is ClaimStatus.partially_supported)
    n_unsupported = sum(1 for c in claims if c.status is ClaimStatus.unsupported)
    n_contradicted = sum(1 for c in claims if c.status is ClaimStatus.contradicted)
    if n_claims:
        claim_support = (n_supported + 0.5 * n_partial) / n_claims
    else:
        claim_support = 0.0 if not evidence else 0.5

    if contradictions:
        peak = max(p.score for p in contradictions)
        source_agreement = clamp(1.0 - peak)
    elif gate is not None:
        source_agreement = clamp(gate.consistency)
    else:
        source_agreement = 1.0

    weights = (
        settings.conf_weight_evidence,
        settings.conf_weight_claim_support,
        settings.conf_weight_source_agreement,
        settings.conf_weight_coverage,
    )
    total = sum(weights) or 1.0
    confidence = (
        weights[0] * evidence_component
        + weights[1] * claim_support
        + weights[2] * source_agreement
        + weights[3] * coverage_component
    ) / total

    # A single contradicted claim is a hard penalty: the system is not merely
    # uncertain, it has evidence against itself.
    if n_contradicted:
        confidence *= 0.7
    if not evidence:
        confidence = 0.0

    confidence = clamp(confidence)
    if confidence >= 0.8:
        label = "high"
    elif confidence >= 0.55:
        label = "moderate"
    elif confidence >= 0.35:
        label = "low"
    else:
        label = "very low"

    return ConfidenceReport(
        confidence=round(confidence, 4),
        evidence_component=round(evidence_component, 4),
        claim_support_component=round(claim_support, 4),
        source_agreement_component=round(source_agreement, 4),
        coverage_component=round(coverage_component, 4),
        evidence_coverage=round(coverage_component, 4),
        claims_verified=n_supported + n_partial,
        claims_total=n_claims,
        unsupported_claims=n_unsupported,
        contradicted_claims=n_contradicted,
        label=label,
    )
