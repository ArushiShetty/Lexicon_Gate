"""Manuscript evidence and citation review (advisory).

Reuses Self-RAG claim extraction and :class:`ClaimVerifier` against **project
evidence quotes and linked sources only**. Does **not** run Ask retrieval,
vector search, or EvidenceGate — that would duplicate the retrieval pipeline.

Status logic (limitations documented here and returned to clients):

* ``supported`` — claim verifier says SUPPORTED against project evidence.
* ``needs_review`` — PARTIALLY_SUPPORTED, uncited substantive claim, missing
  bibliographic fields, or possible duplicate references.
* ``evidence_not_found`` — UNSUPPORTED, empty evidence pool, or inline citation
  that does not map to a linked project source.
* ``possible_mismatch`` — CONTRADICTED or clear polarity/numeric conflict cues.

Automated review is **not** scientific correctness, novelty, or plagiarism-free
proof. Excerpts are taken only from stored project evidence quotes — never
fabricated. The manuscript is never silently rewritten.
"""

from __future__ import annotations

import re
import uuid
from datetime import datetime, timezone
from typing import Any

from ...models.projects import (
    ManuscriptReviewIssue,
    ManuscriptReviewResponse,
    ProjectEvidence,
    ProjectManuscript,
    ProjectSource,
    ResearchProject,
    ReviewIssueKind,
    ReviewVerdict,
)
from ...models.query import Claim, ClaimStatus, EvidenceItem
from ...pipelines.common import number_citations
from ...services.llm.registry import get_llm_provider
from ...verification.claim_extraction import extract_claims
from ...verification.claim_verifier import ClaimVerifier
from . import store as project_store

_CITATION_RE = re.compile(r"\[(\d+)\]")
_DOC_CITE_RE = re.compile(r"\[doc:([^\]]+)\]")
_YEAR_PAREN_RE = re.compile(r"\((?:19|20)\d{2}\)")
_SECTION_REF_HINT_KEYS = {
    "introduction",
    "related_work",
    "literature_review",
    "methodology",
    "results",
    "discussion",
    "background",
    "themes",
    "gaps",
}

LIMITATIONS = [
    "Review uses project evidence quotes and linked source metadata only — no new Library retrieval.",
    "Claim extraction and verification are heuristic; NLI may be unavailable and fall back to lexical overlap.",
    "Statuses are advisory. They do not prove scientific correctness, novelty, or plagiarism-free status.",
    "Evidence excerpts are never invented; if no quote is stored, excerpt fields stay empty.",
    "This review never rewrites manuscript claims; resolve/dismiss only updates issue state.",
]


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def can_run_review(project: ResearchProject, user_id: str) -> bool:
    return project_store.can_run_manuscript_ai(project, user_id)


def can_update_review_issue(project: ResearchProject, user_id: str) -> bool:
    member = project_store.member_for(project, user_id)
    return member is not None and member.role in ("owner", "editor", "reviewer")


def run_manuscript_review(
    project_id: str,
    *,
    section_id: str = "",
    actor_id: str = "",
    actor_name: str = "",
) -> ManuscriptReviewResponse:
    project = project_store.get_project(project_id)
    if project is None:
        raise ValueError("Project not found.")
    ms = project_store.get_manuscript(project_id)
    if ms is None:
        raise ValueError("Manuscript not found.")

    sections = list(ms.sections or [])
    if section_id:
        sections = [s for s in sections if s.section_id == section_id]
        if not sections:
            raise ValueError("Section not found.")

    evidence_items, evidence_by_cite, evidence_meta = _project_evidence_items(
        project.evidence or [], project.sources or []
    )
    warnings: list[str] = []
    if not evidence_items:
        warnings.append(
            "No project evidence quotes available. Claim verification will report "
            "evidence_not_found / needs_review until evidence is added."
        )

    llm = get_llm_provider()
    verifier = ClaimVerifier()
    now = _utcnow_iso()
    new_issues: list[ManuscriptReviewIssue] = []

    # Preserve prior non-open resolutions for matching fingerprints
    prior_closed = [
        i
        for i in (ms.review_issues or [])
        if i.state in ("dismissed", "resolved", "manually_checked")
    ]

    for section in sections:
        body = (section.body or "").strip()
        if not body:
            continue
        new_issues.extend(
            _review_section(
                section=section,
                body=body,
                project=project,
                evidence_items=evidence_items,
                evidence_by_cite=evidence_by_cite,
                evidence_meta=evidence_meta,
                llm=llm,
                verifier=verifier,
                now=now,
            )
        )

    new_issues.extend(
        _biblio_and_duplicate_issues(project.sources or [], now=now)
    )

    # Re-attach closed states when fingerprint matches
    closed_by_fp = {_fingerprint(i): i for i in prior_closed}
    merged: list[ManuscriptReviewIssue] = []
    for issue in new_issues:
        prev = closed_by_fp.get(_fingerprint(issue))
        if prev is not None:
            issue = issue.model_copy(
                update={
                    "state": prev.state,
                    "resolved_by": prev.resolved_by,
                    "resolved_by_name": prev.resolved_by_name,
                    "resolved_at": prev.resolved_at,
                    "resolution_note": prev.resolution_note,
                    "updated_at": now,
                }
            )
        merged.append(issue)

    ms = project_store.save_manuscript_review_issues(
        project_id,
        issues=merged,
        ran_at=now,
        actor_id=actor_id,
        actor_name=actor_name,
    )
    open_issues = [i for i in ms.review_issues if i.state == "open"]
    return ManuscriptReviewResponse(
        manuscript=ms,
        issues=list(ms.review_issues or []),
        open_count=len(open_issues),
        warnings=warnings,
        limitations=list(LIMITATIONS),
    )


def _fingerprint(issue: ManuscriptReviewIssue) -> str:
    return "|".join(
        [
            issue.kind,
            issue.section_id,
            (issue.claim_text or "")[:240],
            issue.document_id,
            issue.citation_marker,
            issue.verdict,
        ]
    )


def _project_evidence_items(
    evidence: list[ProjectEvidence],
    sources: list[ProjectSource],
) -> tuple[list[EvidenceItem], dict[int, ProjectEvidence], dict[str, Any]]:
    """Map stored quotes to EvidenceItems. Never invent quote text."""
    source_by_id = {s.document_id: s for s in sources}
    items: list[EvidenceItem] = []
    meta_by_doc: dict[str, Any] = {}
    for ev in evidence:
        quote = (ev.quote or "").strip()
        if not quote:
            continue
        title = (ev.title or "").strip()
        src = source_by_id.get(ev.document_id)
        if not title and src is not None:
            title = (src.title or "").strip()
        authors = list(ev.authors or []) or list((src.authors if src else []) or [])
        year = ev.year if ev.year is not None else (src.year if src else None)
        doi = (ev.doi or "").strip() or ((src.doi if src else "") or "")
        items.append(
            EvidenceItem(
                citation_id=0,
                chunk_id=ev.chunk_id or ev.evidence_id,
                document_id=ev.document_id,
                document_name=title or ev.document_id,
                page=ev.page,
                text=quote,
                title=title,
                authors=authors,
                year=year,
                doi=doi or None,
                source_quality=0.7,
            )
        )
        meta_by_doc[ev.document_id] = {
            "evidence_id": ev.evidence_id,
            "title": title,
            "quote": quote,
        }
    numbered = number_citations(items)
    by_cite: dict[int, ProjectEvidence] = {}
    quoted = [e for e in evidence if (e.quote or "").strip()]
    for item, ev in zip(numbered, quoted):
        by_cite[item.citation_id] = ev
    return numbered, by_cite, meta_by_doc


def _review_section(
    *,
    section,
    body: str,
    project: ResearchProject,
    evidence_items: list[EvidenceItem],
    evidence_by_cite: dict[int, ProjectEvidence],
    evidence_meta: dict[str, Any],
    llm,
    verifier: ClaimVerifier,
    now: str,
) -> list[ManuscriptReviewIssue]:
    issues: list[ManuscriptReviewIssue] = []
    project_doc_ids = {s.document_id for s in (project.sources or [])}

    # Inline citations not resolvable against project evidence numbering
    markers = _CITATION_RE.findall(body)
    for marker in markers:
        n = int(marker)
        if n not in evidence_by_cite and evidence_items:
            # Citation numbers in manuscript may not match Ask numbering —
            # flag as needs_review with uncertainty rather than inventing a link.
            issues.append(
                _issue(
                    kind="citation_not_in_project",
                    verdict="needs_review",
                    section=section,
                    claim_text="",
                    citation_marker=f"[{n}]",
                    citation_status="unresolved_marker",
                    potential_issue=(
                        f"Inline citation [{n}] could not be reliably mapped to a "
                        "project evidence quote. Confirm the reference manually."
                    ),
                    uncertainty="Citation indices in drafts may not match project evidence order.",
                    now=now,
                )
            )
        elif not evidence_items and markers:
            issues.append(
                _issue(
                    kind="citation_not_in_project",
                    verdict="evidence_not_found",
                    section=section,
                    citation_marker=f"[{n}]",
                    citation_status="no_project_evidence",
                    potential_issue=(
                        f"Citation [{n}] appears but this project has no evidence quotes to inspect."
                    ),
                    uncertainty="Add project evidence before treating citations as verified.",
                    now=now,
                )
            )

    # Stable document_id markers: [doc:<document_id>]
    for doc_marker in _DOC_CITE_RE.findall(body):
        doc_id = str(doc_marker).strip()
        if not doc_id:
            continue
        if doc_id not in project_doc_ids:
            issues.append(
                _issue(
                    kind="citation_not_in_project",
                    verdict="evidence_not_found",
                    section=section,
                    document_id=doc_id,
                    citation_marker=f"[doc:{doc_id}]",
                    citation_status="source_not_linked",
                    potential_issue=(
                        f"Citation marker [doc:{doc_id}] does not match a linked project source."
                    ),
                    uncertainty="Link the Library document under Sources, or update the marker.",
                    now=now,
                )
            )

    # Author-year style citations without a matching source year/title (cautious)
    if _YEAR_PAREN_RE.search(body) and not project_doc_ids:
        issues.append(
            _issue(
                kind="citation_not_in_project",
                verdict="evidence_not_found",
                section=section,
                citation_status="no_linked_sources",
                potential_issue=(
                    "Parenthetical years look like citations, but no sources are linked to this project."
                ),
                uncertainty="Year parentheses are only a heuristic signal.",
                now=now,
            )
        )

    claims = extract_claims(body, llm, max_claims=12)
    if not claims:
        return issues

    if evidence_items:
        result = verifier.verify(claims, evidence_items, query=section.title or "")
        verified = result.claims
    else:
        verified = []
        for claim in claims:
            claim.status = ClaimStatus.unsupported
            claim.rationale = "No project evidence quotes available for verification."
            claim.verifier = "none"
            verified.append(claim)

    for claim in verified:
        verdict, kind = _map_claim_status(claim, has_evidence=bool(evidence_items))
        ev_id = ""
        excerpt = ""
        doc_id = ""
        source_title = ""
        if claim.supporting_citations and evidence_by_cite:
            for cid in claim.supporting_citations:
                pe = evidence_by_cite.get(cid)
                if pe is not None:
                    ev_id = pe.evidence_id
                    excerpt = (pe.quote or "")[:800]
                    doc_id = pe.document_id
                    source_title = pe.title or ""
                    break
        if not excerpt and claim.best_evidence_span:
            # best_evidence_span comes from verifier over provided evidence text only
            excerpt = (claim.best_evidence_span or "")[:800]
        cite_status = "linked" if ev_id else ("none" if not evidence_items else "unlinked")
        uncertainty = ""
        if claim.nli_confidence is not None and claim.nli_confidence < 0.55:
            uncertainty = "Low NLI confidence — treat as inconclusive."
        elif not evidence_items:
            uncertainty = "Verification inconclusive: no project evidence quotes."
        elif claim.status == ClaimStatus.partially_supported:
            uncertainty = "Only partial overlap with evidence; human review recommended."

        issues.append(
            _issue(
                kind=kind,
                verdict=verdict,
                section=section,
                claim_text=claim.text,
                document_id=doc_id,
                source_title=source_title,
                evidence_id=ev_id,
                evidence_excerpt=excerpt,
                citation_status=cite_status,
                verification_status=claim.status.value if hasattr(claim.status, "value") else str(claim.status),
                potential_issue=claim.rationale or _default_issue_text(verdict, claim),
                uncertainty=uncertainty,
                now=now,
                meta={
                    "claim_id": claim.claim_id,
                    "support_score": claim.support_score,
                    "verifier": claim.verifier,
                    "nli_label": claim.nli_label,
                    "nli_confidence": claim.nli_confidence,
                },
            )
        )

        # Uncited substantive claims
        if not _has_inline_citation(claim.text) and len(claim.text.split()) >= 8:
            if verdict in ("evidence_not_found", "needs_review", "possible_mismatch") or not evidence_items:
                issues.append(
                    _issue(
                        kind="claim_needs_citation",
                        verdict="needs_review",
                        section=section,
                        claim_text=claim.text,
                        citation_status="uncited",
                        potential_issue=(
                            "This claim may need a citation. Automated detection is cautious — "
                            "confirm before inserting a reference."
                        ),
                        uncertainty="Heuristic: long claim without [n] or (year).",
                        now=now,
                    )
                )

    # Section-level cautious hint
    if (
        section.key in _SECTION_REF_HINT_KEYS
        and len(body.split()) >= 40
        and not _CITATION_RE.search(body)
        and not _YEAR_PAREN_RE.search(body)
    ):
        issues.append(
            _issue(
                kind="section_may_need_references",
                verdict="needs_review",
                section=section,
                potential_issue=(
                    f"Section '{section.title}' has substantial text without detectable citations. "
                    "It may require references — review manually."
                ),
                uncertainty="Section-level heuristic only; not a requirement.",
                now=now,
            )
        )

    return issues


def _map_claim_status(
    claim: Claim, *, has_evidence: bool
) -> tuple[ReviewVerdict, ReviewIssueKind]:
    status = claim.status
    if status == ClaimStatus.supported:
        return "supported", "claim_verification"
    if status == ClaimStatus.contradicted:
        return "possible_mismatch", "claim_evidence_mismatch"
    if status == ClaimStatus.unsupported:
        return "evidence_not_found", "claim_verification"
    # partially_supported
    return "needs_review", "claim_verification"


def _default_issue_text(verdict: ReviewVerdict, claim: Claim) -> str:
    if verdict == "supported":
        return "Claim appears supported by project evidence (advisory)."
    if verdict == "possible_mismatch":
        return "Possible mismatch between claim wording and project evidence."
    if verdict == "evidence_not_found":
        return "No supporting project evidence was found for this claim."
    return "Claim needs human review against project evidence."


def _has_inline_citation(text: str) -> bool:
    return bool(_CITATION_RE.search(text) or _YEAR_PAREN_RE.search(text) or re.search(r"et al\.", text, re.I))


def _biblio_and_duplicate_issues(
    sources: list[ProjectSource], *, now: str
) -> list[ManuscriptReviewIssue]:
    issues: list[ManuscriptReviewIssue] = []
    seen_doi: dict[str, ProjectSource] = {}
    seen_title: dict[str, ProjectSource] = {}
    for src in sources:
        missing: list[str] = []
        if not (src.title or "").strip():
            missing.append("title")
        if not (src.authors or []):
            missing.append("authors")
        if src.year is None:
            missing.append("year")
        if missing:
            issues.append(
                ManuscriptReviewIssue(
                    issue_id=uuid.uuid4().hex,
                    kind="missing_biblio_fields",
                    verdict="needs_review",
                    state="open",
                    document_id=src.document_id,
                    source_title=src.title or "",
                    citation_status="incomplete_biblio",
                    potential_issue=(
                        f"Linked source is missing bibliographic field(s): {', '.join(missing)}. "
                        "Nothing was invented to fill gaps."
                    ),
                    uncertainty="Based on project source snapshot / Library metadata only.",
                    created_at=now,
                    updated_at=now,
                )
            )
        doi = (src.doi or "").strip().lower()
        if doi:
            if doi in seen_doi:
                issues.append(
                    ManuscriptReviewIssue(
                        issue_id=uuid.uuid4().hex,
                        kind="possible_duplicate_reference",
                        verdict="needs_review",
                        state="open",
                        document_id=src.document_id,
                        source_title=src.title or "",
                        potential_issue=(
                            f"Possible duplicate DOI shared with document "
                            f"{seen_doi[doi].document_id}."
                        ),
                        uncertainty="DOI string match only; confirm before removing.",
                        created_at=now,
                        updated_at=now,
                        meta={"other_document_id": seen_doi[doi].document_id},
                    )
                )
            else:
                seen_doi[doi] = src
        title_key = re.sub(r"\W+", " ", (src.title or "").lower()).strip()
        if title_key and len(title_key) > 12:
            if title_key in seen_title:
                issues.append(
                    ManuscriptReviewIssue(
                        issue_id=uuid.uuid4().hex,
                        kind="possible_duplicate_reference",
                        verdict="needs_review",
                        state="open",
                        document_id=src.document_id,
                        source_title=src.title or "",
                        potential_issue=(
                            f"Possible duplicate title vs document "
                            f"{seen_title[title_key].document_id}."
                        ),
                        uncertainty="Normalized title match only.",
                        created_at=now,
                        updated_at=now,
                        meta={"other_document_id": seen_title[title_key].document_id},
                    )
                )
            else:
                seen_title[title_key] = src
    return issues


def _issue(
    *,
    kind: ReviewIssueKind,
    verdict: ReviewVerdict,
    section,
    now: str,
    claim_text: str = "",
    document_id: str = "",
    source_title: str = "",
    evidence_id: str = "",
    evidence_excerpt: str = "",
    citation_marker: str = "",
    citation_status: str = "",
    verification_status: str = "",
    potential_issue: str = "",
    uncertainty: str = "",
    meta: dict[str, Any] | None = None,
) -> ManuscriptReviewIssue:
    return ManuscriptReviewIssue(
        issue_id=uuid.uuid4().hex,
        kind=kind,
        verdict=verdict,
        state="open",
        section_id=section.section_id,
        section_key=section.key,
        section_title=section.title,
        claim_text=claim_text,
        document_id=document_id,
        source_title=source_title,
        evidence_id=evidence_id,
        evidence_excerpt=evidence_excerpt,
        citation_marker=citation_marker,
        citation_status=citation_status,
        verification_status=verification_status,
        potential_issue=potential_issue,
        uncertainty=uncertainty,
        meta=meta or {},
        created_at=now,
        updated_at=now,
    )
