"""Submission readiness checklist for project manuscripts.

Human-owned checklist statuses. Auto hints/warnings may link empty sections or
open evidence-review issues, but never auto-mark items completed and never
claim venue compliance, acceptance probability, or plagiarism-free status.
"""

from __future__ import annotations

import re
import uuid
from datetime import datetime, timezone
from typing import Any

from ...models.projects import (
    CHECKLIST_ITEM_STATUSES,
    ChecklistItemStatus,
    ManuscriptChecklist,
    ManuscriptChecklistItem,
    ManuscriptChecklistResponse,
    ManuscriptTemplate,
    ProjectManuscript,
    ResearchProject,
)
from . import store as project_store

_DISCLAIMER = (
    "Submission readiness is a researcher checklist only. "
    "It is not journal/conference compliance certification, not an acceptance "
    "prediction, and not a plagiarism-free claim. Always verify venue-specific "
    "instructions manually before submission."
)

# key, label, description, section_key (optional), linked_issue_kinds
_ChecklistDef = tuple[str, str, str, str, tuple[str, ...]]

_COMMON: list[_ChecklistDef] = [
    ("title", "Title", "Confirm the manuscript title is final and accurate.", "title", ()),
    ("abstract", "Abstract", "Review abstract completeness and clarity.", "abstract", ()),
    ("keywords", "Keywords", "Confirm keywords are present and appropriate.", "keywords", ()),
    (
        "required_sections",
        "Required sections",
        "Confirm required sections for this document type have content.",
        "",
        ("section_may_need_references",),
    ),
    (
        "citations",
        "Citations",
        "Check in-text citations against project sources and evidence review.",
        "",
        ("citation_not_in_project", "claim_needs_citation"),
    ),
    (
        "references",
        "References",
        "Review the references section and bibliographic completeness.",
        "references",
        ("missing_biblio_fields", "possible_duplicate_reference"),
    ),
    (
        "unsupported_claims",
        "Unsupported claims",
        "Review open evidence-review findings for unsupported or mismatched claims.",
        "",
        ("claim_verification", "claim_evidence_mismatch"),
    ),
    (
        "figures_tables",
        "Figures and tables",
        "Confirm figures/tables (if any) are present, captioned, and cited in text. "
        "Lexicon Gate does not auto-detect layout compliance.",
        "",
        (),
    ),
    (
        "author_info",
        "Author information",
        "Confirm author names and affiliations on the manuscript record.",
        "",
        (),
    ),
    (
        "formatting_review",
        "Formatting review",
        "Manually check venue formatting (margins, fonts, columns, numbering). "
        "Not an IEEE or venue compliance certification.",
        "",
        (),
    ),
    (
        "final_export_review",
        "Final export review",
        "Review the export/PDF preview before submission. Venue instructions must be checked manually.",
        "",
        (),
    ),
]

_LIT_EXTRA: list[_ChecklistDef] = [
    (
        "search_strategy",
        "Search strategy",
        "Confirm search strategy / inclusion criteria are documented.",
        "search_strategy",
        (),
    ),
    (
        "synthesis",
        "Literature synthesis",
        "Confirm themes/synthesis and research gaps sections are ready.",
        "themes",
        (),
    ),
]

CHECKLIST_TEMPLATES: dict[ManuscriptTemplate, list[_ChecklistDef]] = {
    "ieee_research": list(_COMMON),
    "generic_academic": list(_COMMON),
    "other": list(_COMMON),
    "literature_review": [
        *[c for c in _COMMON if c[0] not in ("figures_tables",)],
        *_LIT_EXTRA,
        next(c for c in _COMMON if c[0] == "figures_tables"),
    ],
}


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def can_edit_checklist(project: ResearchProject, user_id: str) -> bool:
    member = project_store.member_for(project, user_id)
    return member is not None and member.role in ("owner", "editor", "reviewer")


def build_default_checklist(document_type: ManuscriptTemplate) -> ManuscriptChecklist:
    tpl = document_type if document_type in CHECKLIST_TEMPLATES else "ieee_research"
    items: list[ManuscriptChecklistItem] = []
    for key, label, desc, section_key, kinds in CHECKLIST_TEMPLATES[tpl]:
        items.append(
            ManuscriptChecklistItem(
                item_id=uuid.uuid4().hex,
                key=key,
                label=label,
                description=desc,
                status="incomplete",
                section_key=section_key,
                linked_issue_kinds=list(kinds),
            )
        )
    return ManuscriptChecklist(
        items=items,
        document_type=tpl,  # type: ignore[arg-type]
        refreshed_at=_utcnow_iso(),
        disclaimer=_DISCLAIMER,
    )


def ensure_checklist(ms: ProjectManuscript) -> ManuscriptChecklist:
    doc_type = ms.document_type or ms.template or "ieee_research"
    if ms.checklist is None or not ms.checklist.items:
        return build_default_checklist(doc_type)  # type: ignore[arg-type]
    if ms.checklist.document_type != doc_type:
        # Preserve statuses by key when document type changes
        prior = {i.key: i for i in ms.checklist.items}
        fresh = build_default_checklist(doc_type)  # type: ignore[arg-type]
        merged = []
        for item in fresh.items:
            old = prior.get(item.key)
            if old is not None:
                merged.append(
                    item.model_copy(
                        update={
                            "item_id": old.item_id,
                            "status": old.status,
                            "updated_by": old.updated_by,
                            "updated_by_name": old.updated_by_name,
                            "updated_at": old.updated_at,
                        }
                    )
                )
            else:
                merged.append(item)
        return ManuscriptChecklist(
            items=merged,
            document_type=doc_type,  # type: ignore[arg-type]
            refreshed_at=_utcnow_iso(),
            disclaimer=_DISCLAIMER,
        )
    return ms.checklist.model_copy(update={"disclaimer": _DISCLAIMER})


def refresh_checklist_hints(
    ms: ProjectManuscript, checklist: ManuscriptChecklist
) -> ManuscriptChecklist:
    """Attach warnings / section links from manuscript + open review issues. Does not change status."""
    by_key = {s.key: s for s in (ms.sections or [])}
    open_issues = [i for i in (ms.review_issues or []) if i.state == "open"]
    updated_items: list[ManuscriptChecklistItem] = []
    for item in checklist.items:
        section = by_key.get(item.section_key) if item.section_key else None
        warning = ""
        auto_hint = ""
        linked_ids: list[str] = []
        section_id = section.section_id if section else ""

        if item.key == "title":
            title_body = (section.body if section else "") or (ms.title or "")
            if not title_body.strip():
                warning = "Title section/body appears empty."
            auto_hint = "Linked to Title section when present."
        elif item.key == "abstract":
            if section is None or not (section.body or "").strip():
                warning = "Abstract section is empty or missing."
        elif item.key == "keywords":
            if section is None or not (section.body or "").strip():
                warning = "Keywords section is empty or missing."
        elif item.key == "required_sections":
            missing = _missing_required_sections(ms)
            if missing:
                warning = "Sections with little or no content: " + ", ".join(missing)
            auto_hint = "Based on document-type template keys; not venue-mandated."
        elif item.key == "citations":
            linked_ids = [
                i.issue_id
                for i in open_issues
                if i.kind in set(item.linked_issue_kinds or [])
            ]
            if linked_ids:
                warning = f"{len(linked_ids)} open citation-related review issue(s)."
            elif not _body_has_citation_signal(ms):
                warning = "No obvious in-text citations detected (heuristic)."
        elif item.key == "references":
            if section is None or not (section.body or "").strip():
                warning = "References section is empty or missing."
            linked_ids = [
                i.issue_id
                for i in open_issues
                if i.kind in set(item.linked_issue_kinds or [])
            ]
            if linked_ids and not warning:
                warning = f"{len(linked_ids)} open bibliography review issue(s)."
        elif item.key == "unsupported_claims":
            linked_ids = [
                i.issue_id
                for i in open_issues
                if i.kind in set(item.linked_issue_kinds or [])
                and i.verdict in ("evidence_not_found", "possible_mismatch", "needs_review")
            ]
            if linked_ids:
                warning = f"{len(linked_ids)} open claim-verification issue(s)."
            elif not (ms.review_ran_at or "").strip():
                auto_hint = "Run evidence & citation review to refresh claim warnings."
        elif item.key == "author_info":
            if not (ms.authors or []):
                warning = "No authors listed on the manuscript record."
        elif item.key == "figures_tables":
            auto_hint = "Manual check required — layout is not validated by Lexicon Gate."
        elif item.key == "formatting_review":
            auto_hint = "Venue-specific formatting must be checked manually."
        elif item.key == "final_export_review":
            auto_hint = "Not an acceptance prediction or compliance certificate."
        elif item.key in ("search_strategy", "synthesis"):
            if section is None or not (section.body or "").strip():
                warning = f"{item.label} section is empty or missing."

        updated_items.append(
            item.model_copy(
                update={
                    "section_id": section_id,
                    "warning": warning,
                    "auto_hint": auto_hint,
                    "linked_issue_ids": linked_ids,
                }
            )
        )
    return checklist.model_copy(
        update={
            "items": updated_items,
            "refreshed_at": _utcnow_iso(),
            "disclaimer": _DISCLAIMER,
        }
    )


def _missing_required_sections(ms: ProjectManuscript) -> list[str]:
    from ...models.projects import MANUSCRIPT_TEMPLATES

    tpl = ms.document_type or ms.template or "ieee_research"
    required = MANUSCRIPT_TEMPLATES.get(tpl) or MANUSCRIPT_TEMPLATES["ieee_research"]
    by_key = {s.key: s for s in (ms.sections or [])}
    missing = []
    for key, title in required:
        sec = by_key.get(key)
        body = (sec.body if sec else "") or ""
        # title section may be covered by ms.title
        if key == "title" and (ms.title or "").strip():
            continue
        if not sec or len(body.strip()) < 12:
            missing.append(title)
    return missing


def _body_has_citation_signal(ms: ProjectManuscript) -> bool:
    blob = " ".join((s.body or "") for s in (ms.sections or []))
    return bool(re.search(r"\[\d+\]|\((?:19|20)\d{2}\)|et al\.", blob, re.I))


def summarize_checklist(checklist: ManuscriptChecklist) -> dict[str, Any]:
    counts = {"incomplete": 0, "in_progress": 0, "completed": 0}
    warn_count = 0
    for item in checklist.items:
        counts[item.status] = counts.get(item.status, 0) + 1
        if item.warning:
            warn_count += 1
    total = len(checklist.items) or 1
    return {
        "total": len(checklist.items),
        "incomplete": counts["incomplete"],
        "in_progress": counts["in_progress"],
        "completed": counts["completed"],
        "with_warnings": warn_count,
        "completion_ratio": round(counts["completed"] / total, 3),
        "headline": (
            f"{counts['completed']} of {len(checklist.items)} checklist items marked completed "
            "(researcher status only — not an acceptance prediction)."
        ),
    }


def get_checklist_response(project_id: str) -> ManuscriptChecklistResponse:
    ms = project_store.get_manuscript(project_id)
    if ms is None:
        raise ValueError("Manuscript not found.")
    checklist = ensure_checklist(ms)
    checklist = refresh_checklist_hints(ms, checklist)
    # Persist ensured/refreshed checklist (hints only; preserves statuses)
    ms = project_store.save_manuscript_checklist(project_id, checklist)
    return ManuscriptChecklistResponse(
        checklist=ms.checklist or checklist,
        manuscript_id=ms.manuscript_id,
        summary=summarize_checklist(ms.checklist or checklist),
        warnings=[
            "Checklist statuses are set by researchers. Auto warnings are hints only.",
            "Venue-specific instructions must be checked manually.",
        ],
    )


def refresh_checklist_response(project_id: str) -> ManuscriptChecklistResponse:
    return get_checklist_response(project_id)


def update_checklist_item(
    project_id: str,
    item_key: str,
    *,
    status: ChecklistItemStatus,
    actor_id: str,
    actor_name: str = "",
    note: str = "",
) -> ManuscriptChecklistResponse:
    if status not in CHECKLIST_ITEM_STATUSES:
        raise ValueError("Invalid checklist status.")
    ms = project_store.get_manuscript(project_id)
    if ms is None:
        raise ValueError("Manuscript not found.")
    checklist = ensure_checklist(ms)
    found = False
    now = _utcnow_iso()
    items = []
    for item in checklist.items:
        if item.key == item_key or item.item_id == item_key:
            found = True
            items.append(
                item.model_copy(
                    update={
                        "status": status,
                        "updated_by": actor_id,
                        "updated_by_name": actor_name,
                        "updated_at": now,
                        "auto_hint": (note or item.auto_hint or "").strip() or item.auto_hint,
                    }
                )
            )
        else:
            items.append(item)
    if not found:
        raise ValueError("Checklist item not found.")
    checklist = checklist.model_copy(update={"items": items, "refreshed_at": now})
    checklist = refresh_checklist_hints(ms, checklist)
    ms = project_store.save_manuscript_checklist(
        project_id,
        checklist,
        actor_id=actor_id,
        actor_name=actor_name,
        activity_message=f"Updated checklist item '{item_key}' to {status}.",
    )
    return ManuscriptChecklistResponse(
        checklist=ms.checklist or checklist,
        manuscript_id=ms.manuscript_id,
        summary=summarize_checklist(ms.checklist or checklist),
        warnings=[
            "Status change recorded. This is not venue compliance or acceptance probability."
        ],
    )
