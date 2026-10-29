"""Research projects API — CRUD, membership, invites, notes, reviews."""

from __future__ import annotations

import time
from collections import defaultdict, deque

from fastapi import APIRouter, HTTPException, Query, Request, Response

from .auth import current_user_from_request
from ..config import get_settings
from ..models.projects import (
    EvidenceCreateRequest,
    EvidenceListResponse,
    InviteAcceptRequest,
    InviteCreateRequest,
    InviteCreateResponse,
    InviteRejectRequest,
    ManuscriptAiRequest,
    ManuscriptAiResponse,
    ManuscriptAssignRequest,
    ManuscriptCiteRequest,
    ManuscriptCiteResponse,
    ChecklistItemUpdateRequest,
    ManuscriptChecklistResponse,
    ManuscriptCreateRequest,
    ManuscriptReviewIssueUpdate,
    ManuscriptReviewResponse,
    ManuscriptReviewRunRequest,
    ManuscriptSectionCreateRequest,
    ManuscriptSectionOrderRequest,
    ManuscriptSectionPatchRequest,
    ManuscriptSectionUpdate,
    ManuscriptUpdateRequest,
    MemberRoleUpdate,
    NoteCreateRequest,
    NoteUpdateRequest,
    PendingInvitesResponse,
    ProjectCreate,
    ProjectListResponse,
    ProjectManuscript,
    ProjectSourceImportRequest,
    ProjectSourceLinkRequest,
    ProjectSourceListResponse,
    ProjectSourceMutationResponse,
    ProjectUpdate,
    ResearchProject,
    ReviewCreateRequest,
    SectionCommentRequest,
    TaskCommentRequest,
    TaskCreateRequest,
    TaskUpdateRequest,
)
from ..models.papers import PaperImportRequest
from ..services.auth.users import (
    ensure_researcher_id,
    find_by_researcher_id,
    find_by_user_id,
    normalise_researcher_id,
)
from ..services.papers.import_paper import import_paper
from ..services.projects import store as project_store
from ..services.store.knowledge_base import get_knowledge_base

router = APIRouter(prefix="/projects", tags=["projects"])

# Soft rate limit for invite accept (per user): max N attempts per window.
_ACCEPT_WINDOW_SEC = 60
_ACCEPT_MAX_ATTEMPTS = 20
_accept_attempts: dict[str, deque[float]] = defaultdict(deque)


def _require_user(request: Request) -> dict:
    user = current_user_from_request(request)
    if user is None or not user.get("user_id"):
        raise HTTPException(status_code=401, detail="Not signed in.")
    return user


def _check_accept_rate_limit(user_id: str) -> None:
    now = time.monotonic()
    bucket = _accept_attempts[user_id]
    while bucket and now - bucket[0] > _ACCEPT_WINDOW_SEC:
        bucket.popleft()
    if len(bucket) >= _ACCEPT_MAX_ATTEMPTS:
        raise HTTPException(
            status_code=429,
            detail="Too many invite accept attempts. Wait a minute and try again.",
        )
    bucket.append(now)


def _actor(user: dict) -> tuple[str, str, str]:
    return (
        str(user["user_id"]),
        str(user.get("email") or ""),
        str(user.get("name") or ""),
    )


def _get_accessible(project_id: str, user_id: str) -> ResearchProject:
    project = project_store.get_project(project_id)
    if project is None or not project_store.can_view(project, user_id):
        raise HTTPException(status_code=404, detail="Project not found.")
    return project


@router.post("/", response_model=ResearchProject)
def create_project(payload: ProjectCreate, request: Request) -> ResearchProject:
    user = _require_user(request)
    try:
        return project_store.create_project(
            owner_id=str(user["user_id"]),
            owner_email=str(user.get("email") or ""),
            owner_name=str(user.get("name") or ""),
            title=payload.title,
            topic=payload.topic,
            description=payload.description,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/", response_model=ProjectListResponse)
def list_projects(
    request: Request,
    limit: int = Query(default=50, ge=1, le=100),
    include_archived: bool = Query(default=True),
) -> ProjectListResponse:
    user = _require_user(request)
    projects = project_store.list_projects_for_user(
        str(user["user_id"]),
        limit=limit,
        include_archived=include_archived,
    )
    return ProjectListResponse(projects=projects, total=len(projects))


@router.get("/invites/pending", response_model=PendingInvitesResponse)
def list_my_pending_invites(request: Request) -> PendingInvitesResponse:
    """Invites addressed to the signed-in user (no tokens exposed)."""
    user = _require_user(request)
    invites = project_store.list_pending_invites_for_user(
        user_id=str(user.get("user_id") or ""),
        email=str(user.get("email") or ""),
    )
    return PendingInvitesResponse(invites=invites, total=len(invites))


@router.post("/invites/accept", response_model=ResearchProject)
def accept_invite(payload: InviteAcceptRequest, request: Request) -> ResearchProject:
    """Accept an invite. In-app (Researcher ID) invites need no token; legacy email invites do."""
    user = _require_user(request)
    user_id, email, name = _actor(user)
    _check_accept_rate_limit(user_id)
    settings = get_settings()
    full = find_by_user_id(settings, user_id)
    rid = ""
    if full:
        full = ensure_researcher_id(settings, full)
        rid = str(full.get("researcher_id") or "")
    else:
        rid = str(user.get("researcher_id") or "")
    try:
        return project_store.accept_invite(
            invite_id=payload.invite_id.strip(),
            token=(payload.token or "").strip() or None,
            user_id=user_id,
            user_email=email,
            user_name=name,
            user_researcher_id=rid,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/invites/reject")
def reject_invite(payload: InviteRejectRequest, request: Request) -> dict[str, str]:
    """Invitee rejects a pending invitation."""
    user = _require_user(request)
    user_id, email, _name = _actor(user)
    try:
        project_store.reject_invite(
            invite_id=payload.invite_id.strip(),
            user_id=user_id,
            user_email=email,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"status": "rejected"}


@router.get("/{project_id}", response_model=ResearchProject)
def get_project(project_id: str, request: Request) -> ResearchProject:
    user = _require_user(request)
    return _get_accessible(project_id, str(user["user_id"]))


@router.patch("/{project_id}", response_model=ResearchProject)
def update_project(project_id: str, payload: ProjectUpdate, request: Request) -> ResearchProject:
    user = _require_user(request)
    user_id, email, name = _actor(user)
    project = _get_accessible(project_id, user_id)
    if not project_store.can_edit_metadata(project, user_id):
        raise HTTPException(status_code=403, detail="Not allowed to edit this project.")
    if payload.status == "archived" and not project_store.can_manage(project, user_id):
        raise HTTPException(status_code=403, detail="Only the owner can archive this project.")
    try:
        updated = project_store.update_project(
            project_id,
            title=payload.title,
            topic=payload.topic,
            description=payload.description,
            status=payload.status,
            document_ids=payload.document_ids,
            actor_id=user_id,
            actor_email=email,
            actor_name=name,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if updated is None:
        raise HTTPException(status_code=404, detail="Project not found.")
    return updated


@router.post("/{project_id}/archive", response_model=ResearchProject)
def archive_project(project_id: str, request: Request) -> ResearchProject:
    user = _require_user(request)
    user_id, email, name = _actor(user)
    project = _get_accessible(project_id, user_id)
    if not project_store.can_manage(project, user_id):
        raise HTTPException(status_code=403, detail="Only the owner can archive this project.")
    updated = project_store.archive_project(
        project_id, actor_id=user_id, actor_email=email, actor_name=name
    )
    if updated is None:
        raise HTTPException(status_code=404, detail="Project not found.")
    return updated


@router.delete("/{project_id}", status_code=204)
def delete_project(project_id: str, request: Request) -> Response:
    user = _require_user(request)
    user_id = str(user["user_id"])
    project = _get_accessible(project_id, user_id)
    if not project_store.can_manage(project, user_id):
        raise HTTPException(status_code=403, detail="Only the owner can delete this project.")
    if not project_store.delete_project(project_id):
        raise HTTPException(status_code=404, detail="Project not found.")
    return Response(status_code=204)


@router.post("/{project_id}/invites", response_model=InviteCreateResponse)
def create_invite(
    project_id: str, payload: InviteCreateRequest, request: Request
) -> InviteCreateResponse:
    user = _require_user(request)
    user_id, email, name = _actor(user)
    project = _get_accessible(project_id, user_id)
    if not project_store.can_manage_members(project, user_id):
        raise HTTPException(status_code=403, detail="Only the owner can invite members.")

    settings = get_settings()
    recipient_user_id = ""
    recipient_researcher_id = ""
    recipient_email = payload.email or ""
    recipient_name = ""

    if payload.researcher_id:
        try:
            rid = normalise_researcher_id(payload.researcher_id)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        target = find_by_researcher_id(settings, rid)
        if target is None:
            raise HTTPException(status_code=404, detail="No researcher found with that ID.")
        target = ensure_researcher_id(settings, target)
        recipient_user_id = str(target.get("user_id") or "")
        recipient_researcher_id = str(target.get("researcher_id") or "")
        recipient_email = str(target.get("email") or "")
        recipient_name = str(target.get("name") or "")
        if recipient_user_id == user_id:
            raise HTTPException(status_code=400, detail="You cannot invite yourself.")
    elif not recipient_email:
        raise HTTPException(
            status_code=400,
            detail="Enter a Researcher ID to invite a collaborator.",
        )

    try:
        invite, token, updated = project_store.create_invite(
            project_id,
            email=recipient_email,
            role=payload.role,
            invited_by=user_id,
            invited_by_email=email,
            invited_by_name=name,
            recipient_user_id=recipient_user_id,
            recipient_researcher_id=recipient_researcher_id,
            recipient_name=recipient_name,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return InviteCreateResponse(invite=invite, token=token, project=updated)


@router.delete("/{project_id}/invites/{invite_id}", response_model=ResearchProject)
def revoke_invite(project_id: str, invite_id: str, request: Request) -> ResearchProject:
    user = _require_user(request)
    user_id, email, name = _actor(user)
    project = _get_accessible(project_id, user_id)
    if not project_store.can_manage_members(project, user_id):
        raise HTTPException(status_code=403, detail="Only the owner can revoke invites.")
    try:
        return project_store.revoke_invite(
            project_id,
            invite_id,
            actor_id=user_id,
            actor_email=email,
            actor_name=name,
        )
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.patch("/{project_id}/members/{member_user_id}", response_model=ResearchProject)
def update_member_role(
    project_id: str,
    member_user_id: str,
    payload: MemberRoleUpdate,
    request: Request,
) -> ResearchProject:
    user = _require_user(request)
    user_id, email, name = _actor(user)
    project = _get_accessible(project_id, user_id)
    if not project_store.can_manage_members(project, user_id):
        raise HTTPException(status_code=403, detail="Only the owner can change roles.")
    try:
        return project_store.update_member_role(
            project_id,
            member_user_id,
            payload.role,
            actor_id=user_id,
            actor_email=email,
            actor_name=name,
        )
    except ValueError as exc:
        detail = str(exc)
        code = 404 if "not found" in detail.lower() else 400
        raise HTTPException(status_code=code, detail=detail) from exc


@router.delete("/{project_id}/members/{member_user_id}", response_model=ResearchProject)
def remove_member(project_id: str, member_user_id: str, request: Request) -> ResearchProject:
    user = _require_user(request)
    user_id, email, name = _actor(user)
    project = _get_accessible(project_id, user_id)
    if not project_store.can_manage_members(project, user_id):
        raise HTTPException(status_code=403, detail="Only the owner can remove members.")
    try:
        return project_store.remove_member(
            project_id,
            member_user_id,
            actor_id=user_id,
            actor_email=email,
            actor_name=name,
        )
    except ValueError as exc:
        detail = str(exc)
        code = 404 if "not found" in detail.lower() else 400
        raise HTTPException(status_code=code, detail=detail) from exc


def _enrich_sources(project: ResearchProject) -> list:
    """Refresh live Library status onto associations without inventing biblio."""
    kb = get_knowledge_base()
    out = []
    for src in project.sources or []:
        doc = kb.store.get_document(src.document_id)
        if doc is None:
            out.append(src.model_copy(update={"source_status": "missing"}))
            continue
        title = (doc.title or doc.name or src.title or "").strip()
        authors = list(doc.authors or src.authors or [])
        year = doc.year if doc.year is not None else src.year
        doi = (doc.doi or src.doi or "").strip()
        out.append(
            src.model_copy(
                update={
                    "title": title or src.title,
                    "authors": authors,
                    "year": year,
                    "doi": doi,
                    "source_status": "linked",
                }
            )
        )
    return out


@router.get("/{project_id}/sources", response_model=ProjectSourceListResponse)
def list_project_sources(project_id: str, request: Request) -> ProjectSourceListResponse:
    user = _require_user(request)
    project = _get_accessible(project_id, str(user["user_id"]))
    sources = _enrich_sources(project)
    return ProjectSourceListResponse(sources=sources, total=len(sources))


@router.post("/{project_id}/sources", response_model=ProjectSourceMutationResponse)
def link_project_source(
    project_id: str, payload: ProjectSourceLinkRequest, request: Request
) -> ProjectSourceMutationResponse:
    user = _require_user(request)
    user_id, email, name = _actor(user)
    project = _get_accessible(project_id, user_id)
    if not project_store.can_manage_sources(project, user_id):
        raise HTTPException(
            status_code=403,
            detail="Only owners and editors can add project sources.",
        )
    doc_id = (payload.document_id or "").strip()
    kb = get_knowledge_base()
    document = kb.store.get_document(doc_id)
    if document is None:
        raise HTTPException(status_code=404, detail="Library document not found.")
    source = project_store.snapshot_source_from_document(
        document,
        added_by=user_id,
        added_by_name=name,
        added_by_email=email,
    )
    try:
        entry, updated = project_store.add_source(
            project_id,
            source,
            actor_id=user_id,
            actor_email=email,
            actor_name=name,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return ProjectSourceMutationResponse(source=entry, project=updated)


@router.post("/{project_id}/sources/import", response_model=ProjectSourceMutationResponse)
def import_and_link_project_source(
    project_id: str, payload: ProjectSourceImportRequest, request: Request
) -> ProjectSourceMutationResponse:
    """Find Papers → Library ingest → project link (no second file copy).

    Reuses an existing Library document when DOI (or title+year) already matches,
    so repeated imports do not duplicate underlying files or source rows.
    """
    user = _require_user(request)
    user_id, email, name = _actor(user)
    project = _get_accessible(project_id, user_id)
    if not project_store.can_manage_sources(project, user_id):
        raise HTTPException(
            status_code=403,
            detail="Only owners and editors can add project sources.",
        )
    kb = get_knowledge_base()
    warnings: list[str] = []

    # Prefer linking an existing Library / project source over re-ingesting.
    existing = _find_reusable_library_document(
        kb,
        project,
        doi=payload.doi,
        title=payload.title,
        year=payload.year,
    )
    if existing is not None:
        already = next(
            (s for s in (project.sources or []) if s.document_id == existing.document_id),
            None,
        )
        if already is not None:
            return ProjectSourceMutationResponse(
                source=already,
                project=project,
                warnings=["Source was already linked to this project."],
            )
        source = project_store.snapshot_source_from_document(
            existing,
            added_by=user_id,
            added_by_name=name,
            added_by_email=email,
            external_url=(payload.url or "").strip(),
        )
        try:
            entry, updated = project_store.add_source(
                project_id,
                source,
                actor_id=user_id,
                actor_email=email,
                actor_name=name,
            )
        except ValueError as exc:
            if "already linked" in str(exc).lower():
                sources = project_store.list_sources(project_id)
                match = next(s for s in sources if s.document_id == source.document_id)
                refreshed = project_store.get_project(project_id)
                if refreshed is None:
                    raise HTTPException(status_code=404, detail="Project not found.")
                return ProjectSourceMutationResponse(
                    source=match,
                    project=refreshed,
                    warnings=["Source was already linked to this project."],
                )
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return ProjectSourceMutationResponse(
            source=entry,
            project=updated,
            warnings=["Linked existing Library document (skipped re-import)."],
        )

    import_req = PaperImportRequest(
        paper_id=payload.paper_id,
        source=payload.source,  # type: ignore[arg-type]
        title=payload.title,
        authors=payload.authors,
        year=payload.year,
        venue=payload.venue,
        abstract=payload.abstract,
        summary=payload.summary,
        doi=payload.doi,
        pdf_url=payload.pdf_url,
        url=payload.url,
        result_kind=payload.result_kind,  # type: ignore[arg-type]
    )
    try:
        result = import_paper(import_req, kb)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover - network/IO
        raise HTTPException(status_code=502, detail=f"Import failed: {exc}") from exc

    warnings.extend(list(result.warnings or []))
    external = (payload.url or result.paper_url or result.pdf_url or "").strip()
    source = project_store.snapshot_source_from_document(
        result.document,
        added_by=user_id,
        added_by_name=name,
        added_by_email=email,
        external_url=external,
    )
    try:
        entry, updated = project_store.add_source(
            project_id,
            source,
            actor_id=user_id,
            actor_email=email,
            actor_name=name,
        )
    except ValueError as exc:
        # Already linked after a prior import — return existing association.
        if "already linked" in str(exc).lower():
            sources = project_store.list_sources(project_id)
            match = next(s for s in sources if s.document_id == source.document_id)
            refreshed = project_store.get_project(project_id)
            if refreshed is None:
                raise HTTPException(status_code=404, detail="Project not found.")
            return ProjectSourceMutationResponse(
                source=match,
                project=refreshed,
                warnings=warnings + ["Source was already linked to this project."],
            )
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return ProjectSourceMutationResponse(
        source=entry,
        project=updated,
        warnings=warnings,
    )


def _normalize_doi(value: str | None) -> str:
    raw = (value or "").strip().lower()
    if not raw:
        return ""
    for prefix in ("https://doi.org/", "http://doi.org/", "doi:"):
        if raw.startswith(prefix):
            raw = raw[len(prefix) :]
            break
    return raw.strip()


def _normalize_title(value: str | None) -> str:
    return " ".join((value or "").lower().split())


def _find_reusable_library_document(kb, project, *, doi: str | None, title: str | None, year: int | None):
    """Return an existing Library document (or project source snapshot) to link instead of re-importing."""
    ndoi = _normalize_doi(doi)
    ntitle = _normalize_title(title)
    # Prefer an already-linked project source with the same DOI / title.
    for src in project.sources or []:
        if ndoi and _normalize_doi(src.doi) == ndoi:
            live = kb.store.get_document(src.document_id)
            return live or src
        if ntitle and _normalize_title(src.title) == ntitle:
            if year is None or src.year is None or src.year == year:
                live = kb.store.get_document(src.document_id)
                return live or src
    try:
        docs = list(kb.store.list_documents() or [])
    except Exception:
        return None
    if ndoi:
        for doc in docs:
            if _normalize_doi(getattr(doc, "doi", None)) == ndoi:
                return doc
    if ntitle:
        for doc in docs:
            doc_title = _normalize_title(
                getattr(doc, "title", None) or getattr(doc, "name", None)
            )
            if doc_title != ntitle:
                continue
            doc_year = getattr(doc, "year", None)
            if year is None or doc_year is None or doc_year == year:
                return doc
    return None


@router.delete("/{project_id}/sources/{document_id}", response_model=ResearchProject)
def unlink_project_source(
    project_id: str, document_id: str, request: Request
) -> ResearchProject:
    user = _require_user(request)
    user_id, email, name = _actor(user)
    project = _get_accessible(project_id, user_id)
    if not project_store.can_manage_sources(project, user_id):
        raise HTTPException(
            status_code=403,
            detail="Only owners and editors can remove project sources.",
        )
    try:
        return project_store.remove_source(
            project_id,
            document_id,
            actor_id=user_id,
            actor_email=email,
            actor_name=name,
        )
    except ValueError as exc:
        detail = str(exc)
        code = 404 if "not" in detail.lower() else 400
        raise HTTPException(status_code=code, detail=detail) from exc


@router.post("/{project_id}/notes", response_model=ResearchProject)
def add_note(project_id: str, payload: NoteCreateRequest, request: Request) -> ResearchProject:
    user = _require_user(request)
    user_id, email, name = _actor(user)
    project = _get_accessible(project_id, user_id)
    if not project_store.can_edit_own_notes(project, user_id):
        raise HTTPException(
            status_code=403,
            detail="Only owners and editors can add research notes.",
        )
    try:
        return project_store.add_note(
            project_id,
            text=payload.text,
            note_type=payload.note_type,
            document_id=payload.document_id,
            evidence_id=payload.evidence_id,
            author_id=user_id,
            author_email=email,
            author_name=name,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.patch("/{project_id}/notes/{note_id}", response_model=ResearchProject)
def update_note(
    project_id: str, note_id: str, payload: NoteUpdateRequest, request: Request
) -> ResearchProject:
    user = _require_user(request)
    user_id, email, name = _actor(user)
    project = _get_accessible(project_id, user_id)
    if not project_store.can_edit_own_notes(project, user_id):
        raise HTTPException(status_code=403, detail="Not allowed to edit notes.")
    try:
        return project_store.update_note(
            project_id,
            note_id,
            actor_id=user_id,
            actor_email=email,
            actor_name=name,
            is_owner=project_store.can_manage_all_notes(project, user_id),
            text=payload.text,
            note_type=payload.note_type,
            document_id=payload.document_id,
            evidence_id=payload.evidence_id,
        )
    except ValueError as exc:
        code = 404 if "not found" in str(exc).lower() else 400
        raise HTTPException(status_code=code, detail=str(exc)) from exc
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc


@router.delete("/{project_id}/notes/{note_id}", response_model=ResearchProject)
def delete_note(project_id: str, note_id: str, request: Request) -> ResearchProject:
    user = _require_user(request)
    user_id, email, name = _actor(user)
    project = _get_accessible(project_id, user_id)
    if not project_store.can_edit_own_notes(project, user_id):
        raise HTTPException(status_code=403, detail="Not allowed to delete notes.")
    try:
        return project_store.delete_note(
            project_id,
            note_id,
            actor_id=user_id,
            actor_email=email,
            actor_name=name,
            is_owner=project_store.can_manage_all_notes(project, user_id),
        )
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc


def _validate_evidence_quote(
    *,
    document_id: str,
    chunk_id: str,
    quote: str,
) -> tuple[str, int | None, str, list[str], int | None, str, str]:
    """Ensure quote comes from stored chunk text. Returns snapshot + resolved chunk_id."""
    kb = get_knowledge_base()
    doc = kb.store.get_document(document_id)
    if doc is None:
        raise HTTPException(status_code=404, detail="Library document not found.")
    cleaned = (quote or "").strip()
    cid = (chunk_id or "").strip()
    page: int | None = None
    if cid:
        chunk = kb.store.get_chunk(cid)
        if chunk is None:
            raise HTTPException(status_code=400, detail="chunk_id not found in Library.")
        if getattr(chunk.metadata, "document_id", None) != document_id:
            raise HTTPException(status_code=400, detail="chunk_id does not belong to document.")
        page = getattr(chunk.metadata, "page", None)
        if cleaned:
            compact_chunk = " ".join((chunk.text or "").split())
            compact_quote = " ".join(cleaned.split())
            if compact_quote not in compact_chunk:
                raise HTTPException(
                    status_code=400,
                    detail="Quote must appear in the selected chunk text.",
                )
        else:
            cleaned = (chunk.text or "").strip()[:2000]
    elif cleaned:
        found = False
        for chunk in kb.store.all_chunks(document_ids=[document_id]):
            compact_chunk = " ".join((chunk.text or "").split())
            compact_quote = " ".join(cleaned.split())
            if compact_quote and compact_quote in compact_chunk:
                found = True
                cid = chunk.chunk_id
                page = getattr(chunk.metadata, "page", None)
                break
        if not found:
            raise HTTPException(
                status_code=400,
                detail="Quote must appear in a stored chunk for this document.",
            )
    else:
        raise HTTPException(status_code=400, detail="quote or chunk_id is required.")
    return (
        cleaned,
        page,
        (doc.title or doc.name or "").strip(),
        list(doc.authors or []),
        doc.year,
        (doc.doi or "").strip(),
        cid,
    )


@router.get("/{project_id}/evidence", response_model=EvidenceListResponse)
def list_evidence(project_id: str, request: Request) -> EvidenceListResponse:
    user = _require_user(request)
    user_id, _, _ = _actor(user)
    project = _get_accessible(project_id, user_id)
    items = list(project.evidence or [])
    return EvidenceListResponse(evidence=items, total=len(items))


@router.post("/{project_id}/evidence", response_model=ResearchProject)
def add_evidence(
    project_id: str, payload: EvidenceCreateRequest, request: Request
) -> ResearchProject:
    user = _require_user(request)
    user_id, email, name = _actor(user)
    project = _get_accessible(project_id, user_id)
    if not project_store.can_manage_evidence(project, user_id):
        raise HTTPException(
            status_code=403,
            detail="Only owners and editors can add evidence.",
        )
    quote, page, title, authors, year, doi, chunk_id = _validate_evidence_quote(
        document_id=payload.document_id,
        chunk_id=payload.chunk_id,
        quote=payload.quote,
    )
    try:
        return project_store.add_evidence(
            project_id,
            document_id=payload.document_id,
            quote=quote,
            chunk_id=chunk_id,
            kind=payload.kind,
            section_key=payload.section_key,
            claim_ref=payload.claim_ref,
            author_id=user_id,
            author_email=email,
            author_name=name,
            title=title,
            authors=authors,
            year=year,
            doi=doi,
            page=page,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.delete("/{project_id}/evidence/{evidence_id}", response_model=ResearchProject)
def remove_evidence(
    project_id: str, evidence_id: str, request: Request
) -> ResearchProject:
    user = _require_user(request)
    user_id, email, name = _actor(user)
    project = _get_accessible(project_id, user_id)
    if not project_store.can_manage_evidence(project, user_id):
        raise HTTPException(
            status_code=403,
            detail="Only owners and editors can remove evidence.",
        )
    try:
        return project_store.remove_evidence(
            project_id,
            evidence_id,
            actor_id=user_id,
            actor_email=email,
            actor_name=name,
        )
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/{project_id}/manuscript", response_model=ProjectManuscript)
def get_manuscript(project_id: str, request: Request) -> ProjectManuscript:
    user = _require_user(request)
    user_id, _, _ = _actor(user)
    _get_accessible(project_id, user_id)
    ms = project_store.get_manuscript(project_id)
    if ms is None:
        raise HTTPException(status_code=404, detail="Manuscript not found.")
    return ms


@router.post("/{project_id}/manuscript", response_model=ProjectManuscript)
def create_manuscript(
    project_id: str, payload: ManuscriptCreateRequest, request: Request
) -> ProjectManuscript:
    user = _require_user(request)
    user_id, email, name = _actor(user)
    project = _get_accessible(project_id, user_id)
    if not project_store.can_create_manuscript(project, user_id):
        raise HTTPException(
            status_code=403,
            detail="Only the project owner can create a manuscript.",
        )
    try:
        ms, _ = project_store.create_manuscript(
            project_id,
            template=payload.template,
            document_type=payload.document_type or payload.template,
            title=payload.title,
            authors=payload.authors,
            affiliations=payload.affiliations,
            actor_id=user_id,
            actor_email=email,
            actor_name=name,
        )
        return ms
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.patch("/{project_id}/manuscript", response_model=ProjectManuscript)
def update_manuscript_meta(
    project_id: str, payload: ManuscriptUpdateRequest, request: Request
) -> ProjectManuscript:
    user = _require_user(request)
    user_id, email, name = _actor(user)
    project = _get_accessible(project_id, user_id)
    if not project_store.can_edit_manuscript_meta(project, user_id):
        raise HTTPException(status_code=403, detail="Only the owner can update manuscript metadata.")
    try:
        return project_store.update_manuscript_meta(
            project_id,
            actor_id=user_id,
            actor_email=email,
            actor_name=name,
            title=payload.title,
            authors=payload.authors,
            affiliations=payload.affiliations,
            status=payload.status,
            document_type=payload.document_type,
        )
    except ValueError as exc:
        code = 404 if "not found" in str(exc).lower() else 400
        raise HTTPException(status_code=code, detail=str(exc)) from exc


@router.post("/{project_id}/manuscript/sections", response_model=ProjectManuscript)
def add_manuscript_section(
    project_id: str, payload: ManuscriptSectionCreateRequest, request: Request
) -> ProjectManuscript:
    user = _require_user(request)
    user_id, email, name = _actor(user)
    _get_accessible(project_id, user_id)
    try:
        return project_store.add_manuscript_section(
            project_id,
            title=payload.title,
            key=payload.key,
            after_section_id=payload.after_section_id,
            actor_id=user_id,
            actor_email=email,
            actor_name=name,
        )
    except ValueError as exc:
        code = 404 if "not found" in str(exc).lower() else 400
        raise HTTPException(status_code=code, detail=str(exc)) from exc
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc


@router.patch("/{project_id}/manuscript/sections/{section_id}", response_model=ProjectManuscript)
def patch_manuscript_section(
    project_id: str,
    section_id: str,
    payload: ManuscriptSectionPatchRequest,
    request: Request,
) -> ProjectManuscript:
    user = _require_user(request)
    user_id, email, name = _actor(user)
    _get_accessible(project_id, user_id)
    try:
        return project_store.patch_manuscript_section(
            project_id,
            section_id,
            actor_id=user_id,
            actor_email=email,
            actor_name=name,
            title=payload.title,
            body=payload.body,
            status=payload.status,
            save_version=payload.save_version,
            version_summary=payload.version_summary,
        )
    except ValueError as exc:
        code = 404 if "not found" in str(exc).lower() else 400
        raise HTTPException(status_code=code, detail=str(exc)) from exc
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc


@router.put("/{project_id}/manuscript/sections/order", response_model=ProjectManuscript)
def reorder_manuscript_sections(
    project_id: str, payload: ManuscriptSectionOrderRequest, request: Request
) -> ProjectManuscript:
    user = _require_user(request)
    user_id, email, name = _actor(user)
    _get_accessible(project_id, user_id)
    try:
        return project_store.reorder_manuscript_sections(
            project_id,
            payload.section_ids,
            actor_id=user_id,
            actor_email=email,
            actor_name=name,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc


@router.delete("/{project_id}/manuscript/sections/{section_id}", response_model=ProjectManuscript)
def delete_manuscript_section(
    project_id: str, section_id: str, request: Request
) -> ProjectManuscript:
    user = _require_user(request)
    user_id, email, name = _actor(user)
    _get_accessible(project_id, user_id)
    try:
        return project_store.delete_manuscript_section(
            project_id,
            section_id,
            actor_id=user_id,
            actor_email=email,
            actor_name=name,
        )
    except ValueError as exc:
        code = 404 if "not found" in str(exc).lower() else 400
        raise HTTPException(status_code=code, detail=str(exc)) from exc
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc


@router.patch("/{project_id}/manuscript/sections", response_model=ProjectManuscript)
def update_manuscript_sections(
    project_id: str,
    payload: list[ManuscriptSectionUpdate],
    request: Request,
) -> ProjectManuscript:
    """Legacy batch patch by section key."""
    user = _require_user(request)
    user_id, email, name = _actor(user)
    _get_accessible(project_id, user_id)
    updates = [item.model_dump(exclude_unset=True) for item in payload]
    try:
        return project_store.update_manuscript_sections(
            project_id,
            updates,
            actor_id=user_id,
            actor_email=email,
            actor_name=name,
        )
    except ValueError as exc:
        code = 404 if "not found" in str(exc).lower() else 400
        raise HTTPException(status_code=code, detail=str(exc)) from exc
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc


@router.post("/{project_id}/manuscript/assign", response_model=ProjectManuscript)
def assign_manuscript_section(
    project_id: str, payload: ManuscriptAssignRequest, request: Request
) -> ProjectManuscript:
    user = _require_user(request)
    user_id, email, name = _actor(user)
    project = _get_accessible(project_id, user_id)
    if not project_store.can_manage(project, user_id):
        raise HTTPException(status_code=403, detail="Only the owner can assign sections.")
    try:
        return project_store.assign_manuscript_section(
            project_id,
            key=payload.key,
            section_id=payload.section_id,
            assignee_user_id=payload.assignee_user_id,
            actor_id=user_id,
            actor_email=email,
            actor_name=name,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc


@router.post(
    "/{project_id}/manuscript/sections/{section_id}/comments",
    response_model=ProjectManuscript,
)
def add_section_comment(
    project_id: str, section_id: str, payload: SectionCommentRequest, request: Request
) -> ProjectManuscript:
    user = _require_user(request)
    user_id, email, name = _actor(user)
    project = _get_accessible(project_id, user_id)
    if not project_store.can_comment_manuscript_section(project, user_id):
        raise HTTPException(status_code=403, detail="Not allowed to comment on sections.")
    try:
        return project_store.add_section_comment(
            project_id,
            section_id,
            text=payload.text,
            author_id=user_id,
            author_email=email,
            author_name=name,
        )
    except ValueError as exc:
        code = 404 if "not found" in str(exc).lower() else 400
        raise HTTPException(status_code=code, detail=str(exc)) from exc
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc


@router.post("/{project_id}/manuscript/cite", response_model=ManuscriptCiteResponse)
def cite_manuscript_source(
    project_id: str, payload: ManuscriptCiteRequest, request: Request
) -> ManuscriptCiteResponse:
    user = _require_user(request)
    user_id, _, _ = _actor(user)
    _get_accessible(project_id, user_id)
    live = None
    try:
        kb = get_knowledge_base()
        live = kb.store.get_document(payload.document_id)
    except Exception:
        live = None
    try:
        result = project_store.cite_project_source(
            project_id,
            payload.document_id,
            style=payload.style,
            live_document=live,
        )
        return ManuscriptCiteResponse(**result)
    except ValueError as exc:
        code = 404 if "not" in str(exc).lower() else 400
        raise HTTPException(status_code=code, detail=str(exc)) from exc


@router.post("/{project_id}/manuscript/ai", response_model=ManuscriptAiResponse)
def manuscript_ai_action(
    project_id: str, payload: ManuscriptAiRequest, request: Request
) -> ManuscriptAiResponse:
    from ..services.projects.manuscript_ai import run_manuscript_ai

    user = _require_user(request)
    user_id, _, _ = _actor(user)
    project = _get_accessible(project_id, user_id)
    if not project_store.can_run_manuscript_ai(project, user_id):
        raise HTTPException(status_code=403, detail="Only owners and editors can run AI assists.")
    try:
        return run_manuscript_ai(project_id, payload)
    except ValueError as exc:
        code = 404 if "not found" in str(exc).lower() else 400
        raise HTTPException(status_code=code, detail=str(exc)) from exc


@router.get("/{project_id}/manuscript/review", response_model=ManuscriptReviewResponse)
def get_manuscript_review(project_id: str, request: Request) -> ManuscriptReviewResponse:
    from ..services.projects.manuscript_review import LIMITATIONS

    user = _require_user(request)
    user_id, _, _ = _actor(user)
    _get_accessible(project_id, user_id)
    ms = project_store.get_manuscript(project_id)
    if ms is None:
        raise HTTPException(status_code=404, detail="Manuscript not found.")
    issues = list(ms.review_issues or [])
    return ManuscriptReviewResponse(
        manuscript=ms,
        issues=issues,
        open_count=sum(1 for i in issues if i.state == "open"),
        warnings=[],
        limitations=list(LIMITATIONS),
    )


@router.post("/{project_id}/manuscript/review", response_model=ManuscriptReviewResponse)
def run_manuscript_review(
    project_id: str, payload: ManuscriptReviewRunRequest, request: Request
) -> ManuscriptReviewResponse:
    from ..services.projects import manuscript_review as review_mod

    user = _require_user(request)
    user_id, _, name = _actor(user)
    project = _get_accessible(project_id, user_id)
    if not review_mod.can_run_review(project, user_id):
        raise HTTPException(
            status_code=403,
            detail="Only owners and editors can run evidence/citation review.",
        )
    try:
        return review_mod.run_manuscript_review(
            project_id,
            section_id=payload.section_id,
            actor_id=user_id,
            actor_name=name,
        )
    except ValueError as exc:
        code = 404 if "not found" in str(exc).lower() else 400
        raise HTTPException(status_code=code, detail=str(exc)) from exc


@router.patch(
    "/{project_id}/manuscript/review/{issue_id}",
    response_model=ManuscriptReviewResponse,
)
def update_manuscript_review_issue(
    project_id: str,
    issue_id: str,
    payload: ManuscriptReviewIssueUpdate,
    request: Request,
) -> ManuscriptReviewResponse:
    from ..services.projects import manuscript_review as review_mod
    from ..services.projects.manuscript_review import LIMITATIONS

    user = _require_user(request)
    user_id, _, name = _actor(user)
    project = _get_accessible(project_id, user_id)
    if not review_mod.can_update_review_issue(project, user_id):
        raise HTTPException(status_code=403, detail="Not allowed to update review issues.")
    try:
        ms = project_store.update_manuscript_review_issue(
            project_id,
            issue_id,
            state=payload.state,
            resolution_note=payload.resolution_note,
            actor_id=user_id,
            actor_name=name,
        )
    except ValueError as exc:
        code = 404 if "not found" in str(exc).lower() else 400
        raise HTTPException(status_code=code, detail=str(exc)) from exc
    issues = list(ms.review_issues or [])
    return ManuscriptReviewResponse(
        manuscript=ms,
        issues=issues,
        open_count=sum(1 for i in issues if i.state == "open"),
        warnings=[],
        limitations=list(LIMITATIONS),
    )


@router.get("/{project_id}/manuscript/checklist", response_model=ManuscriptChecklistResponse)
def get_manuscript_checklist(
    project_id: str, request: Request
) -> ManuscriptChecklistResponse:
    from ..services.projects import manuscript_checklist as checklist_mod

    user = _require_user(request)
    user_id, _, _ = _actor(user)
    _get_accessible(project_id, user_id)
    try:
        return checklist_mod.get_checklist_response(project_id)
    except ValueError as exc:
        code = 404 if "not found" in str(exc).lower() else 400
        raise HTTPException(status_code=code, detail=str(exc)) from exc


@router.post("/{project_id}/manuscript/checklist/refresh", response_model=ManuscriptChecklistResponse)
def refresh_manuscript_checklist(
    project_id: str, request: Request
) -> ManuscriptChecklistResponse:
    from ..services.projects import manuscript_checklist as checklist_mod

    user = _require_user(request)
    user_id, _, _ = _actor(user)
    project = _get_accessible(project_id, user_id)
    if not checklist_mod.can_edit_checklist(project, user_id):
        raise HTTPException(status_code=403, detail="Not allowed to refresh checklist.")
    try:
        return checklist_mod.refresh_checklist_response(project_id)
    except ValueError as exc:
        code = 404 if "not found" in str(exc).lower() else 400
        raise HTTPException(status_code=code, detail=str(exc)) from exc


@router.patch(
    "/{project_id}/manuscript/checklist/{item_key}",
    response_model=ManuscriptChecklistResponse,
)
def update_manuscript_checklist_item(
    project_id: str,
    item_key: str,
    payload: ChecklistItemUpdateRequest,
    request: Request,
) -> ManuscriptChecklistResponse:
    from ..services.projects import manuscript_checklist as checklist_mod

    user = _require_user(request)
    user_id, _, name = _actor(user)
    project = _get_accessible(project_id, user_id)
    if not checklist_mod.can_edit_checklist(project, user_id):
        raise HTTPException(status_code=403, detail="Not allowed to update checklist.")
    try:
        return checklist_mod.update_checklist_item(
            project_id,
            item_key,
            status=payload.status,
            actor_id=user_id,
            actor_name=name,
            note=payload.note,
        )
    except ValueError as exc:
        code = 404 if "not found" in str(exc).lower() else 400
        raise HTTPException(status_code=code, detail=str(exc)) from exc


@router.post("/{project_id}/tasks", response_model=ResearchProject)
def create_task(
    project_id: str, payload: TaskCreateRequest, request: Request
) -> ResearchProject:
    user = _require_user(request)
    user_id, email, name = _actor(user)
    project = _get_accessible(project_id, user_id)
    if not project_store.can_assign_tasks(project, user_id):
        # Owners assign; editors may create tasks assigned to themselves
        member = project_store.member_for(project, user_id)
        if member is None or member.role != "editor":
            raise HTTPException(status_code=403, detail="Not allowed to create tasks.")
        if payload.assignee_user_id and payload.assignee_user_id != user_id:
            raise HTTPException(
                status_code=403,
                detail="Editors can only assign tasks to themselves.",
            )
    try:
        return project_store.create_task(
            project_id,
            title=payload.title,
            description=payload.description,
            assignee_user_id=payload.assignee_user_id or (
                user_id if not project_store.can_assign_tasks(project, user_id) else ""
            ),
            document_id=payload.document_id,
            evidence_id=payload.evidence_id,
            section_key=payload.section_key,
            creator_id=user_id,
            creator_name=name,
            actor_email=email,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.patch("/{project_id}/tasks/{task_id}", response_model=ResearchProject)
def update_task(
    project_id: str, task_id: str, payload: TaskUpdateRequest, request: Request
) -> ResearchProject:
    user = _require_user(request)
    user_id, email, name = _actor(user)
    _get_accessible(project_id, user_id)
    try:
        return project_store.update_task(
            project_id,
            task_id,
            actor_id=user_id,
            actor_email=email,
            actor_name=name,
            title=payload.title,
            description=payload.description,
            status=payload.status,
            assignee_user_id=payload.assignee_user_id,
            document_id=payload.document_id,
            evidence_id=payload.evidence_id,
            section_key=payload.section_key,
        )
    except ValueError as exc:
        code = 404 if "not found" in str(exc).lower() else 400
        raise HTTPException(status_code=code, detail=str(exc)) from exc
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc


@router.post("/{project_id}/tasks/{task_id}/comments", response_model=ResearchProject)
def add_task_comment(
    project_id: str, task_id: str, payload: TaskCommentRequest, request: Request
) -> ResearchProject:
    user = _require_user(request)
    user_id, email, name = _actor(user)
    _get_accessible(project_id, user_id)
    try:
        return project_store.add_task_comment(
            project_id,
            task_id,
            text=payload.text,
            author_id=user_id,
            author_email=email,
            author_name=name,
        )
    except ValueError as exc:
        code = 404 if "not found" in str(exc).lower() else 400
        raise HTTPException(status_code=code, detail=str(exc)) from exc


@router.post("/{project_id}/reviews", response_model=ResearchProject)
def add_review(
    project_id: str, payload: ReviewCreateRequest, request: Request
) -> ResearchProject:
    user = _require_user(request)
    user_id, email, name = _actor(user)
    project = _get_accessible(project_id, user_id)
    if not project_store.can_review(project, user_id):
        raise HTTPException(
            status_code=403,
            detail="Viewers cannot leave reviews. Ask the owner for reviewer access.",
        )
    try:
        return project_store.add_review(
            project_id,
            text=payload.text,
            author_id=user_id,
            author_email=email,
            author_name=name,
            target=payload.target,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.delete("/{project_id}/reviews/{review_id}", response_model=ResearchProject)
def delete_review(project_id: str, review_id: str, request: Request) -> ResearchProject:
    user = _require_user(request)
    user_id, email, name = _actor(user)
    project = _get_accessible(project_id, user_id)
    is_owner = project_store.can_manage(project, user_id)
    member = project_store.member_for(project, user_id)
    if member is None:
        raise HTTPException(status_code=404, detail="Project not found.")
    try:
        return project_store.delete_review(
            project_id,
            review_id,
            actor_id=user_id,
            actor_email=email,
            actor_name=name,
            is_owner=is_owner,
        )
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
