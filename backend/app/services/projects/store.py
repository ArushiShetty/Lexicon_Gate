"""Per-project JSON persistence with membership, invites, notes, and activity.

Projects: ``backend/data/projects/{project_id}.json``
Invite secrets: ``backend/data/projects/_invites/{invite_id}.json`` (token hash only).

Invite tokens are returned once at creation and accepted via POST body — never
query-string URLs. Path traversal is blocked by id sanitization.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import secrets
import threading
import uuid
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterator, Iterable

from ...config import BACKEND_ROOT
from ...models.projects import (
    AssignableRole,
    MANUSCRIPT_TEMPLATES,
    ManuscriptChecklist,
    ManuscriptReviewIssue,
    ManuscriptSection,
    ManuscriptTemplate,
    ManuscriptVersion,
    NoteType,
    NOTE_TYPES,
    PendingInviteForUser,
    ProjectActivity,
    ProjectEvidence,
    ProjectInvitePublic,
    ProjectManuscript,
    ProjectMember,
    ProjectNote,
    ProjectReview,
    ProjectSource,
    ProjectStatus,
    ProjectTask,
    ResearchProject,
    ReviewIssueState,
    SECTION_STATUSES,
    SectionComment,
    SectionStatus,
    SectionVersion,
    TaskComment,
    TaskStatus,
    TASK_STATUSES,
)

_ROOT = BACKEND_ROOT / "data" / "projects"
_INVITES_ROOT = _ROOT / "_invites"
_MANUSCRIPTS_ROOT = _ROOT / "manuscripts"
_LOCK = threading.RLock()
_LOCK_STATE = threading.local()
_MAX_ACTIVITY = 200
_INVITE_TTL_DAYS = 14
_MAX_MANUSCRIPT_VERSIONS = 50


@contextmanager
def _store_lock() -> Iterator[None]:
    """Reentrant process + cross-process lock for project/invite JSON I/O."""
    depth = getattr(_LOCK_STATE, "depth", 0)
    _ROOT.mkdir(parents=True, exist_ok=True)
    lock_path = _ROOT / ".store.lock"
    with _LOCK:
        if depth == 0:
            fh = lock_path.open("a+")
            try:
                import fcntl

                fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
            except (ImportError, OSError):
                pass
            _LOCK_STATE.fh = fh
        _LOCK_STATE.depth = depth + 1
        try:
            yield
        finally:
            _LOCK_STATE.depth = depth
            if depth == 0:
                fh = getattr(_LOCK_STATE, "fh", None)
                if fh is not None:
                    try:
                        import fcntl

                        fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
                    except (ImportError, OSError):
                        pass
                    fh.close()
                    _LOCK_STATE.fh = None



def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _utcnow_iso() -> str:
    return _utcnow().isoformat()


def _hash_token(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def _verify_token(token: str, token_hash: str) -> bool:
    return hmac.compare_digest(_hash_token(token), token_hash or "")


def _safe_id(value: str, *, label: str = "id") -> str:
    safe = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in (value or ""))
    if not safe or safe != value:
        raise ValueError(f"Invalid {label}.")
    return safe


def _project_path(project_id: str) -> Path:
    safe = _safe_id(project_id, label="project id")
    root = _ROOT.resolve()
    root.mkdir(parents=True, exist_ok=True)
    path = (root / f"{safe}.json").resolve()
    if path.parent != root:
        raise ValueError("Invalid project path.")
    return path


def _invite_path(invite_id: str) -> Path:
    safe = _safe_id(invite_id, label="invite id")
    root = _INVITES_ROOT.resolve()
    root.mkdir(parents=True, exist_ok=True)
    path = (root / f"{safe}.json").resolve()
    if path.parent != root:
        raise ValueError("Invalid invite path.")
    return path


def _read_raw(project_id: str) -> dict | None:
    path = _project_path(project_id)
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _write_raw(project: ResearchProject) -> None:
    path = _project_path(project.project_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(project.model_dump(mode="json"), indent=2) + "\n",
        encoding="utf-8",
    )


def _read_invite_secret(invite_id: str) -> dict | None:
    path = _invite_path(invite_id)
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _write_invite_secret(payload: dict[str, Any]) -> None:
    path = _invite_path(str(payload["invite_id"]))
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _delete_invite_secret(invite_id: str) -> None:
    path = _invite_path(invite_id)
    if path.exists():
        path.unlink()


def _sync_project_invite_status(project_id: str, invite_id: str, status: str) -> None:
    """Keep project.invites[].status aligned with the invite secret (e.g. expired)."""
    raw = _read_raw(project_id)
    if raw is None:
        return
    project = _normalize_project(ResearchProject.model_validate(raw))
    changed = False
    updated: list[ProjectInvitePublic] = []
    for inv in project.invites or []:
        if inv.invite_id == invite_id and inv.status != status:
            updated.append(inv.model_copy(update={"status": status}))  # type: ignore[arg-type]
            changed = True
        else:
            updated.append(inv)
    if not changed:
        return
    project.invites = updated
    project.updated_at = _utcnow_iso()
    _write_raw(project)


def _expire_invite(secret: dict[str, Any]) -> None:
    secret["status"] = "expired"
    _write_invite_secret(secret)
    project_id = str(secret.get("project_id") or "")
    invite_id = str(secret.get("invite_id") or "")
    if project_id and invite_id:
        _sync_project_invite_status(project_id, invite_id, "expired")


def _iter_all() -> Iterable[ResearchProject]:
    root = _ROOT.resolve()
    if not root.exists():
        return
    for path in sorted(root.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            yield _normalize_project(ResearchProject.model_validate(payload))
        except Exception:
            continue


def _normalize_project(project: ResearchProject) -> ResearchProject:
    """Fill defaults for projects persisted before Phase 2; sync sources ↔ document_ids."""
    if project.notes is None:
        project.notes = []
    if project.reviews is None:
        project.reviews = []
    if project.activity is None:
        project.activity = []
    if project.invites is None:
        project.invites = []
    if project.sources is None:
        project.sources = []
    if project.document_ids is None:
        project.document_ids = []
    if project.evidence is None:
        project.evidence = []
    if project.tasks is None:
        project.tasks = []

    # Migrate legacy document_ids-only projects into sources associations.
    if project.document_ids and not project.sources:
        project.sources = [
            ProjectSource(document_id=str(doc_id), source_status="linked")
            for doc_id in project.document_ids
            if doc_id
        ]
    _sync_document_ids(project)
    return project


def _sync_document_ids(project: ResearchProject) -> None:
    """Keep document_ids derived from sources for Ask / legacy callers."""
    project.document_ids = list(
        dict.fromkeys(str(s.document_id) for s in (project.sources or []) if s.document_id)
    )


def member_for(project: ResearchProject, user_id: str) -> ProjectMember | None:
    uid = str(user_id or "")
    for member in project.members or []:
        if member.user_id == uid:
            return member
    if project.owner_id == uid:
        return ProjectMember(user_id=uid, role="owner")
    return None


def can_view(project: ResearchProject, user_id: str) -> bool:
    return member_for(project, user_id) is not None


def can_edit_metadata(project: ResearchProject, user_id: str) -> bool:
    member = member_for(project, user_id)
    return member is not None and member.role in ("owner", "editor")


def can_contribute_notes(project: ResearchProject, user_id: str) -> bool:
    """Editors (and owners) may add research notes."""
    member = member_for(project, user_id)
    return member is not None and member.role in ("owner", "editor")


def can_review(project: ResearchProject, user_id: str) -> bool:
    """Reviewers leave review comments; they cannot edit project metadata."""
    member = member_for(project, user_id)
    return member is not None and member.role in ("owner", "editor", "reviewer")


def can_manage(project: ResearchProject, user_id: str) -> bool:
    member = member_for(project, user_id)
    return member is not None and member.role == "owner"


def can_manage_members(project: ResearchProject, user_id: str) -> bool:
    return can_manage(project, user_id)


def can_manage_sources(project: ResearchProject, user_id: str) -> bool:
    """Owners and editors may link/unlink shared project sources."""
    member = member_for(project, user_id)
    return member is not None and member.role in ("owner", "editor")


def can_edit_own_notes(project: ResearchProject, user_id: str) -> bool:
    member = member_for(project, user_id)
    return member is not None and member.role in ("owner", "editor")


def can_manage_all_notes(project: ResearchProject, user_id: str) -> bool:
    return can_manage(project, user_id)


def can_manage_evidence(project: ResearchProject, user_id: str) -> bool:
    member = member_for(project, user_id)
    return member is not None and member.role in ("owner", "editor")


def can_create_manuscript(project: ResearchProject, user_id: str) -> bool:
    return can_manage(project, user_id)


def can_edit_manuscript_meta(project: ResearchProject, user_id: str) -> bool:
    return can_manage(project, user_id)


def can_edit_manuscript_section(
    project: ResearchProject, user_id: str, section: ManuscriptSection
) -> bool:
    member = member_for(project, user_id)
    if member is None:
        return False
    if member.role == "owner":
        return True
    if member.role != "editor":
        return False
    if not section.assignee_user_id or section.assignee_user_id == user_id:
        return True
    return False


def can_comment_manuscript_section(project: ResearchProject, user_id: str) -> bool:
    member = member_for(project, user_id)
    return member is not None and member.role in ("owner", "editor", "reviewer")


def can_run_manuscript_ai(project: ResearchProject, user_id: str) -> bool:
    member = member_for(project, user_id)
    return member is not None and member.role in ("owner", "editor")


def can_assign_tasks(project: ResearchProject, user_id: str) -> bool:
    return can_manage(project, user_id)


def can_update_task(project: ResearchProject, user_id: str, task: ProjectTask) -> bool:
    member = member_for(project, user_id)
    if member is None:
        return False
    if member.role == "owner":
        return True
    if member.role == "editor":
        return True
    if member.role == "reviewer" and task.assignee_user_id == user_id:
        return True
    return False


def _append_activity(
    project: ResearchProject,
    *,
    type: str,
    actor_id: str = "",
    actor_email: str = "",
    actor_name: str = "",
    message: str = "",
    meta: dict[str, Any] | None = None,
) -> None:
    entry = ProjectActivity(
        activity_id=uuid.uuid4().hex,
        type=type,
        actor_id=actor_id,
        actor_email=actor_email,
        actor_name=actor_name,
        message=message,
        meta=meta or {},
        created_at=_utcnow_iso(),
    )
    project.activity = [entry, *(project.activity or [])][:_MAX_ACTIVITY]


def create_project(
    *,
    owner_id: str,
    owner_email: str = "",
    owner_name: str = "",
    title: str,
    topic: str = "",
    description: str = "",
) -> ResearchProject:
    text = (title or "").strip()
    if not text:
        raise ValueError("Title is required.")
    now = _utcnow_iso()
    project_id = uuid.uuid4().hex
    owner = ProjectMember(
        user_id=str(owner_id),
        email=(owner_email or "").strip().lower(),
        name=(owner_name or "").strip(),
        role="owner",
    )
    project = ResearchProject(
        project_id=project_id,
        title=text,
        topic=(topic or "").strip(),
        description=(description or "").strip(),
        owner_id=str(owner_id),
        members=[owner],
        status="active",
        document_ids=[],
        sources=[],
        evidence=[],
        evidence_notes=[],
        notes=[],
        reviews=[],
        tasks=[],
        activity=[],
        invites=[],
        manuscript_id=None,
        created_at=now,
        updated_at=now,
    )
    _append_activity(
        project,
        type="project_created",
        actor_id=owner.user_id,
        actor_email=owner.email,
        actor_name=owner.name,
        message=f"Created project “{project.title}”.",
    )
    with _store_lock():
        _write_raw(project)
    return project


def get_project(project_id: str) -> ResearchProject | None:
    with _store_lock():
        raw = _read_raw(project_id)
    if raw is None:
        return None
    try:
        return _normalize_project(ResearchProject.model_validate(raw))
    except Exception:
        return None


def list_projects_for_user(
    user_id: str,
    *,
    limit: int = 50,
    include_archived: bool = True,
) -> list[ResearchProject]:
    uid = str(user_id or "")
    out: list[ResearchProject] = []
    with _store_lock():
        for project in _iter_all():
            if not can_view(project, uid):
                continue
            if not include_archived and project.status == "archived":
                continue
            out.append(project)
            if len(out) >= limit:
                break
    out.sort(key=lambda p: p.updated_at or p.created_at or "", reverse=True)
    return out[:limit]


def update_project(
    project_id: str,
    *,
    title: str | None = None,
    topic: str | None = None,
    description: str | None = None,
    status: ProjectStatus | None = None,
    document_ids: list[str] | None = None,
    actor_id: str = "",
    actor_email: str = "",
    actor_name: str = "",
    log_activity: bool = True,
) -> ResearchProject | None:
    with _store_lock():
        raw = _read_raw(project_id)
        if raw is None:
            return None
        project = _normalize_project(ResearchProject.model_validate(raw))
        changed: list[str] = []
        if title is not None:
            cleaned = title.strip()
            if not cleaned:
                raise ValueError("Title is required.")
            if cleaned != project.title:
                project.title = cleaned
                changed.append("title")
        if topic is not None and topic.strip() != project.topic:
            project.topic = topic.strip()
            changed.append("topic")
        if description is not None and description.strip() != project.description:
            project.description = description.strip()
            changed.append("description")
        if status is not None and status != project.status:
            project.status = status
            changed.append("status")
        if document_ids is not None:
            # Full replace: preserve existing association metadata when IDs overlap.
            existing = {s.document_id: s for s in (project.sources or [])}
            new_sources: list[ProjectSource] = []
            for doc_id in dict.fromkeys(str(d) for d in document_ids if d):
                if doc_id in existing:
                    new_sources.append(existing[doc_id])
                else:
                    new_sources.append(
                        ProjectSource(
                            document_id=doc_id,
                            added_by=actor_id,
                            added_by_email=actor_email,
                            added_by_name=actor_name,
                            added_at=_utcnow_iso(),
                            source_status="linked",
                        )
                    )
            project.sources = new_sources
            _sync_document_ids(project)
            changed.append("sources")
        project.updated_at = _utcnow_iso()
        if log_activity and changed:
            activity_type = "project_archived" if status == "archived" else "project_updated"
            if "sources" in changed:
                activity_type = "source_linked"
            _append_activity(
                project,
                type=activity_type,
                actor_id=actor_id,
                actor_email=actor_email,
                actor_name=actor_name,
                message=f"Updated {', '.join(changed)}.",
                meta={"fields": changed},
            )
        _write_raw(project)
        return project


def archive_project(
    project_id: str,
    *,
    actor_id: str = "",
    actor_email: str = "",
    actor_name: str = "",
) -> ResearchProject | None:
    return update_project(
        project_id,
        status="archived",
        actor_id=actor_id,
        actor_email=actor_email,
        actor_name=actor_name,
    )


def snapshot_source_from_document(
    document: Any,
    *,
    added_by: str = "",
    added_by_name: str = "",
    added_by_email: str = "",
    external_url: str = "",
) -> ProjectSource:
    """Build a ProjectSource from a Library Document without inventing biblio."""
    title = str(getattr(document, "title", "") or "").strip()
    name = str(getattr(document, "name", "") or "").strip()
    doi = str(getattr(document, "doi", "") or "").strip()
    authors = list(getattr(document, "authors", None) or [])
    year = getattr(document, "year", None)
    return ProjectSource(
        document_id=str(getattr(document, "document_id", "") or ""),
        added_by=added_by,
        added_by_name=added_by_name,
        added_by_email=(added_by_email or "").strip().lower(),
        added_at=_utcnow_iso(),
        title=title or name,
        authors=[str(a).strip() for a in authors if str(a).strip()],
        year=int(year) if year is not None else None,
        doi=doi,
        external_url=(external_url or "").strip(),
        source_status="linked",
    )


def list_sources(project_id: str) -> list[ProjectSource]:
    with _store_lock():
        raw = _read_raw(project_id)
        if raw is None:
            raise ValueError("Project not found.")
        project = _normalize_project(ResearchProject.model_validate(raw))
        return list(project.sources or [])


def add_source(
    project_id: str,
    source: ProjectSource,
    *,
    actor_id: str = "",
    actor_email: str = "",
    actor_name: str = "",
) -> tuple[ProjectSource, ResearchProject]:
    doc_id = str(source.document_id or "").strip()
    if not doc_id:
        raise ValueError("document_id is required.")
    with _store_lock():
        raw = _read_raw(project_id)
        if raw is None:
            raise ValueError("Project not found.")
        project = _normalize_project(ResearchProject.model_validate(raw))
        for existing in project.sources or []:
            if existing.document_id == doc_id:
                raise ValueError("That source is already linked to this project.")
        entry = source.model_copy(
            update={
                "document_id": doc_id,
                "added_by": source.added_by or actor_id,
                "added_by_email": (source.added_by_email or actor_email or "").strip().lower(),
                "added_by_name": source.added_by_name or actor_name,
                "added_at": source.added_at or _utcnow_iso(),
                "source_status": source.source_status or "linked",
            }
        )
        project.sources = [entry, *(project.sources or [])]
        _sync_document_ids(project)
        project.updated_at = _utcnow_iso()
        label = entry.title or entry.document_id
        _append_activity(
            project,
            type="source_linked",
            actor_id=actor_id,
            actor_email=actor_email,
            actor_name=actor_name,
            message=f"Linked source “{label}”.",
            meta={"document_id": doc_id},
        )
        _write_raw(project)
        return entry, project


def remove_source(
    project_id: str,
    document_id: str,
    *,
    actor_id: str = "",
    actor_email: str = "",
    actor_name: str = "",
) -> ResearchProject:
    doc_id = str(document_id or "").strip()
    if not doc_id:
        raise ValueError("document_id is required.")
    with _store_lock():
        raw = _read_raw(project_id)
        if raw is None:
            raise ValueError("Project not found.")
        project = _normalize_project(ResearchProject.model_validate(raw))
        before = list(project.sources or [])
        kept = [s for s in before if s.document_id != doc_id]
        if len(kept) == len(before):
            raise ValueError("Source not linked to this project.")
        removed = next(s for s in before if s.document_id == doc_id)
        project.sources = kept
        _sync_document_ids(project)
        project.updated_at = _utcnow_iso()
        label = removed.title or removed.document_id
        _append_activity(
            project,
            type="source_unlinked",
            actor_id=actor_id,
            actor_email=actor_email,
            actor_name=actor_name,
            message=f"Removed source “{label}” from the project (Library copy kept).",
            meta={"document_id": doc_id},
        )
        _write_raw(project)
        return project


def delete_project(project_id: str) -> bool:
    with _store_lock():
        raw = _read_raw(project_id)
        if raw is None:
            return False
        try:
            project = ResearchProject.model_validate(raw)
            for invite in project.invites or []:
                _delete_invite_secret(invite.invite_id)
        except Exception:
            pass
        path = _project_path(project_id)
        path.unlink()
        try:
            ms_path = _manuscript_path(project_id)
            if ms_path.exists():
                ms_path.unlink()
        except ValueError:
            pass
        return True


def create_invite(
    project_id: str,
    *,
    email: str = "",
    role: AssignableRole,
    invited_by: str,
    invited_by_email: str = "",
    invited_by_name: str = "",
    recipient_user_id: str = "",
    recipient_researcher_id: str = "",
    recipient_name: str = "",
) -> tuple[ProjectInvitePublic, str, ResearchProject]:
    email_n = (email or "").strip().lower()
    recipient_uid = (recipient_user_id or "").strip()
    if not email_n and not recipient_uid:
        raise ValueError("A Researcher ID or email is required.")
    if email_n and "@" not in email_n:
        raise ValueError("A valid email is required.")
    if role not in ("editor", "reviewer", "viewer"):
        raise ValueError("Invalid role.")
    with _store_lock():
        raw = _read_raw(project_id)
        if raw is None:
            raise ValueError("Project not found.")
        project = _normalize_project(ResearchProject.model_validate(raw))
        if recipient_uid and recipient_uid == invited_by:
            raise ValueError("You cannot invite yourself.")
        for member in project.members:
            if recipient_uid and member.user_id == recipient_uid:
                raise ValueError("That user is already a project member.")
            if email_n and (member.email or "").strip().lower() == email_n:
                raise ValueError("That user is already a project member.")
        for inv in project.invites:
            if inv.status != "pending":
                continue
            if recipient_uid and inv.recipient_user_id == recipient_uid:
                raise ValueError("A pending invite already exists for that researcher.")
            if email_n and inv.email == email_n:
                raise ValueError("A pending invite already exists for that email.")

        invite_id = uuid.uuid4().hex
        token = secrets.token_urlsafe(32)
        now = _utcnow()
        expires = now + timedelta(days=_INVITE_TTL_DAYS)
        public = ProjectInvitePublic(
            invite_id=invite_id,
            project_id=project_id,
            email=email_n,
            recipient_user_id=recipient_uid,
            recipient_researcher_id=(recipient_researcher_id or "").strip().upper(),
            role=role,
            status="pending",
            invited_by=invited_by,
            invited_by_email=invited_by_email,
            invited_by_name=invited_by_name,
            created_at=now.isoformat(),
            expires_at=expires.isoformat(),
        )
        _write_invite_secret(
            {
                "invite_id": invite_id,
                "project_id": project_id,
                "email": email_n,
                "recipient_user_id": recipient_uid,
                "recipient_researcher_id": public.recipient_researcher_id,
                "role": role,
                "token_hash": _hash_token(token),
                "status": "pending",
                "invited_by": invited_by,
                "invited_by_email": (invited_by_email or "").strip().lower(),
                "invited_by_name": invited_by_name,
                "created_at": now.isoformat(),
                "expires_at": expires.isoformat(),
            }
        )
        project.invites = [public, *list(project.invites or [])]
        project.updated_at = now.isoformat()
        who = public.recipient_researcher_id or email_n or recipient_uid[:8]
        _append_activity(
            project,
            type="member_invited",
            actor_id=invited_by,
            actor_email=invited_by_email,
            actor_name=invited_by_name,
            message=f"Invited {who} as {role}.",
            meta={
                "invite_id": invite_id,
                "email": email_n,
                "role": role,
                "recipient_user_id": recipient_uid,
            },
        )
        _write_raw(project)
        # For in-app Researcher ID invites, do not return the plaintext token to the client.
        return_token = "" if recipient_uid else token
        return public, return_token, project


def revoke_invite(
    project_id: str,
    invite_id: str,
    *,
    actor_id: str = "",
    actor_email: str = "",
    actor_name: str = "",
) -> ResearchProject:
    with _store_lock():
        raw = _read_raw(project_id)
        if raw is None:
            raise ValueError("Project not found.")
        project = _normalize_project(ResearchProject.model_validate(raw))
        found = False
        updated: list[ProjectInvitePublic] = []
        for inv in project.invites:
            if inv.invite_id == invite_id:
                found = True
                if inv.status == "pending":
                    inv = inv.model_copy(update={"status": "revoked"})
                updated.append(inv)
            else:
                updated.append(inv)
        if not found:
            raise ValueError("Invite not found.")
        secret = _read_invite_secret(invite_id)
        if secret:
            secret["status"] = "revoked"
            _write_invite_secret(secret)
        project.invites = updated
        project.updated_at = _utcnow_iso()
        _append_activity(
            project,
            type="member_removed",
            actor_id=actor_id,
            actor_email=actor_email,
            actor_name=actor_name,
            message=f"Revoked invite {invite_id[:8]}…",
            meta={"invite_id": invite_id},
        )
        _write_raw(project)
        return project


def list_pending_invites_for_user(
    *,
    user_id: str = "",
    email: str = "",
) -> list[PendingInviteForUser]:
    """Pending invites for this user by recipient_user_id and/or email."""
    email_n = (email or "").strip().lower()
    uid = (user_id or "").strip()
    if not email_n and not uid:
        return []
    out: list[PendingInviteForUser] = []
    now = _utcnow()
    with _store_lock():
        if not _INVITES_ROOT.exists():
            return []
        for path in _INVITES_ROOT.glob("*.json"):
            try:
                secret = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                continue
            if str(secret.get("status")) != "pending":
                continue
            secret_uid = str(secret.get("recipient_user_id") or "")
            secret_email = str(secret.get("email", "")).lower()
            match = False
            if uid and secret_uid and secret_uid == uid:
                match = True
            elif email_n and secret_email == email_n:
                match = True
            if not match:
                continue
            expires_at = str(secret.get("expires_at") or "")
            try:
                if expires_at and datetime.fromisoformat(expires_at) < now:
                    _expire_invite(secret)
                    continue
            except ValueError:
                pass
            project_id = str(secret.get("project_id") or "")
            raw = _read_raw(project_id) if project_id else None
            title = ""
            invited_by_email = str(secret.get("invited_by_email") or "").strip().lower()
            invited_by_name = str(secret.get("invited_by_name") or "").strip()
            if raw:
                try:
                    project = _normalize_project(ResearchProject.model_validate(raw))
                    title = project.title
                    if not invited_by_email or not invited_by_name:
                        for inv in project.invites or []:
                            if inv.invite_id == str(secret.get("invite_id") or ""):
                                invited_by_email = invited_by_email or (
                                    inv.invited_by_email or ""
                                ).strip().lower()
                                invited_by_name = invited_by_name or (
                                    inv.invited_by_name or ""
                                ).strip()
                                break
                except Exception:
                    pass
            can_in_app = bool(secret_uid and uid and secret_uid == uid)
            out.append(
                PendingInviteForUser(
                    invite_id=str(secret["invite_id"]),
                    project_id=project_id,
                    project_title=title,
                    email=secret_email or email_n,
                    role=secret.get("role") or "viewer",  # type: ignore[arg-type]
                    invited_by_email=invited_by_email,
                    invited_by_name=invited_by_name,
                    created_at=str(secret.get("created_at") or ""),
                    expires_at=expires_at,
                    can_accept_in_app=can_in_app,
                )
            )
    out.sort(key=lambda i: i.created_at, reverse=True)
    return out


def list_pending_invites_for_email(email: str) -> list[PendingInviteForUser]:
    return list_pending_invites_for_user(email=email)


def accept_invite(
    *,
    invite_id: str,
    token: str | None = None,
    user_id: str,
    user_email: str,
    user_name: str = "",
    user_researcher_id: str = "",
) -> ResearchProject:
    email_n = (user_email or "").strip().lower()
    token_s = (token or "").strip()
    with _store_lock():
        secret = _read_invite_secret(invite_id)
        if secret is None or str(secret.get("status")) != "pending":
            raise ValueError("Invite not found or no longer valid.")
        expires_at = str(secret.get("expires_at") or "")
        try:
            expired = bool(expires_at and datetime.fromisoformat(expires_at) < _utcnow())
        except ValueError:
            expired = False
        if expired:
            _expire_invite(secret)
            raise ValueError("Invite has expired.")

        recipient_uid = str(secret.get("recipient_user_id") or "")
        in_app = bool(recipient_uid and recipient_uid == str(user_id))
        if in_app:
            # Session-bound accept — no OOB token required.
            pass
        elif recipient_uid and recipient_uid != str(user_id):
            raise ValueError(
                "This invitation is for a different Lexicon Gate account. "
                "Sign in with the invited account to accept."
            )
        else:
            if not token_s:
                raise ValueError("Invite token is required.")
            if not _verify_token(token_s, str(secret.get("token_hash") or "")):
                raise ValueError("Invalid invite token.")
            if str(secret.get("email", "")).lower() != email_n:
                raise ValueError("Sign in with the invited email address to accept.")

        project_id = str(secret.get("project_id") or "")
        raw = _read_raw(project_id)
        if raw is None:
            raise ValueError("Project not found.")
        project = _normalize_project(ResearchProject.model_validate(raw))
        if any(m.user_id == user_id for m in project.members):
            secret["status"] = "accepted"
            _write_invite_secret(secret)
            project.invites = [
                inv.model_copy(update={"status": "accepted"})
                if inv.invite_id == invite_id
                else inv
                for inv in project.invites
            ]
            project.updated_at = _utcnow_iso()
            _write_raw(project)
            return project
        role = secret.get("role") or "viewer"
        if role not in ("editor", "reviewer", "viewer"):
            role = "viewer"
        member = ProjectMember(
            user_id=str(user_id),
            email=email_n,
            name=(user_name or "").strip(),
            role=role,  # type: ignore[arg-type]
            researcher_id=(user_researcher_id or "").strip().upper(),
        )
        project.members = [*list(project.members), member]
        project.invites = [
            inv.model_copy(update={"status": "accepted"})
            if inv.invite_id == invite_id
            else inv
            for inv in project.invites
        ]
        secret["status"] = "accepted"
        _write_invite_secret(secret)
        project.updated_at = _utcnow_iso()
        _append_activity(
            project,
            type="member_joined",
            actor_id=str(user_id),
            actor_email=email_n,
            actor_name=user_name,
            message=f"{email_n or user_id} joined as {role}.",
            meta={"role": role, "invite_id": invite_id},
        )
        _write_raw(project)
        return project


def reject_invite(
    *,
    invite_id: str,
    user_id: str,
    user_email: str = "",
) -> None:
    """Invitee rejects a pending invite addressed to them."""
    email_n = (user_email or "").strip().lower()
    with _store_lock():
        secret = _read_invite_secret(invite_id)
        if secret is None or str(secret.get("status")) != "pending":
            raise ValueError("Invite not found or no longer valid.")
        recipient_uid = str(secret.get("recipient_user_id") or "")
        secret_email = str(secret.get("email") or "").lower()
        allowed = False
        if recipient_uid and recipient_uid == str(user_id):
            allowed = True
        elif secret_email and email_n and secret_email == email_n:
            allowed = True
        if not allowed:
            raise ValueError("You are not the recipient of this invite.")
        expires_at = str(secret.get("expires_at") or "")
        try:
            expired = bool(expires_at and datetime.fromisoformat(expires_at) < _utcnow())
        except ValueError:
            expired = False
        if expired:
            _expire_invite(secret)
            raise ValueError("Invite has expired.")
        secret["status"] = "revoked"
        _write_invite_secret(secret)
        project_id = str(secret.get("project_id") or "")
        _sync_project_invite_status(project_id, invite_id, "revoked")


def update_member_role(
    project_id: str,
    member_user_id: str,
    role: AssignableRole,
    *,
    actor_id: str = "",
    actor_email: str = "",
    actor_name: str = "",
) -> ResearchProject:
    if role not in ("editor", "reviewer", "viewer"):
        raise ValueError("Invalid role.")
    with _store_lock():
        raw = _read_raw(project_id)
        if raw is None:
            raise ValueError("Project not found.")
        project = _normalize_project(ResearchProject.model_validate(raw))
        if member_user_id == project.owner_id:
            raise ValueError("Cannot change the owner role.")
        found = False
        new_members: list[ProjectMember] = []
        for member in project.members:
            if member.user_id == member_user_id:
                if member.role == "owner":
                    raise ValueError("Cannot change the owner role.")
                found = True
                new_members.append(member.model_copy(update={"role": role}))
            else:
                new_members.append(member)
        if not found:
            raise ValueError("Member not found.")
        project.members = new_members
        project.updated_at = _utcnow_iso()
        _append_activity(
            project,
            type="role_changed",
            actor_id=actor_id,
            actor_email=actor_email,
            actor_name=actor_name,
            message=f"Changed role for member to {role}.",
            meta={"member_user_id": member_user_id, "role": role},
        )
        _write_raw(project)
        return project


def remove_member(
    project_id: str,
    member_user_id: str,
    *,
    actor_id: str = "",
    actor_email: str = "",
    actor_name: str = "",
) -> ResearchProject:
    with _store_lock():
        raw = _read_raw(project_id)
        if raw is None:
            raise ValueError("Project not found.")
        project = _normalize_project(ResearchProject.model_validate(raw))
        if member_user_id == project.owner_id:
            raise ValueError("Cannot remove the project owner.")
        before = len(project.members)
        project.members = [m for m in project.members if m.user_id != member_user_id]
        if len(project.members) == before:
            raise ValueError("Member not found.")
        project.updated_at = _utcnow_iso()
        _append_activity(
            project,
            type="member_removed",
            actor_id=actor_id,
            actor_email=actor_email,
            actor_name=actor_name,
            message="Removed a project member.",
            meta={"member_user_id": member_user_id},
        )
        _write_raw(project)
        return project


def add_note(
    project_id: str,
    *,
    text: str,
    author_id: str,
    author_email: str = "",
    author_name: str = "",
    note_type: NoteType = "summary",
    document_id: str = "",
    evidence_id: str = "",
) -> ResearchProject:
    cleaned = (text or "").strip()
    if not cleaned:
        raise ValueError("Note text is required.")
    ntype: NoteType = note_type if note_type in NOTE_TYPES else "summary"
    with _store_lock():
        raw = _read_raw(project_id)
        if raw is None:
            raise ValueError("Project not found.")
        project = _normalize_project(ResearchProject.model_validate(raw))
        doc_id = (document_id or "").strip()
        ev_id = (evidence_id or "").strip()
        if doc_id and doc_id not in set(project.document_ids or []):
            raise ValueError("document_id must be a linked project source.")
        if ev_id and not any(e.evidence_id == ev_id for e in project.evidence or []):
            raise ValueError("evidence_id not found on this project.")
        now = _utcnow_iso()
        note = ProjectNote(
            note_id=uuid.uuid4().hex,
            text=cleaned,
            note_type=ntype,
            document_id=doc_id,
            evidence_id=ev_id,
            author_id=author_id,
            author_email=author_email,
            author_name=author_name,
            created_at=now,
            updated_at=now,
        )
        project.notes = [note, *list(project.notes or [])]
        project.updated_at = now
        _append_activity(
            project,
            type="note_added",
            actor_id=author_id,
            actor_email=author_email,
            actor_name=author_name,
            message="Added a research note.",
            meta={"note_id": note.note_id, "note_type": ntype, "document_id": doc_id},
        )
        _write_raw(project)
        return project


def update_note(
    project_id: str,
    note_id: str,
    *,
    actor_id: str,
    actor_email: str = "",
    actor_name: str = "",
    is_owner: bool = False,
    text: str | None = None,
    note_type: NoteType | None = None,
    document_id: str | None = None,
    evidence_id: str | None = None,
) -> ResearchProject:
    with _store_lock():
        raw = _read_raw(project_id)
        if raw is None:
            raise ValueError("Project not found.")
        project = _normalize_project(ResearchProject.model_validate(raw))
        note = next((n for n in project.notes if n.note_id == note_id), None)
        if note is None:
            raise ValueError("Note not found.")
        if not is_owner and note.author_id != actor_id:
            raise PermissionError("Not allowed to edit this note.")
        updates: dict[str, Any] = {}
        if text is not None:
            cleaned = text.strip()
            if not cleaned:
                raise ValueError("Note text is required.")
            updates["text"] = cleaned
        if note_type is not None:
            if note_type not in NOTE_TYPES:
                raise ValueError("Invalid note type.")
            updates["note_type"] = note_type
        if document_id is not None:
            doc_id = document_id.strip()
            if doc_id and doc_id not in set(project.document_ids or []):
                raise ValueError("document_id must be a linked project source.")
            updates["document_id"] = doc_id
        if evidence_id is not None:
            ev_id = evidence_id.strip()
            if ev_id and not any(e.evidence_id == ev_id for e in project.evidence or []):
                raise ValueError("evidence_id not found on this project.")
            updates["evidence_id"] = ev_id
        updates["updated_at"] = _utcnow_iso()
        updated = note.model_copy(update=updates)
        project.notes = [updated if n.note_id == note_id else n for n in project.notes]
        project.updated_at = updates["updated_at"]
        _append_activity(
            project,
            type="note_updated",
            actor_id=actor_id,
            actor_email=actor_email,
            actor_name=actor_name,
            message="Updated a research note.",
            meta={"note_id": note_id},
        )
        _write_raw(project)
        return project


def delete_note(
    project_id: str,
    note_id: str,
    *,
    actor_id: str,
    actor_email: str = "",
    actor_name: str = "",
    is_owner: bool = False,
) -> ResearchProject:
    with _store_lock():
        raw = _read_raw(project_id)
        if raw is None:
            raise ValueError("Project not found.")
        project = _normalize_project(ResearchProject.model_validate(raw))
        note = next((n for n in project.notes if n.note_id == note_id), None)
        if note is None:
            raise ValueError("Note not found.")
        if not is_owner and note.author_id != actor_id:
            raise PermissionError("Not allowed to delete this note.")
        project.notes = [n for n in project.notes if n.note_id != note_id]
        project.updated_at = _utcnow_iso()
        _append_activity(
            project,
            type="note_deleted",
            actor_id=actor_id,
            actor_email=actor_email,
            actor_name=actor_name,
            message="Deleted a research note.",
            meta={"note_id": note_id},
        )
        _write_raw(project)
        return project


def add_evidence(
    project_id: str,
    *,
    document_id: str,
    quote: str = "",
    chunk_id: str = "",
    kind: str = "user_marked",
    section_key: str = "",
    claim_ref: str = "",
    author_id: str,
    author_email: str = "",
    author_name: str = "",
    title: str = "",
    authors: list[str] | None = None,
    year: int | None = None,
    doi: str = "",
    page: int | None = None,
) -> ResearchProject:
    doc_id = (document_id or "").strip()
    if not doc_id:
        raise ValueError("document_id is required.")
    cleaned_quote = (quote or "").strip()
    with _store_lock():
        raw = _read_raw(project_id)
        if raw is None:
            raise ValueError("Project not found.")
        project = _normalize_project(ResearchProject.model_validate(raw))
        if doc_id not in set(project.document_ids or []):
            raise ValueError("document_id must be a linked project source.")
        norm_quote = " ".join(cleaned_quote.lower().split())
        for existing in project.evidence or []:
            if existing.document_id != doc_id:
                continue
            if " ".join((existing.quote or "").lower().split()) == norm_quote:
                raise ValueError(
                    "That evidence quote is already recorded for this source."
                )
        now = _utcnow_iso()
        item = ProjectEvidence(
            evidence_id=uuid.uuid4().hex,
            document_id=doc_id,
            chunk_id=(chunk_id or "").strip(),
            quote=cleaned_quote,
            kind=kind if kind in ("imported", "extracted", "user_marked") else "user_marked",  # type: ignore[arg-type]
            title=(title or "").strip(),
            authors=list(authors or []),
            year=year,
            doi=(doi or "").strip(),
            page=page,
            section_key=(section_key or "").strip(),
            claim_ref=(claim_ref or "").strip(),
            author_id=author_id,
            author_email=author_email,
            author_name=author_name,
            created_at=now,
            updated_at=now,
        )
        project.evidence = [item, *list(project.evidence or [])]
        project.updated_at = now
        _append_activity(
            project,
            type="evidence_added",
            actor_id=author_id,
            actor_email=author_email,
            actor_name=author_name,
            message="Added project evidence.",
            meta={"evidence_id": item.evidence_id, "document_id": doc_id},
        )
        _write_raw(project)
        return project


def remove_evidence(
    project_id: str,
    evidence_id: str,
    *,
    actor_id: str,
    actor_email: str = "",
    actor_name: str = "",
) -> ResearchProject:
    with _store_lock():
        raw = _read_raw(project_id)
        if raw is None:
            raise ValueError("Project not found.")
        project = _normalize_project(ResearchProject.model_validate(raw))
        before = len(project.evidence or [])
        project.evidence = [e for e in (project.evidence or []) if e.evidence_id != evidence_id]
        if len(project.evidence) == before:
            raise ValueError("Evidence not found.")
        project.updated_at = _utcnow_iso()
        _append_activity(
            project,
            type="evidence_removed",
            actor_id=actor_id,
            actor_email=actor_email,
            actor_name=actor_name,
            message="Removed project evidence.",
            meta={"evidence_id": evidence_id},
        )
        _write_raw(project)
        return project


def _manuscript_path(project_id: str) -> Path:
    safe = _safe_id(project_id, label="project id")
    root = _MANUSCRIPTS_ROOT.resolve()
    root.mkdir(parents=True, exist_ok=True)
    path = (root / f"{safe}.json").resolve()
    if path.parent != root:
        raise ValueError("Invalid manuscript path.")
    return path


def _read_manuscript_raw(project_id: str) -> dict | None:
    path = _manuscript_path(project_id)
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _write_manuscript(ms: ProjectManuscript) -> None:
    path = _manuscript_path(ms.project_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(ms.model_dump(mode="json"), indent=2) + "\n",
        encoding="utf-8",
    )


def _map_section_status(raw: str | None) -> SectionStatus:
    value = (raw or "draft").strip()
    if value in ("in_progress",):
        return "draft"
    if value in ("done",):
        return "approved"
    if value in SECTION_STATUSES:
        return value  # type: ignore[return-value]
    return "draft"


def _slug_section_key(title: str, existing: set[str]) -> str:
    base = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in (title or "").lower())
    base = base.strip("_") or "section"
    key = base[:60]
    if key not in existing:
        return key
    i = 2
    while f"{key}_{i}" in existing:
        i += 1
    return f"{key}_{i}"


def _normalize_manuscript(ms: ProjectManuscript) -> ProjectManuscript:
    """Migrate legacy manuscripts: statuses, section_id/order, document_type."""
    doc_type = ms.document_type or ms.template or "ieee_research"
    if doc_type not in MANUSCRIPT_TEMPLATES:
        doc_type = "ieee_research"
    ms.document_type = doc_type  # type: ignore[assignment]
    ms.template = doc_type  # type: ignore[assignment]
    ms.status = _map_section_status(ms.status)  # type: ignore[assignment]
    if ms.authors is None:
        ms.authors = []
    if ms.affiliations is None:
        ms.affiliations = []
    sections: list[ManuscriptSection] = []
    for idx, sec in enumerate(ms.sections or []):
        data = sec.model_dump() if hasattr(sec, "model_dump") else dict(sec)
        sid = str(data.get("section_id") or "").strip() or uuid.uuid4().hex
        status = _map_section_status(str(data.get("status") or "draft"))
        comments = data.get("comments") or []
        versions = data.get("versions") or []
        sections.append(
            ManuscriptSection(
                section_id=sid,
                key=str(data.get("key") or f"section_{idx}"),
                title=str(data.get("title") or f"Section {idx + 1}"),
                body=str(data.get("body") or ""),
                order=int(data.get("order") if data.get("order") is not None else idx),
                assignee_user_id=str(data.get("assignee_user_id") or ""),
                assignee_name=str(data.get("assignee_name") or ""),
                status=status,
                comments=[SectionComment.model_validate(c) for c in comments],
                versions=[SectionVersion.model_validate(v) for v in versions],
                updated_by=str(data.get("updated_by") or ""),
                updated_by_name=str(data.get("updated_by_name") or ""),
                updated_at=str(data.get("updated_at") or ""),
            )
        )
    sections.sort(key=lambda s: s.order)
    for idx, sec in enumerate(sections):
        sec.order = idx
    ms.sections = sections
    if ms.review_issues is None:
        ms.review_issues = []
    return ms


def get_manuscript(project_id: str) -> ProjectManuscript | None:
    raw = _read_manuscript_raw(project_id)
    if raw is None:
        return None
    try:
        # Tolerate legacy status values before validation.
        if isinstance(raw.get("status"), str):
            raw["status"] = _map_section_status(raw["status"])
        for sec in raw.get("sections") or []:
            if isinstance(sec, dict) and "status" in sec:
                sec["status"] = _map_section_status(str(sec.get("status")))
            if isinstance(sec, dict) and not sec.get("section_id"):
                sec["section_id"] = uuid.uuid4().hex
        ms = ProjectManuscript.model_validate(raw)
        return _normalize_manuscript(ms)
    except Exception:
        return None


def create_manuscript(
    project_id: str,
    *,
    template: ManuscriptTemplate = "ieee_research",
    document_type: ManuscriptTemplate | None = None,
    title: str = "",
    authors: list[str] | None = None,
    affiliations: list[str] | None = None,
    actor_id: str,
    actor_email: str = "",
    actor_name: str = "",
) -> tuple[ProjectManuscript, ResearchProject]:
    tpl: ManuscriptTemplate = document_type or template
    if tpl not in MANUSCRIPT_TEMPLATES:
        tpl = "ieee_research"
    with _store_lock():
        raw = _read_raw(project_id)
        if raw is None:
            raise ValueError("Project not found.")
        project = _normalize_project(ResearchProject.model_validate(raw))
        existing = get_manuscript(project_id)
        if existing is not None and project.manuscript_id:
            raise ValueError("Manuscript already exists for this project.")
        now = _utcnow_iso()
        sections: list[ManuscriptSection] = []
        for order, (key, sec_title) in enumerate(MANUSCRIPT_TEMPLATES[tpl]):
            sections.append(
                ManuscriptSection(
                    section_id=uuid.uuid4().hex,
                    key=key,
                    title=sec_title,
                    body="",
                    order=order,
                    status="draft",
                )
            )
        ms = ProjectManuscript(
            manuscript_id=uuid.uuid4().hex,
            project_id=project_id,
            template=tpl,
            document_type=tpl,
            title=(title or project.title or "").strip(),
            authors=[a.strip() for a in (authors or []) if a and str(a).strip()],
            affiliations=[a.strip() for a in (affiliations or []) if a and str(a).strip()],
            status="draft",
            sections=sections,
            versions=[],
            created_at=now,
            updated_at=now,
        )
        _write_manuscript(ms)
        project.manuscript_id = ms.manuscript_id
        project.updated_at = now
        _append_activity(
            project,
            type="manuscript_created",
            actor_id=actor_id,
            actor_email=actor_email,
            actor_name=actor_name,
            message=f"Created manuscript ({tpl}).",
            meta={"manuscript_id": ms.manuscript_id, "template": tpl},
        )
        _write_raw(project)
        return ms, project


def update_manuscript_meta(
    project_id: str,
    *,
    actor_id: str,
    actor_email: str = "",
    actor_name: str = "",
    title: str | None = None,
    authors: list[str] | None = None,
    affiliations: list[str] | None = None,
    status: str | None = None,
    document_type: ManuscriptTemplate | None = None,
) -> ProjectManuscript:
    with _store_lock():
        raw = _read_raw(project_id)
        if raw is None:
            raise ValueError("Project not found.")
        project = _normalize_project(ResearchProject.model_validate(raw))
        ms = get_manuscript(project_id)
        if ms is None:
            raise ValueError("Manuscript not found.")
        now = _utcnow_iso()
        patch: dict[str, Any] = {"updated_at": now}
        if title is not None:
            patch["title"] = title.strip()
        if authors is not None:
            patch["authors"] = [a.strip() for a in authors if a and str(a).strip()]
        if affiliations is not None:
            patch["affiliations"] = [a.strip() for a in affiliations if a and str(a).strip()]
        if status is not None:
            patch["status"] = _map_section_status(status)
        if document_type is not None and document_type in MANUSCRIPT_TEMPLATES:
            patch["document_type"] = document_type
            patch["template"] = document_type
        ms = ms.model_copy(update=patch)
        _write_manuscript(ms)
        project.updated_at = now
        _append_activity(
            project,
            type="manuscript_updated",
            actor_id=actor_id,
            actor_email=actor_email,
            actor_name=actor_name,
            message="Updated manuscript metadata.",
            meta={},
        )
        _write_raw(project)
        return ms


def _find_section(ms: ProjectManuscript, *, section_id: str = "", key: str = "") -> ManuscriptSection:
    sid = (section_id or "").strip()
    k = (key or "").strip()
    for sec in ms.sections:
        if sid and sec.section_id == sid:
            return sec
        if k and sec.key == k:
            return sec
    raise ValueError("Section not found.")


def add_manuscript_section(
    project_id: str,
    *,
    title: str,
    key: str = "",
    after_section_id: str = "",
    actor_id: str,
    actor_email: str = "",
    actor_name: str = "",
) -> ProjectManuscript:
    cleaned = (title or "").strip()
    if not cleaned:
        raise ValueError("Section title is required.")
    with _store_lock():
        raw = _read_raw(project_id)
        if raw is None:
            raise ValueError("Project not found.")
        project = _normalize_project(ResearchProject.model_validate(raw))
        if not can_edit_manuscript_meta(project, actor_id) and not can_manage_sources(project, actor_id):
            # Owners manage structure; editors may add custom sections when they can edit generally
            member = member_for(project, actor_id)
            if member is None or member.role not in ("owner", "editor"):
                raise PermissionError("Not allowed to add sections.")
        ms = get_manuscript(project_id)
        if ms is None:
            raise ValueError("Manuscript not found.")
        existing_keys = {s.key for s in ms.sections}
        sec_key = (key or "").strip() or _slug_section_key(cleaned, existing_keys)
        if sec_key in existing_keys:
            raise ValueError("Section key already exists.")
        now = _utcnow_iso()
        new_sec = ManuscriptSection(
            section_id=uuid.uuid4().hex,
            key=sec_key,
            title=cleaned,
            body="",
            order=len(ms.sections),
            status="draft",
            updated_by=actor_id,
            updated_by_name=actor_name,
            updated_at=now,
        )
        sections = list(ms.sections)
        after = (after_section_id or "").strip()
        if after:
            idx = next((i for i, s in enumerate(sections) if s.section_id == after), None)
            if idx is None:
                raise ValueError("after_section_id not found.")
            sections.insert(idx + 1, new_sec)
        else:
            sections.append(new_sec)
        for i, s in enumerate(sections):
            s.order = i
        ms.sections = sections
        ms.updated_at = now
        _write_manuscript(ms)
        project.updated_at = now
        _append_activity(
            project,
            type="manuscript_updated",
            actor_id=actor_id,
            actor_email=actor_email,
            actor_name=actor_name,
            message=f"Added section '{cleaned}'.",
            meta={"section_id": new_sec.section_id},
        )
        _write_raw(project)
        return ms


def patch_manuscript_section(
    project_id: str,
    section_id: str,
    *,
    actor_id: str,
    actor_email: str = "",
    actor_name: str = "",
    title: str | None = None,
    body: str | None = None,
    status: str | None = None,
    save_version: bool = True,
    version_summary: str = "",
) -> ProjectManuscript:
    with _store_lock():
        raw = _read_raw(project_id)
        if raw is None:
            raise ValueError("Project not found.")
        project = _normalize_project(ResearchProject.model_validate(raw))
        ms = get_manuscript(project_id)
        if ms is None:
            raise ValueError("Manuscript not found.")
        section = _find_section(ms, section_id=section_id)
        if not can_edit_manuscript_section(project, actor_id, section):
            raise PermissionError("Not allowed to edit this section.")
        now = _utcnow_iso()
        versions = list(section.versions or [])
        if save_version and (body is not None or title is not None):
            versions = [
                SectionVersion(
                    version_id=uuid.uuid4().hex,
                    body=section.body,
                    title=section.title,
                    saved_by=actor_id,
                    saved_by_name=actor_name,
                    saved_at=now,
                    summary=(version_summary or "Prior version").strip(),
                ),
                *versions,
            ][:_MAX_MANUSCRIPT_VERSIONS]
        patch: dict[str, Any] = {
            "updated_by": actor_id,
            "updated_by_name": actor_name,
            "updated_at": now,
            "versions": versions,
        }
        if title is not None:
            cleaned = title.strip()
            if not cleaned:
                raise ValueError("Section title is required.")
            patch["title"] = cleaned
        if body is not None:
            patch["body"] = body
        if status is not None:
            mapped = _map_section_status(status)
            if mapped not in SECTION_STATUSES:
                raise ValueError("Invalid section status.")
            patch["status"] = mapped
        updated = section.model_copy(update=patch)
        ms.sections = [updated if s.section_id == section.section_id else s for s in ms.sections]
        ms.updated_at = now
        _write_manuscript(ms)
        project.updated_at = now
        _append_activity(
            project,
            type="manuscript_updated",
            actor_id=actor_id,
            actor_email=actor_email,
            actor_name=actor_name,
            message=f"Updated section '{updated.title}'.",
            meta={"section_id": section.section_id},
        )
        _write_raw(project)
        return ms


def reorder_manuscript_sections(
    project_id: str,
    section_ids: list[str],
    *,
    actor_id: str,
    actor_email: str = "",
    actor_name: str = "",
) -> ProjectManuscript:
    with _store_lock():
        raw = _read_raw(project_id)
        if raw is None:
            raise ValueError("Project not found.")
        project = _normalize_project(ResearchProject.model_validate(raw))
        member = member_for(project, actor_id)
        if member is None or member.role not in ("owner", "editor"):
            raise PermissionError("Not allowed to reorder sections.")
        ms = get_manuscript(project_id)
        if ms is None:
            raise ValueError("Manuscript not found.")
        by_id = {s.section_id: s for s in ms.sections}
        if set(section_ids) != set(by_id.keys()):
            raise ValueError("section_ids must include every section exactly once.")
        ordered = []
        for i, sid in enumerate(section_ids):
            sec = by_id[sid]
            ordered.append(sec.model_copy(update={"order": i}))
        now = _utcnow_iso()
        ms.sections = ordered
        ms.updated_at = now
        _write_manuscript(ms)
        project.updated_at = now
        _append_activity(
            project,
            type="manuscript_updated",
            actor_id=actor_id,
            actor_email=actor_email,
            actor_name=actor_name,
            message="Reordered manuscript sections.",
            meta={},
        )
        _write_raw(project)
        return ms


def delete_manuscript_section(
    project_id: str,
    section_id: str,
    *,
    actor_id: str,
    actor_email: str = "",
    actor_name: str = "",
) -> ProjectManuscript:
    with _store_lock():
        raw = _read_raw(project_id)
        if raw is None:
            raise ValueError("Project not found.")
        project = _normalize_project(ResearchProject.model_validate(raw))
        if not can_manage(project, actor_id):
            raise PermissionError("Only the owner can delete sections.")
        ms = get_manuscript(project_id)
        if ms is None:
            raise ValueError("Manuscript not found.")
        section = _find_section(ms, section_id=section_id)
        remaining = [s for s in ms.sections if s.section_id != section_id]
        if not remaining:
            raise ValueError("Cannot delete the last section.")
        for i, s in enumerate(remaining):
            s.order = i
        now = _utcnow_iso()
        ms.sections = remaining
        ms.updated_at = now
        _write_manuscript(ms)
        project.updated_at = now
        _append_activity(
            project,
            type="manuscript_updated",
            actor_id=actor_id,
            actor_email=actor_email,
            actor_name=actor_name,
            message=f"Deleted section '{section.title}'.",
            meta={"section_id": section_id},
        )
        _write_raw(project)
        return ms


def update_manuscript_sections(
    project_id: str,
    updates: list[dict[str, Any]],
    *,
    actor_id: str,
    actor_email: str = "",
    actor_name: str = "",
    save_version: bool = True,
    version_summary: str = "",
) -> ProjectManuscript:
    """Legacy batch update by section key — preserves prior API."""
    ms = get_manuscript(project_id)
    if ms is None:
        raise ValueError("Manuscript not found.")
    for upd in updates:
        key = str(upd.get("key") or "").strip()
        section = _find_section(ms, key=key)
        if upd.get("body") is not None or upd.get("status") is not None:
            ms = patch_manuscript_section(
                project_id,
                section.section_id,
                actor_id=actor_id,
                actor_email=actor_email,
                actor_name=actor_name,
                body=upd.get("body"),
                status=upd.get("status"),
                save_version=save_version,
                version_summary=version_summary,
            )
        if "assignee_user_id" in upd:
            ms = assign_manuscript_section(
                project_id,
                section_id=section.section_id,
                key="",
                assignee_user_id=str(upd.get("assignee_user_id") or ""),
                actor_id=actor_id,
                actor_email=actor_email,
                actor_name=actor_name,
            )
    return get_manuscript(project_id) or ms


def assign_manuscript_section(
    project_id: str,
    *,
    key: str = "",
    section_id: str = "",
    assignee_user_id: str,
    actor_id: str,
    actor_email: str = "",
    actor_name: str = "",
) -> ProjectManuscript:
    with _store_lock():
        raw = _read_raw(project_id)
        if raw is None:
            raise ValueError("Project not found.")
        project = _normalize_project(ResearchProject.model_validate(raw))
        if not can_manage(project, actor_id):
            raise PermissionError("Only the owner can assign sections.")
        ms = get_manuscript(project_id)
        if ms is None:
            raise ValueError("Manuscript not found.")
        section = _find_section(ms, section_id=section_id, key=key)
        assignee = (assignee_user_id or "").strip()
        assignee_name = ""
        if assignee:
            member = member_for(project, assignee)
            if member is None:
                raise ValueError("Assignee must be a project member.")
            if member.role not in ("owner", "editor", "reviewer"):
                raise ValueError("Assignee must be owner, editor, or reviewer.")
            assignee_name = member.name or member.email or assignee
        now = _utcnow_iso()
        updated = section.model_copy(
            update={
                "assignee_user_id": assignee,
                "assignee_name": assignee_name,
                "updated_by": actor_id,
                "updated_by_name": actor_name,
                "updated_at": now,
            }
        )
        ms.sections = [updated if s.section_id == section.section_id else s for s in ms.sections]
        ms.updated_at = now
        _write_manuscript(ms)
        project.updated_at = now
        _append_activity(
            project,
            type="manuscript_updated",
            actor_id=actor_id,
            actor_email=actor_email,
            actor_name=actor_name,
            message=f"Assigned section '{section.title}'.",
            meta={"section_id": section.section_id, "assignee": assignee},
        )
        _write_raw(project)
        return ms


def add_section_comment(
    project_id: str,
    section_id: str,
    *,
    text: str,
    author_id: str,
    author_email: str = "",
    author_name: str = "",
) -> ProjectManuscript:
    cleaned = (text or "").strip()
    if not cleaned:
        raise ValueError("Comment text is required.")
    with _store_lock():
        raw = _read_raw(project_id)
        if raw is None:
            raise ValueError("Project not found.")
        project = _normalize_project(ResearchProject.model_validate(raw))
        if not can_comment_manuscript_section(project, author_id):
            raise PermissionError("Not allowed to comment on sections.")
        ms = get_manuscript(project_id)
        if ms is None:
            raise ValueError("Manuscript not found.")
        section = _find_section(ms, section_id=section_id)
        now = _utcnow_iso()
        comment = SectionComment(
            comment_id=uuid.uuid4().hex,
            text=cleaned,
            author_id=author_id,
            author_email=author_email,
            author_name=author_name,
            created_at=now,
        )
        updated = section.model_copy(
            update={"comments": [*list(section.comments or []), comment]}
        )
        ms.sections = [updated if s.section_id == section.section_id else s for s in ms.sections]
        ms.updated_at = now
        _write_manuscript(ms)
        project.updated_at = now
        _append_activity(
            project,
            type="comment_added",
            actor_id=author_id,
            actor_email=author_email,
            actor_name=author_name,
            message="Commented on a manuscript section.",
            meta={"section_id": section_id, "comment_id": comment.comment_id},
        )
        _write_raw(project)
        return ms


def cite_project_source(
    project_id: str,
    document_id: str,
    *,
    style: str = "apa",
    live_document: Any = None,
) -> dict[str, Any]:
    """Build citation strings from a linked project source. Never invent metadata."""
    from ...models.query import EvidenceItem
    from ...services.citations import format_apa, format_bibtex

    project = get_project(project_id)
    if project is None:
        raise ValueError("Project not found.")
    doc_id = (document_id or "").strip()
    source = next((s for s in (project.sources or []) if s.document_id == doc_id), None)
    if source is None:
        raise ValueError("document_id is not a linked project source.")
    title = (source.title or "").strip()
    authors = list(source.authors or [])
    year = source.year
    doi = (source.doi or "").strip()
    warnings: list[str] = []
    if live_document is not None:
        title = title or (getattr(live_document, "title", None) or getattr(live_document, "name", "") or "")
        if not authors:
            authors = list(getattr(live_document, "authors", None) or [])
        if year is None:
            year = getattr(live_document, "year", None)
        if not doi:
            doi = (getattr(live_document, "doi", None) or "") or ""
    if not title:
        title = "Untitled"
        warnings.append("Title missing in source metadata; using Untitled.")
    if not authors:
        warnings.append("Authors missing; citation uses Unknown.")
    if year is None:
        warnings.append("Year missing; citation uses n.d.")
    item = EvidenceItem(
        citation_id=1,
        chunk_id="",
        document_id=doc_id,
        document_name=title,
        title=title,
        authors=authors,
        year=year,
        doi=doi or None,
        text="",
    )
    apa = format_apa(item)
    bibtex = format_bibtex(item)
    citation = apa if style != "bibtex" else bibtex
    marker = f"[doc:{doc_id}]"
    return {
        "document_id": doc_id,
        "citation": citation,
        "apa": apa,
        "bibtex": bibtex,
        "marker": marker,
        "title": title,
        "authors": authors,
        "year": year,
        "doi": doi,
        "warnings": warnings,
    }


def save_manuscript_review_issues(
    project_id: str,
    *,
    issues: list[ManuscriptReviewIssue],
    ran_at: str,
    actor_id: str = "",
    actor_name: str = "",
) -> ProjectManuscript:
    with _store_lock():
        ms = get_manuscript(project_id)
        if ms is None:
            raise ValueError("Manuscript not found.")
        now = _utcnow_iso()
        ms.review_issues = list(issues)
        ms.review_ran_at = ran_at or now
        ms.updated_at = now
        _write_manuscript(ms)
        raw = _read_raw(project_id)
        if raw is not None:
            project = _normalize_project(ResearchProject.model_validate(raw))
            project.updated_at = now
            _append_activity(
                project,
                type="manuscript_updated",
                actor_id=actor_id,
                actor_name=actor_name,
                message="Ran evidence and citation review.",
                meta={"issue_count": len(issues)},
            )
            _write_raw(project)
        return ms


def update_manuscript_review_issue(
    project_id: str,
    issue_id: str,
    *,
    state: ReviewIssueState,
    resolution_note: str = "",
    actor_id: str = "",
    actor_name: str = "",
) -> ProjectManuscript:
    if state not in ("open", "dismissed", "resolved", "manually_checked"):
        raise ValueError("Invalid review issue state.")
    with _store_lock():
        ms = get_manuscript(project_id)
        if ms is None:
            raise ValueError("Manuscript not found.")
        issue = next((i for i in (ms.review_issues or []) if i.issue_id == issue_id), None)
        if issue is None:
            raise ValueError("Review issue not found.")
        now = _utcnow_iso()
        updated = issue.model_copy(
            update={
                "state": state,
                "resolution_note": (resolution_note or "").strip(),
                "updated_at": now,
                "resolved_by": actor_id if state != "open" else "",
                "resolved_by_name": actor_name if state != "open" else "",
                "resolved_at": now if state != "open" else "",
            }
        )
        ms.review_issues = [
            updated if i.issue_id == issue_id else i for i in (ms.review_issues or [])
        ]
        ms.updated_at = now
        _write_manuscript(ms)
        raw = _read_raw(project_id)
        if raw is not None:
            project = _normalize_project(ResearchProject.model_validate(raw))
            project.updated_at = now
            _append_activity(
                project,
                type="manuscript_updated",
                actor_id=actor_id,
                actor_name=actor_name,
                message=f"Marked review issue as {state}.",
                meta={"issue_id": issue_id, "state": state},
            )
            _write_raw(project)
        return ms


def save_manuscript_checklist(
    project_id: str,
    checklist: ManuscriptChecklist,
    *,
    actor_id: str = "",
    actor_name: str = "",
    activity_message: str = "",
) -> ProjectManuscript:
    with _store_lock():
        ms = get_manuscript(project_id)
        if ms is None:
            raise ValueError("Manuscript not found.")
        now = _utcnow_iso()
        ms.checklist = checklist
        ms.updated_at = now
        _write_manuscript(ms)
        if activity_message:
            raw = _read_raw(project_id)
            if raw is not None:
                project = _normalize_project(ResearchProject.model_validate(raw))
                project.updated_at = now
                _append_activity(
                    project,
                    type="manuscript_updated",
                    actor_id=actor_id,
                    actor_name=actor_name,
                    message=activity_message,
                    meta={},
                )
                _write_raw(project)
        return ms


def create_task(
    project_id: str,
    *,
    title: str,
    description: str = "",
    assignee_user_id: str = "",
    document_id: str = "",
    evidence_id: str = "",
    section_key: str = "",
    creator_id: str,
    creator_name: str = "",
    actor_email: str = "",
) -> ResearchProject:
    cleaned = (title or "").strip()
    if not cleaned:
        raise ValueError("Task title is required.")
    with _store_lock():
        raw = _read_raw(project_id)
        if raw is None:
            raise ValueError("Project not found.")
        project = _normalize_project(ResearchProject.model_validate(raw))
        assignee = (assignee_user_id or "").strip()
        assignee_name = ""
        if assignee:
            member = member_for(project, assignee)
            if member is None:
                raise ValueError("Assignee must be a project member.")
            assignee_name = member.name or member.email or assignee
        doc_id = (document_id or "").strip()
        if doc_id and doc_id not in set(project.document_ids or []):
            raise ValueError("document_id must be a linked project source.")
        ev_id = (evidence_id or "").strip()
        if ev_id and not any(e.evidence_id == ev_id for e in project.evidence or []):
            raise ValueError("evidence_id not found on this project.")
        now = _utcnow_iso()
        task = ProjectTask(
            task_id=uuid.uuid4().hex,
            title=cleaned,
            description=(description or "").strip(),
            assignee_user_id=assignee,
            assignee_name=assignee_name,
            creator_id=creator_id,
            creator_name=creator_name,
            status="todo",
            document_id=doc_id,
            evidence_id=ev_id,
            section_key=(section_key or "").strip(),
            comments=[],
            created_at=now,
            updated_at=now,
        )
        project.tasks = [task, *list(project.tasks or [])]
        project.updated_at = now
        _append_activity(
            project,
            type="task_created",
            actor_id=creator_id,
            actor_email=actor_email,
            actor_name=creator_name,
            message=f"Created task: {cleaned}",
            meta={"task_id": task.task_id},
        )
        _write_raw(project)
        return project


def update_task(
    project_id: str,
    task_id: str,
    *,
    actor_id: str,
    actor_email: str = "",
    actor_name: str = "",
    title: str | None = None,
    description: str | None = None,
    status: TaskStatus | None = None,
    assignee_user_id: str | None = None,
    document_id: str | None = None,
    evidence_id: str | None = None,
    section_key: str | None = None,
) -> ResearchProject:
    with _store_lock():
        raw = _read_raw(project_id)
        if raw is None:
            raise ValueError("Project not found.")
        project = _normalize_project(ResearchProject.model_validate(raw))
        task = next((t for t in (project.tasks or []) if t.task_id == task_id), None)
        if task is None:
            raise ValueError("Task not found.")
        if not can_update_task(project, actor_id, task):
            raise PermissionError("Not allowed to update this task.")
        patch: dict[str, Any] = {}
        if title is not None:
            cleaned = title.strip()
            if not cleaned:
                raise ValueError("Task title is required.")
            patch["title"] = cleaned
        if description is not None:
            patch["description"] = description.strip()
        if status is not None:
            if status not in TASK_STATUSES:
                raise ValueError("Invalid task status.")
            patch["status"] = status
        if assignee_user_id is not None:
            if not can_assign_tasks(project, actor_id):
                raise PermissionError("Only the owner can reassign tasks.")
            assignee = assignee_user_id.strip()
            assignee_name = ""
            if assignee:
                member = member_for(project, assignee)
                if member is None:
                    raise ValueError("Assignee must be a project member.")
                assignee_name = member.name or member.email or assignee
            patch["assignee_user_id"] = assignee
            patch["assignee_name"] = assignee_name
        if document_id is not None:
            doc_id = document_id.strip()
            if doc_id and doc_id not in set(project.document_ids or []):
                raise ValueError("document_id must be a linked project source.")
            patch["document_id"] = doc_id
        if evidence_id is not None:
            ev_id = evidence_id.strip()
            if ev_id and not any(e.evidence_id == ev_id for e in project.evidence or []):
                raise ValueError("evidence_id not found on this project.")
            patch["evidence_id"] = ev_id
        if section_key is not None:
            patch["section_key"] = section_key.strip()
        patch["updated_at"] = _utcnow_iso()
        updated = task.model_copy(update=patch)
        project.tasks = [updated if t.task_id == task_id else t for t in project.tasks]
        project.updated_at = patch["updated_at"]
        _append_activity(
            project,
            type="task_updated",
            actor_id=actor_id,
            actor_email=actor_email,
            actor_name=actor_name,
            message="Updated a research task.",
            meta={"task_id": task_id},
        )
        _write_raw(project)
        return project


def add_task_comment(
    project_id: str,
    task_id: str,
    *,
    text: str,
    author_id: str,
    author_email: str = "",
    author_name: str = "",
) -> ResearchProject:
    cleaned = (text or "").strip()
    if not cleaned:
        raise ValueError("Comment text is required.")
    with _store_lock():
        raw = _read_raw(project_id)
        if raw is None:
            raise ValueError("Project not found.")
        project = _normalize_project(ResearchProject.model_validate(raw))
        task = next((t for t in (project.tasks or []) if t.task_id == task_id), None)
        if task is None:
            raise ValueError("Task not found.")
        now = _utcnow_iso()
        comment = TaskComment(
            comment_id=uuid.uuid4().hex,
            text=cleaned,
            author_id=author_id,
            author_email=author_email,
            author_name=author_name,
            created_at=now,
        )
        updated = task.model_copy(
            update={
                "comments": [*list(task.comments or []), comment],
                "updated_at": now,
            }
        )
        project.tasks = [updated if t.task_id == task_id else t for t in project.tasks]
        project.updated_at = now
        _append_activity(
            project,
            type="comment_added",
            actor_id=author_id,
            actor_email=author_email,
            actor_name=author_name,
            message="Commented on a task.",
            meta={"task_id": task_id, "comment_id": comment.comment_id},
        )
        _write_raw(project)
        return project


def add_review(
    project_id: str,
    *,
    text: str,
    author_id: str,
    author_email: str = "",
    author_name: str = "",
    target: str = "project",
) -> ResearchProject:
    cleaned = (text or "").strip()
    if not cleaned:
        raise ValueError("Review text is required.")
    with _store_lock():
        raw = _read_raw(project_id)
        if raw is None:
            raise ValueError("Project not found.")
        project = _normalize_project(ResearchProject.model_validate(raw))
        now = _utcnow_iso()
        review = ProjectReview(
            review_id=uuid.uuid4().hex,
            text=cleaned,
            author_id=author_id,
            author_email=author_email,
            author_name=author_name,
            target=(target or "project").strip() or "project",
            created_at=now,
        )
        project.reviews = [review, *list(project.reviews or [])]
        project.updated_at = now
        _append_activity(
            project,
            type="review_added",
            actor_id=author_id,
            actor_email=author_email,
            actor_name=author_name,
            message="Added a review comment.",
            meta={"review_id": review.review_id, "target": review.target},
        )
        _write_raw(project)
        return project


def delete_review(
    project_id: str,
    review_id: str,
    *,
    actor_id: str,
    actor_email: str = "",
    actor_name: str = "",
    is_owner: bool = False,
) -> ResearchProject:
    with _store_lock():
        raw = _read_raw(project_id)
        if raw is None:
            raise ValueError("Project not found.")
        project = _normalize_project(ResearchProject.model_validate(raw))
        review = next((r for r in project.reviews if r.review_id == review_id), None)
        if review is None:
            raise ValueError("Review not found.")
        if not is_owner and review.author_id != actor_id:
            raise PermissionError("Not allowed to delete this review.")
        project.reviews = [r for r in project.reviews if r.review_id != review_id]
        project.updated_at = _utcnow_iso()
        _append_activity(
            project,
            type="review_deleted",
            actor_id=actor_id,
            actor_email=actor_email,
            actor_name=actor_name,
            message="Deleted a review comment.",
            meta={"review_id": review_id},
        )
        _write_raw(project)
        return project
