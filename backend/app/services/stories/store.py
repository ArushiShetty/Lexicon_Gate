"""JSON persistence for Write / Stories.

Stories: ``backend/data/stories/{story_id}.json``
"""

from __future__ import annotations

import json
import re
import threading
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

from ...config import BACKEND_ROOT
from ...models.stories import (
    IEEE_SECTION_KEYS,
    PendingStoryInvite,
    Story,
    StoryCollaborator,
    StoryInvite,
    StoryListItem,
    assemble_ieee_body,
    empty_ieee_sections,
)

_ROOT = BACKEND_ROOT / "data" / "stories"
_LOCK = threading.RLock()
_LOCK_STATE = threading.local()

_SLUG_RE = re.compile(r"[^a-z0-9]+")
_EXCERPT_LEN = 220


@contextmanager
def _store_lock() -> Iterator[None]:
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


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_id(value: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in (value or ""))
    if not safe or safe != value:
        raise ValueError("Invalid story id.")
    return safe


def _path_for(story_id: str) -> Path:
    safe = _safe_id(story_id)
    root = _ROOT.resolve()
    path = (root / f"{safe}.json").resolve()
    if not str(path).startswith(str(root)):
        raise ValueError("Invalid story id.")
    return path


def slugify(title: str) -> str:
    base = _SLUG_RE.sub("-", (title or "").lower()).strip("-")
    return base[:80] or "story"


def make_excerpt(body_md: str) -> str:
    text = re.sub(r"[#>*`_\[\]()]+", " ", body_md or "")
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= _EXCERPT_LEN:
        return text
    return text[:_EXCERPT_LEN].rsplit(" ", 1)[0].strip() + "…"


def _normalize_sections(raw: dict[str, str] | None) -> dict[str, str]:
    base = empty_ieee_sections()
    if not raw:
        return base
    for key in IEEE_SECTION_KEYS:
        if key in raw and raw[key] is not None:
            base[key] = str(raw[key])
    return base


def _sync_body(story: Story) -> Story:
    if story.format == "ieee":
        story.body_md = assemble_ieee_body(
            title=story.title,
            authors_line=story.authors_line,
            affiliation=story.affiliation,
            sections=story.sections,
        )
    story.excerpt = make_excerpt(story.body_md)
    return story


def _has_writable_content(story: Story) -> bool:
    if (story.body_md or "").strip():
        return True
    return any((story.sections or {}).get(k, "").strip() for k in IEEE_SECTION_KEYS)


def _to_list_item(story: Story, *, my_role: str | None = None) -> StoryListItem:
    return StoryListItem(
        story_id=story.story_id,
        slug=story.slug,
        title=story.title,
        excerpt=story.excerpt,
        author_user_id=story.author_user_id,
        author_name=story.author_name,
        author_researcher_id=story.author_researcher_id,
        status=story.status,
        work_status=story.work_status,
        format=story.format,
        my_role=my_role,  # type: ignore[arg-type]
        created_at=story.created_at,
        updated_at=story.updated_at,
        published_at=story.published_at,
    )


def is_owner(story: Story, user_id: str, *, researcher_id: str = "") -> bool:
    uid = (user_id or "").strip()
    author = (story.author_user_id or "").strip()
    if uid and author == uid:
        return True
    rid = (researcher_id or "").strip().upper()
    author_rid = (story.author_researcher_id or "").strip().upper()
    return bool(rid and author_rid and rid == author_rid)


def _invite_matches_user(
    invite: StoryInvite,
    *,
    user_id: str = "",
    researcher_id: str = "",
    email: str = "",
) -> bool:
    uid = (user_id or "").strip()
    if uid and (invite.recipient_user_id or "").strip() == uid:
        return True
    rid = (researcher_id or "").strip().upper()
    if rid and (invite.recipient_researcher_id or "").strip().upper() == rid:
        return True
    mail = (email or "").strip().lower()
    return bool(mail and (invite.recipient_email or "").strip().lower() == mail)


def collaborator_for(story: Story, user_id: str) -> StoryCollaborator | None:
    uid = (user_id or "").strip()
    if not uid:
        return None
    for collab in story.collaborators or []:
        if collab.user_id == uid:
            return collab
    return None


def access_role(story: Story, user_id: str) -> str | None:
    if is_owner(story, user_id):
        return "owner"
    collab = collaborator_for(story, user_id)
    return collab.role if collab else None


def can_view(story: Story, user_id: str) -> bool:
    return access_role(story, user_id) is not None


def can_edit(story: Story, user_id: str) -> bool:
    if (story.work_status or "open") == "completed":
        return False
    role = access_role(story, user_id)
    return role in ("owner", "editor")


def _ensure_lists(story: Story) -> Story:
    if story.collaborators is None:
        story.collaborators = []
    if story.invites is None:
        story.invites = []
    return story


def _load(story_id: str) -> Story | None:
    from ..supabase import stories as sb_stories

    if sb_stories.enabled():
        data = sb_stories.load_story_payload(story_id)
        if data is None:
            return None
        story = Story.model_validate(data)
        story.sections = _normalize_sections(story.sections)
        _ensure_lists(story)
        if "format" not in data and (story.body_md or "").strip():
            if not any((story.sections or {}).values()):
                story.format = "freeform"
        return story

    path = _path_for(story_id)
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    story = Story.model_validate(data)
    story.sections = _normalize_sections(story.sections)
    _ensure_lists(story)
    # Legacy freeform posts (pre-IEEE) keep their body.
    if "format" not in data and (story.body_md or "").strip():
        if not any((story.sections or {}).values()):
            story.format = "freeform"
    return story


def _save(story: Story) -> Story:
    from ..supabase import stories as sb_stories

    if sb_stories.enabled():
        payload = json.loads(story.model_dump_json())
        sb_stories.save_story_payload(story.story_id, story.author_user_id, payload)
        return story

    path = _path_for(story.story_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(story.model_dump_json(indent=2), encoding="utf-8")
    return story


def _all_stories() -> list[Story]:
    from ..supabase import stories as sb_stories

    if sb_stories.enabled():
        stories: list[Story] = []
        for data in sb_stories.list_story_payloads():
            try:
                story = Story.model_validate(data)
                story.sections = _normalize_sections(story.sections)
                _ensure_lists(story)
                stories.append(story)
            except Exception:
                continue
        return stories

    _ROOT.mkdir(parents=True, exist_ok=True)
    stories = []
    for path in _ROOT.glob("*.json"):
        if path.name.startswith("."):
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            story = Story.model_validate(data)
            story.sections = _normalize_sections(story.sections)
            _ensure_lists(story)
            stories.append(story)
        except Exception:
            continue
    return stories


def _unique_slug(base: str, *, exclude_id: str | None = None) -> str:
    candidate = slugify(base)
    existing = {
        s.slug
        for s in _all_stories()
        if exclude_id is None or s.story_id != exclude_id
    }
    if candidate not in existing:
        return candidate
    n = 2
    while f"{candidate}-{n}" in existing:
        n += 1
    return f"{candidate}-{n}"


def create_story(
    *,
    title: str,
    body_md: str = "",
    author_user_id: str,
    author_name: str = "",
    author_researcher_id: str = "",
    format: str = "ieee",
    authors_line: str = "",
    affiliation: str = "",
    sections: dict[str, str] | None = None,
) -> Story:
    with _store_lock():
        now = _utcnow_iso()
        story_id = uuid.uuid4().hex
        fmt = format if format in ("ieee", "freeform") else "ieee"
        story = Story(
            story_id=story_id,
            slug=_unique_slug(title),
            title=title.strip(),
            body_md=body_md or "",
            excerpt="",
            author_user_id=author_user_id,
            author_name=author_name or "",
            author_researcher_id=author_researcher_id or "",
            status="draft",
            format=fmt,  # type: ignore[arg-type]
            authors_line=(authors_line or "").strip(),
            affiliation=(affiliation or "").strip(),
            sections=_normalize_sections(sections),
            created_at=now,
            updated_at=now,
            published_at=None,
            collaborators=[],
            invites=[],
        )
        if fmt == "ieee" and not (authors_line or "").strip() and author_name:
            story.authors_line = author_name
        _sync_body(story)
        return _save(story)


def get_story(story_id: str) -> Story | None:
    with _store_lock():
        try:
            return _load(story_id)
        except ValueError:
            return None


def list_mine(author_user_id: str) -> list[StoryListItem]:
    with _store_lock():
        mine = [s for s in _all_stories() if s.author_user_id == author_user_id]
        mine.sort(key=lambda s: s.updated_at, reverse=True)
        return [_to_list_item(s, my_role="owner") for s in mine]


def list_shared(user_id: str) -> list[StoryListItem]:
    """Papers where the user is an accepted collaborator (not the owner)."""
    with _store_lock():
        uid = (user_id or "").strip()
        shared: list[StoryListItem] = []
        for story in _all_stories():
            if story.author_user_id == uid:
                continue
            collab = collaborator_for(story, uid)
            if collab is None:
                continue
            shared.append(_to_list_item(story, my_role=collab.role))
        shared.sort(key=lambda s: s.updated_at, reverse=True)
        return shared


def list_public(*, limit: int = 50) -> list[StoryListItem]:
    with _store_lock():
        published = [s for s in _all_stories() if s.status == "published"]
        published.sort(
            key=lambda s: s.published_at or s.updated_at,
            reverse=True,
        )
        return [_to_list_item(s) for s in published[: max(1, min(limit, 100))]]


def get_public_by_slug(slug: str) -> Story | None:
    with _store_lock():
        needle = (slug or "").strip().lower()
        if not needle:
            return None
        for story in _all_stories():
            if story.slug == needle and story.status == "published":
                return story
        return None


def update_story(
    story_id: str,
    *,
    actor_user_id: str,
    title: str | None = None,
    body_md: str | None = None,
    format: str | None = None,
    authors_line: str | None = None,
    affiliation: str | None = None,
    sections: dict[str, str] | None = None,
) -> Story:
    with _store_lock():
        story = _load(story_id)
        if story is None:
            raise LookupError("Story not found.")
        if not can_edit(story, actor_user_id):
            raise PermissionError("You do not have edit access to this paper.")
        if title is not None:
            story.title = title.strip()
            if story.status == "draft" and is_owner(story, actor_user_id):
                story.slug = _unique_slug(story.title, exclude_id=story.story_id)
        if format in ("ieee", "freeform") and is_owner(story, actor_user_id):
            story.format = format  # type: ignore[assignment]
        if authors_line is not None:
            story.authors_line = authors_line.strip()
        if affiliation is not None:
            story.affiliation = affiliation.strip()
        if sections is not None:
            story.sections = _normalize_sections(sections)
        if body_md is not None and story.format == "freeform":
            story.body_md = body_md
        story.updated_at = _utcnow_iso()
        _sync_body(story)
        return _save(story)


def set_work_status(
    story_id: str,
    *,
    actor_user_id: str,
    actor_name: str = "",
    status: str,
) -> Story:
    """The owner or an editor marks the shared paper finished, or the owner reopens it."""
    if status not in ("open", "completed"):
        raise ValueError("Work status must be open or completed.")
    with _store_lock():
        story = _load(story_id)
        if story is None:
            raise LookupError("Story not found.")
        role = access_role(story, actor_user_id)
        if status == "completed":
            if role not in ("owner", "editor"):
                raise PermissionError("Only someone editing this paper can close it.")
            story.work_closed_by_name = (actor_name or "").strip() or "Collaborator"
            story.work_closed_by_user_id = (actor_user_id or "").strip()
            story.work_closed_at = _utcnow_iso()
        else:
            if not is_owner(story, actor_user_id):
                raise PermissionError("Only the owner can reopen this work.")
        story.work_status = status  # type: ignore[assignment]
        story.updated_at = _utcnow_iso()
        return _save(story)


def publish_story(story_id: str, *, author_user_id: str) -> Story:
    with _store_lock():
        story = _load(story_id)
        if story is None:
            raise LookupError("Story not found.")
        if story.author_user_id != author_user_id:
            raise PermissionError("Only the author can publish this story.")
        if not (story.title or "").strip():
            raise ValueError("Add a title before publishing.")
        _sync_body(story)
        if not _has_writable_content(story):
            raise ValueError("Add some writing before publishing.")
        now = _utcnow_iso()
        story.status = "published"
        story.published_at = story.published_at or now
        story.updated_at = now
        return _save(story)


def unpublish_story(story_id: str, *, author_user_id: str) -> Story:
    with _store_lock():
        story = _load(story_id)
        if story is None:
            raise LookupError("Story not found.")
        if story.author_user_id != author_user_id:
            raise PermissionError("Only the author can unpublish this story.")
        story.status = "draft"
        story.updated_at = _utcnow_iso()
        return _save(story)


def delete_story(story_id: str, *, author_user_id: str) -> None:
    from ..supabase import stories as sb_stories

    with _store_lock():
        story = _load(story_id)
        if story is None:
            raise LookupError("Story not found.")
        if story.author_user_id != author_user_id:
            raise PermissionError("Only the author can delete this story.")
        if sb_stories.enabled():
            sb_stories.delete_story(story_id)
            return
        path = _path_for(story_id)
        if path.exists():
            path.unlink()


def create_invite(
    story_id: str,
    *,
    owner_user_id: str,
    owner_name: str = "",
    owner_researcher_id: str = "",
    recipient_user_id: str,
    recipient_researcher_id: str = "",
    recipient_name: str = "",
    recipient_email: str = "",
    role: str = "editor",
) -> StoryInvite:
    with _store_lock():
        story = _load(story_id)
        if story is None:
            raise LookupError("Story not found.")
        if not is_owner(story, owner_user_id, researcher_id=owner_researcher_id):
            raise PermissionError("Only the owner can invite collaborators.")
        if role not in ("editor", "viewer"):
            raise ValueError("Role must be editor or viewer.")
        rid = (recipient_user_id or "").strip()
        if not rid:
            raise ValueError("Recipient user is required.")
        if rid == owner_user_id:
            raise ValueError("You cannot invite yourself.")
        if collaborator_for(story, rid) is not None:
            raise ValueError("That researcher is already a collaborator.")
        for inv in story.invites or []:
            if inv.status == "pending" and inv.recipient_user_id == rid:
                raise ValueError("A pending invite already exists for that researcher.")
        invite = StoryInvite(
            invite_id=uuid.uuid4().hex,
            recipient_user_id=rid,
            recipient_researcher_id=(recipient_researcher_id or "").strip().upper(),
            recipient_name=recipient_name or "",
            recipient_email=(recipient_email or "").strip().lower(),
            role=role,  # type: ignore[arg-type]
            status="pending",
            invited_by=owner_user_id,
            invited_by_name=owner_name or "",
            created_at=_utcnow_iso(),
        )
        story.invites = [invite, *(story.invites or [])]
        story.updated_at = _utcnow_iso()
        _save(story)
        return invite


def revoke_invite(story_id: str, invite_id: str, *, owner_user_id: str) -> Story:
    with _store_lock():
        story = _load(story_id)
        if story is None:
            raise LookupError("Story not found.")
        if not is_owner(story, owner_user_id):
            raise PermissionError("Only the owner can revoke invites.")
        found = False
        updated: list[StoryInvite] = []
        for inv in story.invites or []:
            if inv.invite_id == invite_id:
                found = True
                if inv.status == "pending":
                    inv = inv.model_copy(update={"status": "revoked"})
                updated.append(inv)
            else:
                updated.append(inv)
        if not found:
            raise LookupError("Invite not found.")
        story.invites = updated
        story.updated_at = _utcnow_iso()
        return _save(story)


def delete_invite_history(story_id: str, invite_id: str, *, actor_user_id: str) -> Story:
    """Remove one invite line from the shared history. A pending invite is cancelled too."""
    with _store_lock():
        story = _load(story_id)
        if story is None:
            raise LookupError("Story not found.")
        if access_role(story, actor_user_id) not in ("owner", "editor", "viewer"):
            raise PermissionError("Only people on this paper can delete history.")
        needle = (invite_id or "").strip()
        remaining = [inv for inv in (story.invites or []) if inv.invite_id != needle]
        if len(remaining) == len(story.invites or []):
            raise LookupError("History entry not found.")
        story.invites = remaining
        story.updated_at = _utcnow_iso()
        return _save(story)


def clear_close_history(story_id: str, *, actor_user_id: str) -> Story:
    """Drop the 'finished and closed' line. The paper stays closed."""
    with _store_lock():
        story = _load(story_id)
        if story is None:
            raise LookupError("Story not found.")
        if access_role(story, actor_user_id) not in ("owner", "editor", "viewer"):
            raise PermissionError("Only people on this paper can delete history.")
        if not (
            (story.work_closed_by_name or "").strip()
            or (story.work_closed_by_user_id or "").strip()
            or story.work_closed_at
        ):
            raise LookupError("No close record to delete.")
        story.work_closed_by_name = ""
        story.work_closed_by_user_id = ""
        story.work_closed_at = None
        story.updated_at = _utcnow_iso()
        return _save(story)


def list_pending_invites_for_user(
    user_id: str,
    *,
    researcher_id: str = "",
    email: str = "",
) -> list[PendingStoryInvite]:
    with _store_lock():
        uid = (user_id or "").strip()
        if not uid and not (researcher_id or "").strip() and not (email or "").strip():
            return []
        pending: list[PendingStoryInvite] = []
        for story in _all_stories():
            for inv in story.invites or []:
                if inv.status == "pending" and _invite_matches_user(
                    inv, user_id=uid, researcher_id=researcher_id, email=email
                ):
                    pending.append(
                        PendingStoryInvite(
                            invite_id=inv.invite_id,
                            story_id=story.story_id,
                            story_title=story.title,
                            role=inv.role,
                            invited_by=inv.invited_by,
                            invited_by_name=inv.invited_by_name,
                            created_at=inv.created_at,
                        )
                    )
        pending.sort(key=lambda i: i.created_at, reverse=True)
        return pending


def accept_invite(invite_id: str, *, user_id: str, name: str = "", email: str = "", researcher_id: str = "") -> Story:
    with _store_lock():
        uid = (user_id or "").strip()
        needle = (invite_id or "").strip()
        if not needle:
            raise LookupError("Invite not found.")
        for story in _all_stories():
            for inv in story.invites or []:
                if inv.invite_id != needle:
                    continue
                if inv.status != "pending":
                    raise ValueError("This invite is no longer pending.")
                if not _invite_matches_user(
                    inv, user_id=uid, researcher_id=researcher_id, email=email
                ):
                    raise PermissionError("This invite is for a different account.")
                if collaborator_for(story, uid) is None:
                    story.collaborators = [
                        StoryCollaborator(
                            user_id=uid,
                            name=name or inv.recipient_name or "",
                            email=email or inv.recipient_email or "",
                            researcher_id=(researcher_id or inv.recipient_researcher_id or "").strip().upper(),
                            role=inv.role,
                            added_at=_utcnow_iso(),
                        ),
                        *(story.collaborators or []),
                    ]
                story.invites = [
                    (
                        i.model_copy(update={"status": "accepted"})
                        if i.invite_id == needle
                        else i
                    )
                    for i in (story.invites or [])
                ]
                story.updated_at = _utcnow_iso()
                return _save(story)
        raise LookupError("Invite not found.")


def decline_invite(
    invite_id: str,
    *,
    user_id: str,
    researcher_id: str = "",
    email: str = "",
) -> None:
    with _store_lock():
        uid = (user_id or "").strip()
        needle = (invite_id or "").strip()
        if not needle:
            raise LookupError("Invite not found.")
        for story in _all_stories():
            for inv in story.invites or []:
                if inv.invite_id != needle:
                    continue
                if inv.status != "pending":
                    raise ValueError("This invite is no longer pending.")
                if not _invite_matches_user(
                    inv, user_id=uid, researcher_id=researcher_id, email=email
                ):
                    raise PermissionError("This invite is for a different account.")
                story.invites = [
                    (
                        i.model_copy(update={"status": "declined"})
                        if i.invite_id == needle
                        else i
                    )
                    for i in (story.invites or [])
                ]
                story.updated_at = _utcnow_iso()
                _save(story)
                return
        raise LookupError("Invite not found.")


def remove_collaborator(story_id: str, collaborator_user_id: str, *, actor_user_id: str) -> Story:
    with _store_lock():
        story = _load(story_id)
        if story is None:
            raise LookupError("Story not found.")
        target = (collaborator_user_id or "").strip()
        actor = (actor_user_id or "").strip()
        if not target:
            raise ValueError("Collaborator id is required.")
        # Owner can remove anyone; collaborator can leave themselves.
        if not is_owner(story, actor) and target != actor:
            raise PermissionError("Only the owner can remove other collaborators.")
        if is_owner(story, target):
            raise ValueError("Cannot remove the paper owner.")
        before = len(story.collaborators or [])
        story.collaborators = [c for c in (story.collaborators or []) if c.user_id != target]
        if len(story.collaborators) == before:
            raise LookupError("Collaborator not found.")
        story.updated_at = _utcnow_iso()
        return _save(story)


def update_collaborator_role(
    story_id: str,
    collaborator_user_id: str,
    *,
    owner_user_id: str,
    role: str,
) -> Story:
    with _store_lock():
        story = _load(story_id)
        if story is None:
            raise LookupError("Story not found.")
        if not is_owner(story, owner_user_id):
            raise PermissionError("Only the owner can change collaborator roles.")
        if role not in ("editor", "viewer"):
            raise ValueError("Role must be editor or viewer.")
        target = (collaborator_user_id or "").strip()
        found = False
        updated: list[StoryCollaborator] = []
        for collab in story.collaborators or []:
            if collab.user_id == target:
                found = True
                updated.append(collab.model_copy(update={"role": role}))
            else:
                updated.append(collab)
        if not found:
            raise LookupError("Collaborator not found.")
        story.collaborators = updated
        story.updated_at = _utcnow_iso()
        return _save(story)