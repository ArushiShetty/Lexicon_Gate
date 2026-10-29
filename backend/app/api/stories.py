"""Write / Stories API — draft, publish, IEEE paper edit, collaborators, public read."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query, Request, Response
from fastapi.responses import Response as RawResponse

from .auth import current_user_from_request
from ..config import get_settings
from ..models.stories import (
    PendingStoryInviteList,
    PublicStory,
    Story,
    StoryCollaboratorUpdate,
    StoryCreate,
    StoryDetail,
    StoryInvite,
    StoryInviteCreate,
    StoryListResponse,
    StoryUpdate,
    WorkStatusUpdate,
)
from ..services.auth import users as user_store
from ..services.stories import store as story_store
from ..services.stories.ieee_docx import build_ieee_docx, safe_docx_filename

router = APIRouter(prefix="/stories", tags=["stories"])


def _require_user(request: Request) -> dict:
    user = current_user_from_request(request)
    if user is None:
        raise HTTPException(status_code=401, detail="Not signed in.")
    return user


def _user_id(user: dict) -> str:
    return str(user.get("user_id") or user.get("id") or "")


def _identity(user: dict) -> dict:
    settings = get_settings()
    full = user_store.find_by_user_id(settings, _user_id(user)) or user
    return full


def _with_role(story: Story, user_id: str) -> StoryDetail:
    role = story_store.access_role(story, user_id)
    return StoryDetail(**story.model_dump(), my_role=role)  # type: ignore[arg-type]


def _to_public(story: Story) -> PublicStory:
    return PublicStory(
        story_id=story.story_id,
        slug=story.slug,
        title=story.title,
        body_md=story.body_md,
        excerpt=story.excerpt,
        author_name=story.author_name,
        author_researcher_id=story.author_researcher_id,
        format=story.format,
        authors_line=story.authors_line,
        affiliation=story.affiliation,
        sections=story.sections or {},
        published_at=story.published_at,
        updated_at=story.updated_at,
    )


def _docx_response(story: Story) -> RawResponse:
    try:
        data = build_ieee_docx(story)
    except Exception as exc:  # pragma: no cover - defensive
        raise HTTPException(status_code=500, detail=f"Could not build Word document: {exc}") from exc
    filename = safe_docx_filename(story.title)
    return RawResponse(
        content=data,
        media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"',
            "Cache-Control": "no-store",
        },
    )


@router.get("/public", response_model=StoryListResponse)
def public_feed(limit: int = Query(default=40, ge=1, le=100)) -> StoryListResponse:
    stories = story_store.list_public(limit=limit)
    return StoryListResponse(stories=stories, total=len(stories))


@router.get("/public/{slug}/docx")
def public_story_docx(slug: str) -> RawResponse:
    story = story_store.get_public_by_slug(slug)
    if story is None:
        raise HTTPException(status_code=404, detail="Story not found.")
    return _docx_response(story)


@router.get("/public/{slug}", response_model=PublicStory)
def public_story(slug: str) -> PublicStory:
    story = story_store.get_public_by_slug(slug)
    if story is None:
        raise HTTPException(status_code=404, detail="Story not found.")
    return _to_public(story)


@router.get("/mine", response_model=StoryListResponse)
def list_mine(request: Request) -> StoryListResponse:
    user = _require_user(request)
    stories = story_store.list_mine(_user_id(user))
    return StoryListResponse(stories=stories, total=len(stories))


@router.get("/shared", response_model=StoryListResponse)
def list_shared(request: Request) -> StoryListResponse:
    user = _require_user(request)
    stories = story_store.list_shared(_user_id(user))
    return StoryListResponse(stories=stories, total=len(stories))


@router.get("/invites/pending", response_model=PendingStoryInviteList)
def pending_story_invites(request: Request) -> PendingStoryInviteList:
    user = _identity(_require_user(request))
    invites = story_store.list_pending_invites_for_user(
        _user_id(user),
        researcher_id=str(user.get("researcher_id") or ""),
        email=str(user.get("email") or ""),
    )
    return PendingStoryInviteList(invites=invites)


@router.post("/invites/{invite_id}/accept", response_model=StoryDetail)
def accept_story_invite(invite_id: str, request: Request) -> StoryDetail:
    user = _identity(_require_user(request))
    try:
        story = story_store.accept_invite(
            invite_id,
            user_id=_user_id(user),
            name=str(user.get("name") or ""),
            email=str(user.get("email") or ""),
            researcher_id=str(user.get("researcher_id") or ""),
        )
        return _with_role(story, _user_id(user))
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/invites/{invite_id}/decline", status_code=204)
def decline_story_invite(invite_id: str, request: Request) -> Response:
    user = _identity(_require_user(request))
    try:
        story_store.decline_invite(
            invite_id,
            user_id=_user_id(user),
            researcher_id=str(user.get("researcher_id") or ""),
            email=str(user.get("email") or ""),
        )
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return Response(status_code=204)


@router.post("", response_model=StoryDetail)
def create_story(payload: StoryCreate, request: Request) -> StoryDetail:
    user = _require_user(request)
    story = story_store.create_story(
        title=payload.title,
        body_md=payload.body_md or "",
        author_user_id=_user_id(user),
        author_name=str(user.get("name") or ""),
        author_researcher_id=str(user.get("researcher_id") or ""),
        format=payload.format,
        authors_line=payload.authors_line,
        affiliation=payload.affiliation,
        sections=payload.sections,
    )
    return _with_role(story, _user_id(user))


@router.get("/{story_id}/docx")
def download_story_docx(story_id: str, request: Request) -> RawResponse:
    user = _require_user(request)
    story = story_store.get_story(story_id)
    if story is None:
        raise HTTPException(status_code=404, detail="Story not found.")
    if not story_store.can_view(story, _user_id(user)):
        raise HTTPException(status_code=403, detail="You do not have access to this paper.")
    return _docx_response(story)


@router.get("/{story_id}", response_model=StoryDetail)
def get_story(story_id: str, request: Request) -> StoryDetail:
    user = _require_user(request)
    story = story_store.get_story(story_id)
    if story is None:
        raise HTTPException(status_code=404, detail="Story not found.")
    if not story_store.can_view(story, _user_id(user)):
        raise HTTPException(status_code=403, detail="You do not have access to this paper.")
    return _with_role(story, _user_id(user))


@router.patch("/{story_id}", response_model=StoryDetail)
def update_story(story_id: str, payload: StoryUpdate, request: Request) -> StoryDetail:
    user = _require_user(request)
    try:
        story = story_store.update_story(
            story_id,
            actor_user_id=_user_id(user),
            title=payload.title,
            body_md=payload.body_md,
            format=payload.format,
            authors_line=payload.authors_line,
            affiliation=payload.affiliation,
            sections=payload.sections,
        )
        return _with_role(story, _user_id(user))
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/{story_id}/publish", response_model=StoryDetail)
def publish_story(story_id: str, request: Request) -> StoryDetail:
    user = _require_user(request)
    try:
        story = story_store.publish_story(story_id, author_user_id=_user_id(user))
        return _with_role(story, _user_id(user))
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/{story_id}/work-status", response_model=StoryDetail)
def update_work_status(story_id: str, payload: WorkStatusUpdate, request: Request) -> StoryDetail:
    user = _identity(_require_user(request))
    try:
        story = story_store.set_work_status(
            story_id,
            actor_user_id=_user_id(user),
            actor_name=str(user.get("name") or ""),
            status=payload.status,
        )
        return _with_role(story, _user_id(user))
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/{story_id}/unpublish", response_model=StoryDetail)
def unpublish_story(story_id: str, request: Request) -> StoryDetail:
    user = _require_user(request)
    try:
        story = story_store.unpublish_story(story_id, author_user_id=_user_id(user))
        return _with_role(story, _user_id(user))
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc


@router.delete("/{story_id}", status_code=204)
def delete_story(story_id: str, request: Request) -> Response:
    user = _require_user(request)
    try:
        story_store.delete_story(story_id, author_user_id=_user_id(user))
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    return Response(status_code=204)


@router.post("/{story_id}/invites", response_model=StoryInvite)
def invite_collaborator(
    story_id: str, payload: StoryInviteCreate, request: Request
) -> StoryInvite:
    user = _identity(_require_user(request))
    settings = get_settings()
    raw = (payload.researcher_id or "").strip()
    recipient = user_store.find_by_researcher_id(settings, raw)
    if recipient is None:
        matches = user_store.find_users_by_name(settings, raw)
        if len(matches) == 1:
            recipient = matches[0]
        elif len(matches) > 1:
            raise HTTPException(
                status_code=409,
                detail="More than one researcher has that name. Use their Researcher ID.",
            )
    if recipient is None:
        raise HTTPException(status_code=404, detail="No researcher found with that ID or name.")
    try:
        return story_store.create_invite(
            story_id,
            owner_user_id=_user_id(user),
            owner_name=str(user.get("name") or ""),
            owner_researcher_id=str(user.get("researcher_id") or ""),
            recipient_user_id=str(recipient.get("user_id") or ""),
            recipient_researcher_id=str(recipient.get("researcher_id") or ""),
            recipient_name=str(recipient.get("name") or ""),
            recipient_email=str(recipient.get("email") or ""),
            role=payload.role,
        )
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.delete("/{story_id}/history/invites/{invite_id}", response_model=StoryDetail)
def delete_invite_history(story_id: str, invite_id: str, request: Request) -> StoryDetail:
    user = _require_user(request)
    try:
        story = story_store.delete_invite_history(
            story_id, invite_id, actor_user_id=_user_id(user)
        )
        return _with_role(story, _user_id(user))
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc


@router.delete("/{story_id}/history/close", response_model=StoryDetail)
def delete_close_history(story_id: str, request: Request) -> StoryDetail:
    user = _require_user(request)
    try:
        story = story_store.clear_close_history(story_id, actor_user_id=_user_id(user))
        return _with_role(story, _user_id(user))
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc


@router.delete("/{story_id}/invites/{invite_id}", response_model=StoryDetail)
def revoke_collaborator_invite(story_id: str, invite_id: str, request: Request) -> StoryDetail:
    user = _require_user(request)
    try:
        story = story_store.revoke_invite(
            story_id, invite_id, owner_user_id=_user_id(user)
        )
        return _with_role(story, _user_id(user))
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc


@router.patch("/{story_id}/collaborators/{collaborator_user_id}", response_model=StoryDetail)
def patch_collaborator_role(
    story_id: str,
    collaborator_user_id: str,
    payload: StoryCollaboratorUpdate,
    request: Request,
) -> StoryDetail:
    user = _require_user(request)
    try:
        story = story_store.update_collaborator_role(
            story_id,
            collaborator_user_id,
            owner_user_id=_user_id(user),
            role=payload.role,
        )
        return _with_role(story, _user_id(user))
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.delete("/{story_id}/collaborators/{collaborator_user_id}", response_model=StoryDetail)
def remove_story_collaborator(
    story_id: str, collaborator_user_id: str, request: Request
) -> StoryDetail:
    user = _require_user(request)
    try:
        story = story_store.remove_collaborator(
            story_id,
            collaborator_user_id,
            actor_user_id=_user_id(user),
        )
        return _with_role(story, _user_id(user))
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
