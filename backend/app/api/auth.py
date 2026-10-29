"""Signup, login, logout, current user, and Researcher ID lookup."""

from __future__ import annotations

import time
from collections import defaultdict, deque

from fastapi import APIRouter, HTTPException, Request, Response

from ..config import get_settings
from ..models.auth import GoogleAuthRequest, LoginRequest, ResearcherPublic, SignupRequest, UserPublic
from ..services.auth.google import verify_google_id_token
from ..services.auth.sessions import create_session, get_session, revoke_session
from ..services.auth.users import (
    authenticate,
    create_user,
    ensure_researcher_id,
    find_by_researcher_id,
    find_by_user_id,
    normalise_researcher_id,
    public_user,
    researcher_public,
    upsert_google_user,
    valid_email,
)

router = APIRouter(prefix="/auth", tags=["auth"])

COOKIE_NAME = "selfrag_session"

_LOOKUP_WINDOW_SEC = 60
_LOOKUP_MAX = 30
_lookup_attempts: dict[str, deque[float]] = defaultdict(deque)


def _set_session_cookie(response: Response, token: str, ttl: int) -> None:
    response.set_cookie(
        key=COOKIE_NAME,
        value=token,
        httponly=True,
        samesite="lax",
        max_age=ttl,
        path="/",
    )


def _clear_session_cookie(response: Response) -> None:
    response.delete_cookie(key=COOKIE_NAME, path="/")


def current_user_from_request(request: Request) -> dict | None:
    return get_session(request.cookies.get(COOKIE_NAME))


def _check_lookup_rate(key: str) -> None:
    now = time.monotonic()
    bucket = _lookup_attempts[key]
    while bucket and now - bucket[0] > _LOOKUP_WINDOW_SEC:
        bucket.popleft()
    if len(bucket) >= _LOOKUP_MAX:
        raise HTTPException(
            status_code=429,
            detail="Too many Researcher ID lookups. Wait a minute and try again.",
        )
    bucket.append(now)


def _user_public_with_rid(user: dict) -> UserPublic:
    settings = get_settings()
    ensured = ensure_researcher_id(settings, user)
    return UserPublic(**public_user(ensured))


@router.post("/signup", response_model=UserPublic)
def signup(payload: SignupRequest, response: Response) -> UserPublic:
    settings = get_settings()
    if not valid_email(payload.email):
        raise HTTPException(status_code=400, detail="Enter a valid email address.")
    try:
        user = create_user(settings, payload.email, payload.password, payload.name)
    except ValueError:
        raise HTTPException(status_code=409, detail="An account with that email already exists.")
    token = create_session(user, settings.session_ttl_seconds)
    _set_session_cookie(response, token, settings.session_ttl_seconds)
    return UserPublic(**public_user(user))


@router.post("/login", response_model=UserPublic)
def login(payload: LoginRequest, response: Response) -> UserPublic:
    settings = get_settings()
    user = authenticate(settings, payload.email, payload.password)
    if user is None:
        raise HTTPException(status_code=401, detail="Email or password is incorrect.")
    token = create_session(user, settings.session_ttl_seconds)
    _set_session_cookie(response, token, settings.session_ttl_seconds)
    return UserPublic(**public_user(user))


@router.post("/google", response_model=UserPublic)
def google_auth(payload: GoogleAuthRequest, response: Response) -> UserPublic:
    settings = get_settings()
    client_id = (settings.google_client_id or "").strip()
    if not client_id:
        raise HTTPException(
            status_code=503,
            detail="Google sign-in is not configured. Set SELFRAG_GOOGLE_CLIENT_ID.",
        )
    try:
        profile = verify_google_id_token(payload.credential, client_id)
        user = upsert_google_user(
            settings,
            google_sub=profile["google_sub"],
            email=profile["email"],
            name=profile.get("name") or "",
        )
    except ValueError as exc:
        reason = str(exc)
        if reason == "google_not_configured":
            raise HTTPException(status_code=503, detail="Google sign-in is not configured.") from exc
        if reason == "google_unreachable":
            raise HTTPException(status_code=502, detail="Could not reach Google to verify sign-in.") from exc
        raise HTTPException(status_code=401, detail="Google sign-in failed. Try again.") from exc
    user = ensure_researcher_id(settings, user)
    token = create_session(user, settings.session_ttl_seconds)
    _set_session_cookie(response, token, settings.session_ttl_seconds)
    return UserPublic(**public_user(user))


@router.post("/logout")
def logout(request: Request, response: Response) -> dict[str, str]:
    revoke_session(request.cookies.get(COOKIE_NAME))
    _clear_session_cookie(response)
    return {"status": "ok"}


@router.get("/me", response_model=UserPublic)
def me(request: Request) -> UserPublic:
    session = current_user_from_request(request)
    if session is None:
        raise HTTPException(status_code=401, detail="Not signed in.")
    settings = get_settings()
    user = find_by_user_id(settings, str(session.get("user_id") or ""))
    if user is None:
        # Fall back to session fields; still try lazy ID if possible.
        return UserPublic(
            user_id=str(session.get("user_id", "")),
            email=str(session.get("email", "")),
            name=str(session.get("name", "")),
            researcher_id=str(session.get("researcher_id", "")),
        )
    return _user_public_with_rid(user)


@router.get("/researchers/{researcher_id}", response_model=ResearcherPublic)
def lookup_researcher(researcher_id: str, request: Request) -> ResearcherPublic:
    """Resolve a Researcher ID to a safe public card. Does not grant project access."""
    session = current_user_from_request(request)
    if session is None:
        raise HTTPException(status_code=401, detail="Not signed in.")
    _check_lookup_rate(str(session.get("user_id") or "anon"))
    try:
        rid = normalise_researcher_id(researcher_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    settings = get_settings()
    user = find_by_researcher_id(settings, rid)
    if user is None:
        # Ensure lazy IDs exist so lookups work after first /me for older accounts —
        # still return 404 if this specific ID is unknown.
        raise HTTPException(status_code=404, detail="No researcher found with that ID.")
    user = ensure_researcher_id(settings, user)
    return ResearcherPublic(**researcher_public(user))
