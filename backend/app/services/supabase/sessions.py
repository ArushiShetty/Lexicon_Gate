"""Supabase-backed session store (app_sessions)."""

from __future__ import annotations

import secrets
from datetime import datetime, timezone
from typing import Any

from .client import rest_delete, rest_insert, rest_select, supabase_enabled

_TABLE = "app_sessions"


def _iso(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()


def _from_iso(value: str) -> float:
    raw = str(value or "").strip()
    if not raw:
        return 0.0
    if raw.endswith("Z"):
        raw = raw[:-1] + "+00:00"
    try:
        return datetime.fromisoformat(raw).timestamp()
    except ValueError:
        return 0.0


def create_session(user: dict[str, Any], ttl_seconds: int) -> str:
    import time

    token = secrets.token_urlsafe(32)
    expires = time.time() + max(60, int(ttl_seconds))
    rest_insert(
        _TABLE,
        {
            "token": token,
            "user_id": str(user.get("user_id") or ""),
            "email": str(user.get("email") or ""),
            "name": str(user.get("name") or ""),
            "researcher_id": str(user.get("researcher_id") or ""),
            "expires_at": _iso(expires),
        },
    )
    return token


def get_session(token: str | None) -> dict[str, Any] | None:
    if not token:
        return None
    import time

    rows = rest_select(
        _TABLE,
        params={"token": f"eq.{token}", "select": "*", "limit": "1"},
    )
    if not rows:
        return None
    row = rows[0]
    expires = _from_iso(str(row.get("expires_at") or ""))
    if expires < time.time():
        revoke_session(token)
        return None
    return {
        "user_id": str(row.get("user_id") or ""),
        "email": str(row.get("email") or ""),
        "name": str(row.get("name") or ""),
        "researcher_id": str(row.get("researcher_id") or ""),
        "expires_at": expires,
    }


def revoke_session(token: str | None) -> None:
    if not token:
        return
    rest_delete(_TABLE, {"token": f"eq.{token}"})


def clear_sessions() -> None:
    # Delete all rows (service role). PostgREST needs a filter; use expires_at not null.
    rest_delete(_TABLE, {"expires_at": "not.is.null"})


def enabled() -> bool:
    return supabase_enabled()
