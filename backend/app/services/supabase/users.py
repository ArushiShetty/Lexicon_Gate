"""Supabase-backed user directory (app_users)."""

from __future__ import annotations

import secrets
import uuid
from typing import Any

from .client import rest_insert, rest_select, rest_upsert, supabase_enabled

_TABLE = "app_users"
_RID_ALPHABET = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"


def _normalise_email(email: str) -> str:
    return (email or "").strip().lower()


def _row_to_user(row: dict[str, Any]) -> dict[str, Any]:
    google_sub = row.get("google_sub")
    return {
        "user_id": str(row.get("user_id") or ""),
        "email": str(row.get("email") or ""),
        "name": str(row.get("name") or ""),
        "password_hash": str(row.get("password_hash") or ""),
        "google_sub": str(google_sub) if google_sub else None,
        "auth_provider": str(row.get("auth_provider") or ""),
        "researcher_id": str(row.get("researcher_id") or ""),
    }


def _generate_researcher_id(existing: set[str]) -> str:
    for _ in range(64):
        body = "".join(secrets.choice(_RID_ALPHABET) for _ in range(8))
        rid = f"LG-{body[:4]}-{body[4:]}"
        if rid not in existing:
            return rid
    raise RuntimeError("Could not allocate a unique Researcher ID.")


def list_researcher_ids() -> set[str]:
    rows = rest_select(_TABLE, params={"select": "researcher_id"})
    out: set[str] = set()
    for row in rows:
        rid = str(row.get("researcher_id") or "").strip().upper()
        if rid:
            out.add(rid)
    return out


def find_by_email(email: str) -> dict[str, Any] | None:
    wanted = _normalise_email(email)
    rows = rest_select(
        _TABLE,
        params={"email": f"eq.{wanted}", "select": "*", "limit": "1"},
    )
    return _row_to_user(rows[0]) if rows else None


def find_by_user_id(user_id: str) -> dict[str, Any] | None:
    uid = str(user_id or "").strip()
    if not uid:
        return None
    rows = rest_select(
        _TABLE,
        params={"user_id": f"eq.{uid}", "select": "*", "limit": "1"},
    )
    return _row_to_user(rows[0]) if rows else None


def find_users_by_name(name: str) -> list[dict[str, Any]]:
    needle = (name or "").strip().lower()
    if len(needle) < 2 or any(ch in needle for ch in ",.()"):
        return []
    rows = rest_select(
        _TABLE,
        params={"name": f"ilike.{needle}", "select": "*", "limit": "8"},
    )
    return [_row_to_user(row) for row in rows]


def find_by_researcher_id(researcher_id: str) -> dict[str, Any] | None:
    rid = str(researcher_id or "").strip().upper()
    if not rid:
        return None
    rows = rest_select(
        _TABLE,
        params={"researcher_id": f"eq.{rid}", "select": "*", "limit": "1"},
    )
    return _row_to_user(rows[0]) if rows else None


def find_by_google_sub(google_sub: str) -> dict[str, Any] | None:
    sub = str(google_sub or "").strip()
    if not sub:
        return None
    rows = rest_select(
        _TABLE,
        params={"google_sub": f"eq.{sub}", "select": "*", "limit": "1"},
    )
    return _row_to_user(rows[0]) if rows else None


def create_user(email: str, password_hash: str, name: str = "") -> dict[str, Any]:
    email_n = _normalise_email(email)
    if find_by_email(email_n):
        raise ValueError("email_taken")
    record = {
        "user_id": uuid.uuid4().hex,
        "email": email_n,
        "name": (name or "").strip()[:80],
        "password_hash": password_hash,
        "researcher_id": _generate_researcher_id(list_researcher_ids()),
        "auth_provider": "",
        "google_sub": None,
    }
    rows = rest_insert(_TABLE, record)
    return _row_to_user(rows[0]) if rows else record


def save_user(user: dict[str, Any]) -> dict[str, Any]:
    payload = {
        "user_id": str(user.get("user_id") or ""),
        "email": str(user.get("email") or ""),
        "name": str(user.get("name") or ""),
        "password_hash": str(user.get("password_hash") or ""),
        "google_sub": user.get("google_sub") or None,
        "auth_provider": str(user.get("auth_provider") or ""),
        "researcher_id": str(user.get("researcher_id") or "") or None,
    }
    rows = rest_upsert(_TABLE, payload, on_conflict="user_id")
    return _row_to_user(rows[0]) if rows else payload


def ensure_researcher_id(user: dict[str, Any], normalise_fn) -> dict[str, Any]:
    current = str(user.get("researcher_id") or "").strip()
    if current:
        try:
            user["researcher_id"] = normalise_fn(current)
            return save_user(user) if user.get("user_id") else user
        except ValueError:
            pass
    uid = str(user.get("user_id") or "")
    if uid:
        existing = find_by_user_id(uid)
        if existing and str(existing.get("researcher_id") or "").strip():
            try:
                existing["researcher_id"] = normalise_fn(str(existing["researcher_id"]))
                return save_user(existing)
            except ValueError:
                pass
            existing["researcher_id"] = _generate_researcher_id(list_researcher_ids())
            return save_user(existing)
    user["researcher_id"] = _generate_researcher_id(list_researcher_ids())
    if uid:
        return save_user(user)
    return user


def upsert_google_user(*, google_sub: str, email: str, name: str = "") -> dict[str, Any]:
    sub = str(google_sub or "").strip()
    email_n = _normalise_email(email)
    display = (name or "").strip()[:80]
    if not sub:
        raise ValueError("missing_google_sub")

    by_sub = find_by_google_sub(sub)
    if by_sub is not None:
        if display and not str(by_sub.get("name") or "").strip():
            by_sub["name"] = display
            return save_user(by_sub)
        return by_sub

    by_email = find_by_email(email_n)
    if by_email is not None:
        by_email["google_sub"] = sub
        if display and not str(by_email.get("name") or "").strip():
            by_email["name"] = display
        if not str(by_email.get("auth_provider") or "").strip():
            by_email["auth_provider"] = (
                "google" if not str(by_email.get("password_hash") or "") else "linked"
            )
        return save_user(by_email)

    record = {
        "user_id": uuid.uuid4().hex,
        "email": email_n,
        "name": display,
        "password_hash": "",
        "google_sub": sub,
        "auth_provider": "google",
        "researcher_id": _generate_researcher_id(list_researcher_ids()),
    }
    rows = rest_insert(_TABLE, record)
    return _row_to_user(rows[0]) if rows else record


def enabled() -> bool:
    return supabase_enabled()
