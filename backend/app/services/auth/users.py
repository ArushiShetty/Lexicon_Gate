"""JSON user directory under the data dir."""

from __future__ import annotations

import json
import re
import secrets
import threading
import uuid
from pathlib import Path
from typing import Any

from ...config import Settings
from .passwords import hash_password, verify_password

_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
# Meet-like public code: LG-XXXX-XXXX (no ambiguous I/O/0/1).
_RID_ALPHABET = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"
_RID_RE = re.compile(r"^LG-[A-Z2-9]{4}-[A-Z2-9]{4}$")
_lock = threading.Lock()


def normalise_email(email: str) -> str:
    return (email or "").strip().lower()


def valid_email(email: str) -> bool:
    return bool(_EMAIL_RE.match(normalise_email(email)))


def normalise_researcher_id(value: str) -> str:
    """Canonical LG-XXXX-XXXX form; raises ValueError if invalid."""
    raw = (value or "").strip().upper().replace(" ", "")
    if not raw:
        raise ValueError("Researcher ID is required.")
    if _RID_RE.match(raw):
        return raw
    compact = raw.replace("-", "")
    if compact.startswith("LG") and len(compact) == 10:
        candidate = f"LG-{compact[2:6]}-{compact[6:10]}"
        if _RID_RE.match(candidate):
            return candidate
    raise ValueError("Invalid Researcher ID format. Expected LG-XXXX-XXXX.")


def users_path(settings: Settings) -> Path:
    path = Path(settings.data_dir) / "auth" / "users.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _load(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    if isinstance(payload, list):
        return payload
    return payload.get("users", []) if isinstance(payload, dict) else []


def _save(path: Path, users: list[dict[str, Any]]) -> None:
    path.write_text(json.dumps({"users": users}, indent=2), encoding="utf-8")


def _generate_researcher_id(existing: set[str]) -> str:
    for _ in range(64):
        body = "".join(secrets.choice(_RID_ALPHABET) for _ in range(8))
        rid = f"LG-{body[:4]}-{body[4:]}"
        if rid not in existing:
            return rid
    raise RuntimeError("Could not allocate a unique Researcher ID.")


def _assigned_ids(users: list[dict[str, Any]]) -> set[str]:
    out: set[str] = set()
    for user in users:
        rid = str(user.get("researcher_id") or "").strip().upper()
        if rid:
            out.add(rid)
    return out


def ensure_researcher_id(settings: Settings, user: dict[str, Any]) -> dict[str, Any]:
    """Lazy-assign a stable Researcher ID if missing. Persists to disk or Supabase."""
    from ..supabase import users as sb_users

    if sb_users.enabled():
        return sb_users.ensure_researcher_id(user, normalise_researcher_id)

    current = str(user.get("researcher_id") or "").strip()
    if current:
        try:
            user["researcher_id"] = normalise_researcher_id(current)
            return user
        except ValueError:
            pass
    path = users_path(settings)
    uid = str(user.get("user_id") or "")
    with _lock:
        users = _load(path)
        for idx, row in enumerate(users):
            if str(row.get("user_id") or "") != uid:
                continue
            existing = str(row.get("researcher_id") or "").strip()
            if existing:
                try:
                    row["researcher_id"] = normalise_researcher_id(existing)
                except ValueError:
                    row["researcher_id"] = _generate_researcher_id(_assigned_ids(users))
            else:
                row["researcher_id"] = _generate_researcher_id(_assigned_ids(users))
            users[idx] = row
            _save(path, users)
            user = dict(row)
            return user
        # User not on disk yet (shouldn't happen for persisted accounts).
        user["researcher_id"] = _generate_researcher_id(_assigned_ids(users))
        return user


def find_by_email(settings: Settings, email: str) -> dict[str, Any] | None:
    from ..supabase import users as sb_users

    if sb_users.enabled():
        return sb_users.find_by_email(email)
    wanted = normalise_email(email)
    with _lock:
        for user in _load(users_path(settings)):
            if normalise_email(str(user.get("email", ""))) == wanted:
                return user
    return None


def find_by_user_id(settings: Settings, user_id: str) -> dict[str, Any] | None:
    from ..supabase import users as sb_users

    if sb_users.enabled():
        return sb_users.find_by_user_id(user_id)
    uid = str(user_id or "").strip()
    if not uid:
        return None
    with _lock:
        for user in _load(users_path(settings)):
            if str(user.get("user_id") or "") == uid:
                return user
    return None


def find_users_by_name(settings: Settings, name: str) -> list[dict[str, Any]]:
    """Case-insensitive exact display-name matches. Empty if the name is blank."""
    from ..supabase import users as sb_users

    needle = (name or "").strip().lower()
    if len(needle) < 2:
        return []
    if sb_users.enabled():
        return sb_users.find_users_by_name(needle)
    with _lock:
        return [
            user
            for user in _load(users_path(settings))
            if str(user.get("name") or "").strip().lower() == needle
        ]


def find_by_researcher_id(settings: Settings, researcher_id: str) -> dict[str, Any] | None:
    from ..supabase import users as sb_users

    try:
        wanted = normalise_researcher_id(researcher_id)
    except ValueError:
        return None
    if sb_users.enabled():
        return sb_users.find_by_researcher_id(wanted)
    with _lock:
        users = _load(users_path(settings))
        for user in users:
            rid = str(user.get("researcher_id") or "").strip().upper()
            if rid == wanted:
                return user
    return None


def create_user(settings: Settings, email: str, password: str, name: str = "") -> dict[str, Any]:
    from ..supabase import users as sb_users

    if sb_users.enabled():
        return sb_users.create_user(email, hash_password(password), name=name)

    path = users_path(settings)
    with _lock:
        users = _load(path)
        email_n = normalise_email(email)
        if any(normalise_email(str(u.get("email", ""))) == email_n for u in users):
            raise ValueError("email_taken")
        record = {
            "user_id": uuid.uuid4().hex,
            "email": email_n,
            "name": (name or "").strip()[:80],
            "password_hash": hash_password(password),
            "researcher_id": _generate_researcher_id(_assigned_ids(users)),
        }
        users.append(record)
        _save(path, users)
    return record


def authenticate(settings: Settings, email: str, password: str) -> dict[str, Any] | None:
    user = find_by_email(settings, email)
    if user is None:
        return None
    stored = str(user.get("password_hash") or "")
    if not stored:
        return None
    if not verify_password(password, stored):
        return None
    return ensure_researcher_id(settings, user)


def find_by_google_sub(settings: Settings, google_sub: str) -> dict[str, Any] | None:
    from ..supabase import users as sb_users

    if sb_users.enabled():
        return sb_users.find_by_google_sub(google_sub)
    sub = str(google_sub or "").strip()
    if not sub:
        return None
    with _lock:
        for user in _load(users_path(settings)):
            if str(user.get("google_sub") or "") == sub:
                return user
    return None


def upsert_google_user(
    settings: Settings,
    *,
    google_sub: str,
    email: str,
    name: str = "",
) -> dict[str, Any]:
    """Create or link a user authenticated via Google."""
    from ..supabase import users as sb_users

    sub = str(google_sub or "").strip()
    email_n = normalise_email(email)
    if not sub:
        raise ValueError("missing_google_sub")
    if not valid_email(email_n):
        raise ValueError("invalid_email")
    display = (name or "").strip()[:80]

    if sb_users.enabled():
        return sb_users.upsert_google_user(google_sub=sub, email=email_n, name=display)

    path = users_path(settings)
    with _lock:
        users = _load(path)
        for idx, row in enumerate(users):
            if str(row.get("google_sub") or "") == sub:
                if display and not str(row.get("name") or "").strip():
                    row["name"] = display
                if email_n and normalise_email(str(row.get("email") or "")) != email_n:
                    # Keep original email if already set; still ensure google_sub link.
                    pass
                users[idx] = row
                _save(path, users)
                return dict(row)

        for idx, row in enumerate(users):
            if normalise_email(str(row.get("email") or "")) != email_n:
                continue
            row["google_sub"] = sub
            if display and not str(row.get("name") or "").strip():
                row["name"] = display
            if not str(row.get("auth_provider") or "").strip():
                row["auth_provider"] = "google" if not str(row.get("password_hash") or "") else "linked"
            users[idx] = row
            _save(path, users)
            return dict(row)

        record = {
            "user_id": uuid.uuid4().hex,
            "email": email_n,
            "name": display,
            "password_hash": "",
            "google_sub": sub,
            "auth_provider": "google",
            "researcher_id": _generate_researcher_id(_assigned_ids(users)),
        }
        users.append(record)
        _save(path, users)
        return dict(record)


def public_user(user: dict[str, Any]) -> dict[str, str]:
    return {
        "user_id": str(user.get("user_id", "")),
        "email": str(user.get("email", "")),
        "name": str(user.get("name", "")),
        "researcher_id": str(user.get("researcher_id", "")),
    }


def researcher_public(user: dict[str, Any]) -> dict[str, str]:
    """Minimum safe identity for Find Researcher (never email or secrets)."""
    name = (str(user.get("name") or "")).strip()
    return {
        "researcher_id": str(user.get("researcher_id") or ""),
        "display_name": name or "Researcher",
    }
