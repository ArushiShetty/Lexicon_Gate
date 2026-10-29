"""Opaque session tokens, kept on disk so reloads keep you signed in."""

from __future__ import annotations

import json
import secrets
import threading
import time
from pathlib import Path
from typing import Any

_lock = threading.Lock()
_sessions: dict[str, dict[str, Any]] = {}
_loaded_from: Path | None = None


def _path() -> Path:
    from ...config import get_settings

    path = Path(get_settings().data_dir) / "auth" / "sessions.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _hydrate() -> None:
    global _loaded_from
    path = _path()
    if _loaded_from == path:
        return
    _sessions.clear()
    _loaded_from = path
    if not path.exists():
        return
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return
    records = payload.get("sessions", payload) if isinstance(payload, dict) else {}
    if isinstance(records, dict):
        for token, record in records.items():
            if isinstance(record, dict):
                _sessions[token] = record


def _persist() -> None:
    path = _path()
    path.write_text(json.dumps({"sessions": _sessions}, indent=2), encoding="utf-8")


def create_session(user: dict[str, Any], ttl_seconds: int) -> str:
    from ..supabase import sessions as sb_sessions

    if sb_sessions.enabled():
        return sb_sessions.create_session(user, ttl_seconds)

    token = secrets.token_urlsafe(32)
    expires = time.time() + max(60, int(ttl_seconds))
    with _lock:
        _hydrate()
        _sessions[token] = {
            "user_id": user.get("user_id"),
            "email": user.get("email"),
            "name": user.get("name", ""),
            "researcher_id": user.get("researcher_id", ""),
            "expires_at": expires,
        }
        _persist()
    return token


def get_session(token: str | None) -> dict[str, Any] | None:
    from ..supabase import sessions as sb_sessions

    if sb_sessions.enabled():
        return sb_sessions.get_session(token)

    if not token:
        return None
    now = time.time()
    with _lock:
        _hydrate()
        record = _sessions.get(token)
        if record is None:
            return None
        if float(record.get("expires_at", 0)) < now:
            _sessions.pop(token, None)
            _persist()
            return None
        return dict(record)


def revoke_session(token: str | None) -> None:
    from ..supabase import sessions as sb_sessions

    if sb_sessions.enabled():
        sb_sessions.revoke_session(token)
        return

    if not token:
        return
    with _lock:
        _hydrate()
        if token in _sessions:
            _sessions.pop(token, None)
            _persist()


def clear_sessions() -> None:
    from ..supabase import sessions as sb_sessions

    if sb_sessions.enabled():
        sb_sessions.clear_sessions()
        return

    global _loaded_from
    with _lock:
        _sessions.clear()
        path = _path()
        if path.exists():
            path.unlink()
        _loaded_from = None
