"""Per-user Ask chat threads for Workspace Recents (ChatGPT-style)."""

from __future__ import annotations

import json
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ...config import BACKEND_ROOT
from ...models.workspace import AskChatRecord, AskChatTurn

_ROOT = BACKEND_ROOT / "data" / "workspace" / "chats"
_LOCK = threading.RLock()
_MAX_PER_USER = 40


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _user_path(user_id: str) -> Path:
    safe = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in (user_id or "anon"))
    root = _ROOT.resolve()
    root.mkdir(parents=True, exist_ok=True)
    path = (root / f"{safe}.json").resolve()
    if path.parent != root:
        raise ValueError("Invalid ask chat path.")
    return path


def _read(user_id: str) -> list[dict[str, Any]]:
    path = _user_path(user_id)
    if not path.exists():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    items = payload.get("chats", []) if isinstance(payload, dict) else payload
    return items if isinstance(items, list) else []


def _write(user_id: str, items: list[dict[str, Any]]) -> None:
    path = _user_path(user_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"chats": items}, indent=2) + "\n", encoding="utf-8")


def _title_from_turns(title: str, turns: list[AskChatTurn]) -> str:
    cleaned = (title or "").strip()
    if cleaned:
        return cleaned[:120]
    for turn in turns:
        q = (turn.query or "").strip()
        if q:
            return q[:120]
    return "New chat"


def save_chat(
    user_id: str,
    *,
    chat_id: str | None,
    title: str = "",
    turns: list[AskChatTurn] | list[dict[str, Any]] | None = None,
) -> AskChatRecord:
    uid = str(user_id or "").strip()
    if not uid:
        raise ValueError("user_id is required.")
    parsed: list[AskChatTurn] = []
    for raw in turns or []:
        if isinstance(raw, AskChatTurn):
            parsed.append(raw)
        else:
            parsed.append(AskChatTurn.model_validate(raw))
    if not parsed:
        raise ValueError("At least one turn is required.")

    now = _now()
    cid = str(chat_id or "").strip() or str(uuid.uuid4())
    with _LOCK:
        items = _read(uid)
        existing: dict[str, Any] | None = None
        for row in items:
            if str(row.get("chat_id")) == cid:
                existing = row
                break
        record = AskChatRecord(
            chat_id=cid,
            user_id=uid,
            title=_title_from_turns(title or (existing or {}).get("title", ""), parsed),
            created_at=str((existing or {}).get("created_at") or now),
            updated_at=now,
            turns=parsed,
        )
        payload = record.model_dump(mode="json")
        items = [row for row in items if str(row.get("chat_id")) != cid]
        items.append(payload)
        items = items[-_MAX_PER_USER:]
        _write(uid, items)
    return record


def list_chats(user_id: str, limit: int = 24) -> list[AskChatRecord]:
    uid = str(user_id or "").strip()
    if not uid:
        return []
    with _LOCK:
        items = _read(uid)
    out: list[AskChatRecord] = []
    for raw in reversed(items):
        try:
            out.append(AskChatRecord.model_validate(raw))
        except Exception:
            continue
        if len(out) >= limit:
            break
    return out


def get_chat(user_id: str, chat_id: str) -> AskChatRecord | None:
    uid = str(user_id or "").strip()
    cid = str(chat_id or "").strip()
    if not uid or not cid:
        return None
    with _LOCK:
        items = _read(uid)
    for raw in reversed(items):
        if str(raw.get("chat_id")) != cid:
            continue
        try:
            return AskChatRecord.model_validate(raw)
        except Exception:
            return None
    return None


def delete_chat(user_id: str, chat_id: str) -> bool:
    uid = str(user_id or "").strip()
    cid = str(chat_id or "").strip()
    if not uid or not cid:
        return False
    with _LOCK:
        items = _read(uid)
        kept = [row for row in items if str(row.get("chat_id")) != cid]
        if len(kept) == len(items):
            return False
        _write(uid, kept)
        return True


def migrate_legacy_queries(user_id: str) -> int:
    """Promote single-query Ask history into chat threads (idempotent)."""
    from . import ask_queries

    uid = str(user_id or "").strip()
    if not uid:
        return 0
    existing = {chat.chat_id for chat in list_chats(uid, limit=_MAX_PER_USER)}
    covered_query_ids: set[str] = set()
    for chat in list_chats(uid, limit=_MAX_PER_USER):
        for turn in chat.turns:
            qid = ""
            if isinstance(turn.result, dict):
                qid = str(turn.result.get("query_id") or "")
            if qid:
                covered_query_ids.add(qid)

    created = 0
    for entry in ask_queries.list_ask_queries(uid, limit=_MAX_PER_USER):
        result = entry["result"]
        qid = str(result.query_id or "").strip()
        if not qid or qid in existing or qid in covered_query_ids:
            continue
        save_chat(
            uid,
            chat_id=qid,
            title=result.query,
            turns=[
                AskChatTurn(
                    query=result.query,
                    result=result.model_dump(mode="json"),
                    error="",
                )
            ],
        )
        # Preserve original timestamps when possible.
        with _LOCK:
            items = _read(uid)
            for row in items:
                if str(row.get("chat_id")) == qid:
                    created_at = str(entry.get("created_at") or row.get("created_at") or _now())
                    row["created_at"] = created_at
                    row["updated_at"] = created_at
                    break
            _write(uid, items)
        created += 1
        existing.add(qid)
        covered_query_ids.add(qid)
    return created
