"""Per-user paper search history for Workspace recents."""

from __future__ import annotations

import json
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path

from ...config import BACKEND_ROOT
from ...models.papers import PaperHit
from ...models.workspace import PaperSearchRecord

_ROOT = BACKEND_ROOT / "data" / "workspace"
_LOCK = threading.RLock()
_MAX_PER_USER = 50


def _user_path(user_id: str) -> Path:
    safe = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in (user_id or "anon"))
    root = _ROOT.resolve()
    path = (root / f"{safe}.json").resolve()
    if path.parent != root:
        raise ValueError("Invalid workspace path.")
    return path


def _read(user_id: str) -> list[dict]:
    path = _user_path(user_id)
    if not path.exists():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    items = payload.get("searches", []) if isinstance(payload, dict) else payload
    return items if isinstance(items, list) else []


def _write(user_id: str, items: list[dict]) -> None:
    path = _user_path(user_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"searches": items}, indent=2) + "\n", encoding="utf-8")


def save_paper_search(
    user_id: str,
    *,
    query: str,
    provider: str = "",
    papers: list[PaperHit] | list[dict] | None = None,
) -> PaperSearchRecord:
    text = (query or "").strip()
    if not text:
        raise ValueError("Query is required.")
    hits: list[dict] = []
    for paper in papers or []:
        if isinstance(paper, PaperHit):
            hits.append(paper.model_dump())
        elif isinstance(paper, dict):
            hits.append(PaperHit.model_validate(paper).model_dump())
    record = PaperSearchRecord(
        search_id=str(uuid.uuid4()),
        user_id=user_id,
        query=text,
        provider=provider or "",
        papers=[PaperHit.model_validate(h) for h in hits],
        created_at=datetime.now(timezone.utc).isoformat(),
    )
    with _LOCK:
        items = _read(user_id)
        items.append(record.model_dump(mode="json"))
        items = items[-_MAX_PER_USER:]
        _write(user_id, items)
    return record


def list_paper_searches(user_id: str, limit: int = 20) -> list[PaperSearchRecord]:
    with _LOCK:
        items = _read(user_id)
    out: list[PaperSearchRecord] = []
    for raw in reversed(items):
        try:
            out.append(PaperSearchRecord.model_validate(raw))
        except Exception:
            continue
        if len(out) >= limit:
            break
    return out


def get_paper_search(user_id: str, search_id: str) -> PaperSearchRecord | None:
    with _LOCK:
        items = _read(user_id)
    for raw in reversed(items):
        if str(raw.get("search_id")) == search_id:
            try:
                return PaperSearchRecord.model_validate(raw)
            except Exception:
                return None
    return None


def delete_paper_search(user_id: str, search_id: str) -> bool:
    with _LOCK:
        items = _read(user_id)
        kept = [raw for raw in items if str(raw.get("search_id")) != search_id]
        if len(kept) == len(items):
            return False
        _write(user_id, kept)
        return True
