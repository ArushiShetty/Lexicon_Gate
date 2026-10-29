"""Per-user Ask query history for Workspace recents."""

from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ...config import BACKEND_ROOT
from ...models.query import PipelineResult

_ROOT = BACKEND_ROOT / "data" / "workspace" / "ask"
_LOCK = threading.RLock()
_MAX_PER_USER = 50


def _user_path(user_id: str) -> Path:
    safe = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in (user_id or "anon"))
    root = _ROOT.resolve()
    root.mkdir(parents=True, exist_ok=True)
    path = (root / f"{safe}.json").resolve()
    if path.parent != root:
        raise ValueError("Invalid ask history path.")
    return path


def _read(user_id: str) -> list[dict[str, Any]]:
    path = _user_path(user_id)
    if not path.exists():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    items = payload.get("queries", []) if isinstance(payload, dict) else payload
    return items if isinstance(items, list) else []


def _write(user_id: str, items: list[dict[str, Any]]) -> None:
    path = _user_path(user_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"queries": items}, indent=2) + "\n", encoding="utf-8")


def save_ask_query(user_id: str, result: PipelineResult) -> None:
    uid = str(user_id or "").strip()
    if not uid:
        return
    record = {
        "query_id": result.query_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "result": result.model_dump(mode="json"),
    }
    with _LOCK:
        items = [row for row in _read(uid) if str(row.get("query_id")) != result.query_id]
        items.append(record)
        items = items[-_MAX_PER_USER:]
        _write(uid, items)


def list_ask_queries(user_id: str, limit: int = 20) -> list[dict[str, Any]]:
    uid = str(user_id or "").strip()
    if not uid:
        return []
    with _LOCK:
        items = _read(uid)
    out: list[dict[str, Any]] = []
    for raw in reversed(items):
        if not isinstance(raw, dict):
            continue
        result_raw = raw.get("result")
        if not isinstance(result_raw, dict):
            continue
        try:
            result = PipelineResult.model_validate(result_raw)
        except Exception:
            continue
        out.append(
            {
                "query_id": result.query_id,
                "created_at": str(raw.get("created_at") or ""),
                "result": result,
            }
        )
        if len(out) >= limit:
            break
    return out


def get_ask_query(user_id: str, query_id: str) -> PipelineResult | None:
    uid = str(user_id or "").strip()
    qid = str(query_id or "").strip()
    if not uid or not qid:
        return None
    with _LOCK:
        items = _read(uid)
    for raw in reversed(items):
        if str(raw.get("query_id")) != qid:
            continue
        result_raw = raw.get("result")
        if not isinstance(result_raw, dict):
            return None
        try:
            return PipelineResult.model_validate(result_raw)
        except Exception:
            return None
    return None


def delete_ask_query(user_id: str, query_id: str) -> bool:
    uid = str(user_id or "").strip()
    qid = str(query_id or "").strip()
    if not uid or not qid:
        return False
    with _LOCK:
        items = _read(uid)
        kept = [row for row in items if str(row.get("query_id")) != qid]
        if len(kept) == len(items):
            return False
        _write(uid, kept)
        return True
