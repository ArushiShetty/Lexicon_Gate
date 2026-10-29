"""Supabase-backed story documents (app_stories.payload jsonb)."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from .client import rest_delete, rest_select, rest_upsert, supabase_enabled

_TABLE = "app_stories"


def save_story_payload(story_id: str, author_user_id: str, payload: dict[str, Any]) -> None:
    rest_upsert(
        _TABLE,
        {
            "story_id": story_id,
            "author_user_id": author_user_id,
            "payload": payload,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        },
        on_conflict="story_id",
    )


def load_story_payload(story_id: str) -> dict[str, Any] | None:
    rows = rest_select(
        _TABLE,
        params={"story_id": f"eq.{story_id}", "select": "payload", "limit": "1"},
    )
    if not rows:
        return None
    payload = rows[0].get("payload")
    return payload if isinstance(payload, dict) else None


def list_story_payloads() -> list[dict[str, Any]]:
    rows = rest_select(
        _TABLE,
        params={"select": "payload", "order": "updated_at.desc"},
    )
    out: list[dict[str, Any]] = []
    for row in rows:
        payload = row.get("payload")
        if isinstance(payload, dict):
            out.append(payload)
    return out


def delete_story(story_id: str) -> None:
    rest_delete(_TABLE, {"story_id": f"eq.{story_id}"})


def enabled() -> bool:
    return supabase_enabled()
