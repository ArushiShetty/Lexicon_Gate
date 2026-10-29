"""Supabase PostgREST client (optional persistence).

When ``SELFRAG_SUPABASE_URL`` and ``SELFRAG_SUPABASE_SERVICE_KEY`` are set,
auth users/sessions and Write stories are stored in Postgres via Supabase.
Otherwise the app keeps using local JSON files.
"""

from __future__ import annotations

from typing import Any

import httpx

from ...config import get_settings


def supabase_enabled() -> bool:
    settings = get_settings()
    return bool((settings.supabase_url or "").strip() and (settings.supabase_service_key or "").strip())


def _base_url() -> str:
    return (get_settings().supabase_url or "").rstrip("/")


def _headers(*, prefer: str | None = None) -> dict[str, str]:
    key = (get_settings().supabase_service_key or "").strip()
    headers = {
        "apikey": key,
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
    }
    if prefer:
        headers["Prefer"] = prefer
    return headers


def rest_select(
    table: str,
    *,
    params: dict[str, str] | None = None,
    timeout: float = 20.0,
) -> list[dict[str, Any]]:
    url = f"{_base_url()}/rest/v1/{table}"
    response = httpx.get(url, headers=_headers(), params=params or {}, timeout=timeout)
    response.raise_for_status()
    payload = response.json()
    if isinstance(payload, list):
        return [row for row in payload if isinstance(row, dict)]
    return []


def rest_insert(
    table: str,
    rows: dict[str, Any] | list[dict[str, Any]],
    *,
    timeout: float = 20.0,
) -> list[dict[str, Any]]:
    url = f"{_base_url()}/rest/v1/{table}"
    response = httpx.post(
        url,
        headers=_headers(prefer="return=representation"),
        json=rows,
        timeout=timeout,
    )
    response.raise_for_status()
    payload = response.json()
    if isinstance(payload, list):
        return [row for row in payload if isinstance(row, dict)]
    if isinstance(payload, dict):
        return [payload]
    return []


def rest_upsert(
    table: str,
    rows: dict[str, Any] | list[dict[str, Any]],
    *,
    on_conflict: str,
    timeout: float = 20.0,
) -> list[dict[str, Any]]:
    url = f"{_base_url()}/rest/v1/{table}"
    response = httpx.post(
        url,
        headers=_headers(prefer="resolution=merge-duplicates,return=representation"),
        params={"on_conflict": on_conflict},
        json=rows,
        timeout=timeout,
    )
    response.raise_for_status()
    payload = response.json()
    if isinstance(payload, list):
        return [row for row in payload if isinstance(row, dict)]
    if isinstance(payload, dict):
        return [payload]
    return []


def rest_patch(
    table: str,
    match: dict[str, str],
    patch: dict[str, Any],
    *,
    timeout: float = 20.0,
) -> list[dict[str, Any]]:
    url = f"{_base_url()}/rest/v1/{table}"
    response = httpx.patch(
        url,
        headers=_headers(prefer="return=representation"),
        params=match,
        json=patch,
        timeout=timeout,
    )
    response.raise_for_status()
    payload = response.json()
    if isinstance(payload, list):
        return [row for row in payload if isinstance(row, dict)]
    if isinstance(payload, dict):
        return [payload]
    return []


def rest_delete(
    table: str,
    match: dict[str, str],
    *,
    timeout: float = 20.0,
) -> None:
    url = f"{_base_url()}/rest/v1/{table}"
    response = httpx.delete(url, headers=_headers(), params=match, timeout=timeout)
    response.raise_for_status()
