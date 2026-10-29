"""Supabase dual-backend helpers (offline / mocked)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from app.config import get_settings
from app.services.supabase import supabase_enabled
from app.services.supabase.client import rest_select


def test_supabase_disabled_by_default(monkeypatch):
    monkeypatch.delenv("SELFRAG_SUPABASE_URL", raising=False)
    monkeypatch.delenv("SELFRAG_SUPABASE_SERVICE_KEY", raising=False)
    get_settings.cache_clear()
    assert supabase_enabled() is False
    get_settings.cache_clear()


def test_supabase_enabled_when_both_set(monkeypatch):
    monkeypatch.setenv("SELFRAG_SUPABASE_URL", "https://example.supabase.co")
    monkeypatch.setenv("SELFRAG_SUPABASE_SERVICE_KEY", "service-role-test-key")
    get_settings.cache_clear()
    assert supabase_enabled() is True
    get_settings.cache_clear()
    monkeypatch.delenv("SELFRAG_SUPABASE_URL", raising=False)
    monkeypatch.delenv("SELFRAG_SUPABASE_SERVICE_KEY", raising=False)
    get_settings.cache_clear()


def test_rest_select_builds_request(monkeypatch):
    monkeypatch.setenv("SELFRAG_SUPABASE_URL", "https://example.supabase.co")
    monkeypatch.setenv("SELFRAG_SUPABASE_SERVICE_KEY", "service-role-test-key")
    get_settings.cache_clear()

    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()
    mock_response.json.return_value = [{"user_id": "abc", "email": "a@b.co"}]

    with patch("app.services.supabase.client.httpx.get", return_value=mock_response) as get:
        rows = rest_select("app_users", params={"email": "eq.a@b.co", "select": "*"})

    assert rows == [{"user_id": "abc", "email": "a@b.co"}]
    assert get.call_args.args[0] == "https://example.supabase.co/rest/v1/app_users"
    headers = get.call_args.kwargs["headers"]
    assert headers["apikey"] == "service-role-test-key"
    assert headers["Authorization"] == "Bearer service-role-test-key"

    monkeypatch.delenv("SELFRAG_SUPABASE_URL", raising=False)
    monkeypatch.delenv("SELFRAG_SUPABASE_SERVICE_KEY", raising=False)
    get_settings.cache_clear()
