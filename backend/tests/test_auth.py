"""Signup, login, and session cookie tests."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.config import get_settings
from app.main import create_app
from app.services.auth.sessions import clear_sessions
from app.services.embeddings.registry import reset_embedding_cache
from app.services.llm.registry import reset_llm_cache
from app.services.store.knowledge_base import reset_knowledge_base
from app.services.vectorstore.registry import reset_vector_store_cache


@pytest.fixture
def auth_client(settings, monkeypatch):
    reset_knowledge_base()
    reset_llm_cache()
    reset_embedding_cache()
    reset_vector_store_cache()
    get_settings.cache_clear()
    monkeypatch.setenv("SELFRAG_DATA_DIR", str(settings.data_dir))
    monkeypatch.setenv("SELFRAG_LLM_PROVIDER", "extractive")
    monkeypatch.setenv("SELFRAG_EMBEDDING_PROVIDER", "lsa")
    monkeypatch.setenv("SELFRAG_VECTOR_STORE", "numpy")
    monkeypatch.setenv("SELFRAG_AUTH_REQUIRED", "true")
    from app import config as config_mod

    settings.auth_required = True
    config_mod.settings = settings
    clear_sessions()
    app = create_app()
    with TestClient(app) as test_client:
        yield test_client
    clear_sessions()
    reset_knowledge_base()
    get_settings.cache_clear()


def test_signup_sets_cookie_and_me(auth_client: TestClient):
    response = auth_client.post(
        "/api/auth/signup",
        json={"email": "Ada@Example.com", "password": "secret-pass", "name": "Ada"},
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["email"] == "ada@example.com"
    assert body["name"] == "Ada"
    assert "selfrag_session" in response.cookies
    me = auth_client.get("/api/auth/me")
    assert me.status_code == 200
    assert me.json()["email"] == "ada@example.com"


def test_duplicate_signup_rejected(auth_client: TestClient):
    payload = {"email": "user@example.com", "password": "secret-pass"}
    assert auth_client.post("/api/auth/signup", json=payload).status_code == 200
    again = auth_client.post("/api/auth/signup", json=payload)
    assert again.status_code == 409


def test_login_and_protected_documents(auth_client: TestClient):
    auth_client.post(
        "/api/auth/signup",
        json={"email": "user@example.com", "password": "secret-pass"},
    )
    auth_client.post("/api/auth/logout")
    denied = auth_client.get("/api/documents")
    assert denied.status_code == 401

    bad = auth_client.post(
        "/api/auth/login",
        json={"email": "user@example.com", "password": "wrong-password"},
    )
    assert bad.status_code == 401

    ok = auth_client.post(
        "/api/auth/login",
        json={"email": "user@example.com", "password": "secret-pass"},
    )
    assert ok.status_code == 200
    listed = auth_client.get("/api/documents")
    assert listed.status_code == 200
    assert listed.json()["total_documents"] == 0


def test_health_is_public(auth_client: TestClient):
    response = auth_client.get("/api/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_invalid_email_rejected(auth_client: TestClient):
    response = auth_client.post(
        "/api/auth/signup",
        json={"email": "not-an-email", "password": "secret-pass"},
    )
    assert response.status_code == 400


def test_google_auth_creates_session(auth_client: TestClient, monkeypatch):
    from app.api import auth as auth_api
    from app.config import get_settings

    monkeypatch.setenv("SELFRAG_GOOGLE_CLIENT_ID", "test-client.apps.googleusercontent.com")
    get_settings.cache_clear()
    monkeypatch.setattr(
        auth_api,
        "verify_google_id_token",
        lambda credential, client_id, timeout=12.0: {
            "google_sub": "google-sub-1",
            "email": "scholar@gmail.com",
            "name": "Scholar",
        },
    )
    response = auth_client.post("/api/auth/google", json={"credential": "fake.jwt.token.value"})
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["email"] == "scholar@gmail.com"
    assert body["name"] == "Scholar"
    assert body["researcher_id"].startswith("LG-")
    assert "selfrag_session" in response.cookies
    me = auth_client.get("/api/auth/me")
    assert me.status_code == 200
    assert me.json()["email"] == "scholar@gmail.com"


def test_google_auth_links_existing_email(auth_client: TestClient, monkeypatch):
    from app.api import auth as auth_api
    from app.config import get_settings

    monkeypatch.setenv("SELFRAG_GOOGLE_CLIENT_ID", "test-client.apps.googleusercontent.com")
    get_settings.cache_clear()
    auth_client.post(
        "/api/auth/signup",
        json={"email": "linked@example.com", "password": "secret-pass", "name": "Linked"},
    )
    auth_client.post("/api/auth/logout")
    monkeypatch.setattr(
        auth_api,
        "verify_google_id_token",
        lambda credential, client_id, timeout=12.0: {
            "google_sub": "google-sub-linked",
            "email": "linked@example.com",
            "name": "Linked Google",
        },
    )
    response = auth_client.post("/api/auth/google", json={"credential": "fake.jwt.token.value"})
    assert response.status_code == 200, response.text
    assert response.json()["email"] == "linked@example.com"
    assert response.json()["name"] == "Linked"


def test_google_auth_requires_config(auth_client: TestClient, monkeypatch):
    from app.config import get_settings

    monkeypatch.setenv("SELFRAG_GOOGLE_CLIENT_ID", "")
    get_settings.cache_clear()
    response = auth_client.post("/api/auth/google", json={"credential": "fake.jwt.token.value"})
    assert response.status_code == 503
