"""Researcher ID generation, lookup, and project invite-by-ID tests."""

from __future__ import annotations

import json

from fastapi.testclient import TestClient

from app.config import get_settings
from app.main import create_app
from app.services.auth.passwords import hash_password
from app.services.auth.sessions import clear_sessions
from app.services.auth import users as users_mod
from app.services.projects import store as project_store


def _prepare(settings, monkeypatch, tmp_path):
    monkeypatch.setenv("SELFRAG_AUTH_REQUIRED", "true")
    monkeypatch.setenv("SELFRAG_DATA_DIR", str(tmp_path))
    get_settings.cache_clear()
    clear_sessions()
    settings.auth_required = True
    settings.data_dir = tmp_path
    from app import config as config_mod

    config_mod.settings = settings
    root = tmp_path / "projects"
    monkeypatch.setattr(project_store, "_ROOT", root)
    monkeypatch.setattr(project_store, "_INVITES_ROOT", root / "_invites")


def _client():
    return TestClient(create_app())


def _signup(client: TestClient, email: str, name: str = "User"):
    resp = client.post(
        "/api/auth/signup",
        json={"email": email, "password": "secret-pass", "name": name},
    )
    assert resp.status_code == 200, resp.text
    return resp.json()


def test_researcher_id_on_signup_and_stable(settings, monkeypatch, tmp_path):
    _prepare(settings, monkeypatch, tmp_path)
    client = _client()
    body = _signup(client, "rid@example.com", "Rid")
    rid = body["researcher_id"]
    assert rid.startswith("LG-")
    assert len(rid.replace("-", "")) == 10
    me = client.get("/api/auth/me").json()
    assert me["researcher_id"] == rid
    client.post("/api/auth/logout")
    again = client.post(
        "/api/auth/login",
        json={"email": "rid@example.com", "password": "secret-pass"},
    )
    assert again.status_code == 200
    assert again.json()["researcher_id"] == rid


def test_lazy_researcher_id_for_legacy_user(settings, monkeypatch, tmp_path):
    _prepare(settings, monkeypatch, tmp_path)
    path = users_mod.users_path(get_settings())
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "users": [
                    {
                        "user_id": "abc123",
                        "email": "legacy@example.com",
                        "name": "Legacy",
                        "password_hash": hash_password("secret-pass"),
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    client = _client()
    login = client.post(
        "/api/auth/login",
        json={"email": "legacy@example.com", "password": "secret-pass"},
    )
    assert login.status_code == 200, login.text
    assert login.json()["researcher_id"].startswith("LG-")
    me = client.get("/api/auth/me").json()
    assert me["researcher_id"] == login.json()["researcher_id"]


def test_lookup_researcher_safe_card(settings, monkeypatch, tmp_path):
    _prepare(settings, monkeypatch, tmp_path)
    owner = _client()
    _signup(owner, "own@example.com", "Owner")
    guest = _client()
    guest_body = _signup(guest, "guest@example.com", "Guest")
    rid = guest_body["researcher_id"]

    owner2 = _client()
    owner2.post("/api/auth/login", json={"email": "own@example.com", "password": "secret-pass"})
    found = owner2.get(f"/api/auth/researchers/{rid}")
    assert found.status_code == 200
    body = found.json()
    assert body["researcher_id"] == rid
    assert body["display_name"] == "Guest"
    assert "email" not in body
    assert owner2.get("/api/auth/researchers/LG-AAAA-AAAA").status_code == 404
    assert owner2.get("/api/auth/researchers/not-an-id").status_code == 400


def test_invite_by_researcher_id_accept_reject(settings, monkeypatch, tmp_path):
    _prepare(settings, monkeypatch, tmp_path)
    owner = _client()
    _signup(owner, "o2@example.com", "Owner")
    project_id = owner.post("/api/projects/", json={"title": "Collab"}).json()["project_id"]

    guest = _client()
    guest_body = _signup(guest, "g2@example.com", "Guest")
    rid = guest_body["researcher_id"]

    owner2 = _client()
    owner2.post("/api/auth/login", json={"email": "o2@example.com", "password": "secret-pass"})
    created = owner2.post(
        f"/api/projects/{project_id}/invites",
        json={"researcher_id": rid, "role": "editor"},
    )
    assert created.status_code == 200, created.text
    assert created.json()["token"] == ""
    invite_id = created.json()["invite"]["invite_id"]
    assert created.json()["invite"]["recipient_researcher_id"] == rid

    assert guest.get(f"/api/projects/{project_id}").status_code == 404

    # Owner must not see the invite in their own pending inbox.
    owner_pending = owner2.get("/api/projects/invites/pending").json()
    assert not any(i["invite_id"] == invite_id for i in owner_pending.get("invites") or [])

    pending = guest.get("/api/projects/invites/pending").json()
    assert pending["total"] >= 1
    assert any(i["invite_id"] == invite_id and i["can_accept_in_app"] for i in pending["invites"])

    # Unrelated account: no inbox entry; accept without token is wrong-account (not token required).
    stranger = _client()
    _signup(stranger, "stranger2@example.com", "Stranger")
    stranger_pending = stranger.get("/api/projects/invites/pending").json()
    assert not any(i["invite_id"] == invite_id for i in stranger_pending.get("invites") or [])
    wrong = stranger.post("/api/projects/invites/accept", json={"invite_id": invite_id})
    assert wrong.status_code == 400
    assert "different Lexicon Gate account" in wrong.json()["detail"]

    accepted = guest.post("/api/projects/invites/accept", json={"invite_id": invite_id})
    assert accepted.status_code == 200
    assert any(
        m["email"] == "g2@example.com" and m["role"] == "editor" for m in accepted.json()["members"]
    )

    self_inv = owner2.post(
        f"/api/projects/{project_id}/invites",
        json={"researcher_id": owner2.get("/api/auth/me").json()["researcher_id"], "role": "viewer"},
    )
    assert self_inv.status_code == 400


def test_reject_invite_by_researcher(settings, monkeypatch, tmp_path):
    _prepare(settings, monkeypatch, tmp_path)
    owner = _client()
    _signup(owner, "o3@example.com", "Owner")
    project_id = owner.post("/api/projects/", json={"title": "Reject"}).json()["project_id"]
    guest = _client()
    guest_body = _signup(guest, "g3@example.com", "Guest")

    owner2 = _client()
    owner2.post("/api/auth/login", json={"email": "o3@example.com", "password": "secret-pass"})
    invite = owner2.post(
        f"/api/projects/{project_id}/invites",
        json={"researcher_id": guest_body["researcher_id"], "role": "viewer"},
    ).json()
    invite_id = invite["invite"]["invite_id"]
    assert guest.post("/api/projects/invites/reject", json={"invite_id": invite_id}).status_code == 200
    assert guest.post("/api/projects/invites/accept", json={"invite_id": invite_id}).status_code == 400
    assert guest.get(f"/api/projects/{project_id}").status_code == 404
