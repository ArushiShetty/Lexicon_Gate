"""Research projects collaboration and RBAC tests."""

from __future__ import annotations

from fastapi.testclient import TestClient

from app.main import create_app
from app.services.auth.sessions import clear_sessions
from app.services.auth.users import create_user
from app.services.projects import store as project_store


def _auth_client(settings, monkeypatch, tmp_path, email: str, name: str = "User"):
    monkeypatch.setenv("SELFRAG_AUTH_REQUIRED", "true")
    from app.config import get_settings

    get_settings.cache_clear()
    clear_sessions()
    root = tmp_path / "projects"
    monkeypatch.setattr(project_store, "_ROOT", root)
    monkeypatch.setattr(project_store, "_INVITES_ROOT", root / "_invites")
    create_user(settings, email, "secret-pass", name)
    app = create_app()
    client = TestClient(app)
    login = client.post("/api/auth/login", json={"email": email, "password": "secret-pass"})
    assert login.status_code == 200
    return client


def _second_client(settings, email: str, name: str = "Other", *, create: bool = True):
    """New session on the already-patched project store."""
    clear_sessions()
    if create:
        try:
            create_user(settings, email, "secret-pass", name)
        except ValueError:
            pass
    app = create_app()
    client = TestClient(app)
    login = client.post("/api/auth/login", json={"email": email, "password": "secret-pass"})
    assert login.status_code == 200
    return client


def _relogin(email: str) -> TestClient:
    clear_sessions()
    client = TestClient(create_app())
    login = client.post("/api/auth/login", json={"email": email, "password": "secret-pass"})
    assert login.status_code == 200
    return client


def test_create_list_get_update_archive_delete(settings, monkeypatch, tmp_path):
    client = _auth_client(settings, monkeypatch, tmp_path, "owner@example.com", "Owner")
    created = client.post(
        "/api/projects/",
        json={"title": "Self-RAG verification", "topic": "Evidence-gated QA", "description": "x"},
    )
    assert created.status_code == 200
    project_id = created.json()["project_id"]
    assert created.json()["activity"]
    assert client.get("/api/projects/").json()["total"] >= 1
    assert client.get(f"/api/projects/{project_id}").status_code == 200
    updated = client.patch(f"/api/projects/{project_id}", json={"title": "Self-RAG v2"})
    assert updated.status_code == 200
    assert updated.json()["title"] == "Self-RAG v2"
    assert client.post(f"/api/projects/{project_id}/archive").json()["status"] == "archived"
    assert client.delete(f"/api/projects/{project_id}").status_code == 204
    assert client.get(f"/api/projects/{project_id}").status_code == 404


def test_create_requires_title(settings, monkeypatch, tmp_path):
    client = _auth_client(settings, monkeypatch, tmp_path, "valid@example.com")
    assert client.post("/api/projects/", json={"title": "", "topic": "x"}).status_code == 422
    assert client.post("/api/projects/", json={"title": "   ", "topic": "x"}).status_code == 400


def test_requires_auth(settings, monkeypatch, tmp_path):
    monkeypatch.setenv("SELFRAG_AUTH_REQUIRED", "true")
    from app.config import get_settings

    get_settings.cache_clear()
    clear_sessions()
    root = tmp_path / "projects"
    monkeypatch.setattr(project_store, "_ROOT", root)
    monkeypatch.setattr(project_store, "_INVITES_ROOT", root / "_invites")
    client = TestClient(create_app())
    assert client.get("/api/projects/").status_code == 401


def test_nonexistent_project_is_404(settings, monkeypatch, tmp_path):
    client = _auth_client(settings, monkeypatch, tmp_path, "miss@example.com")
    assert client.get("/api/projects/does-not-exist").status_code == 404


def test_other_user_cannot_access(settings, monkeypatch, tmp_path):
    owner = _auth_client(settings, monkeypatch, tmp_path, "proj.owner@example.com", "Owner")
    project_id = owner.post(
        "/api/projects/",
        json={"title": "Private project", "topic": "secret"},
    ).json()["project_id"]
    other = _second_client(settings, "intruder@example.com", "Intruder")
    assert other.get(f"/api/projects/{project_id}").status_code == 404
    assert other.patch(f"/api/projects/{project_id}", json={"title": "Hijacked"}).status_code == 404


def test_invite_accept_role_change_and_rbac(settings, monkeypatch, tmp_path):
    owner = _auth_client(settings, monkeypatch, tmp_path, "collab.owner@example.com", "Owner")
    project_id = owner.post(
        "/api/projects/",
        json={"title": "Collab project", "topic": "RBAC"},
    ).json()["project_id"]

    editor_user = create_user(settings, "editor@example.com", "secret-pass", "Editor")
    invite_resp = owner.post(
        f"/api/projects/{project_id}/invites",
        json={"email": "editor@example.com", "role": "editor"},
    )
    assert invite_resp.status_code == 200
    invite_body = invite_resp.json()
    invite_id = invite_body["invite"]["invite_id"]
    token = invite_body["token"]
    assert token
    assert "token" not in invite_body["invite"]

    editor = _second_client(settings, "editor@example.com", "Editor", create=False)
    bad = editor.post(
        "/api/projects/invites/accept",
        json={"invite_id": invite_id, "token": "wrong-token-value-here"},
    )
    assert bad.status_code == 400

    ok = editor.post(
        "/api/projects/invites/accept",
        json={"invite_id": invite_id, "token": token},
    )
    assert ok.status_code == 200
    assert any(m["email"] == "editor@example.com" and m["role"] == "editor" for m in ok.json()["members"])

    note = editor.post(f"/api/projects/{project_id}/notes", json={"text": "Evidence gap on datasets."})
    assert note.status_code == 200
    assert note.json()["notes"][0]["text"].startswith("Evidence gap")
    meta = editor.patch(f"/api/projects/{project_id}", json={"topic": "Updated by editor"})
    assert meta.status_code == 200

    assert (
        editor.post(
            f"/api/projects/{project_id}/invites",
            json={"email": "someone@example.com", "role": "viewer"},
        ).status_code
        == 403
    )
    assert editor.post(f"/api/projects/{project_id}/archive").status_code == 403

    owner_login = _relogin("collab.owner@example.com")
    rev_invite = owner_login.post(
        f"/api/projects/{project_id}/invites",
        json={"email": "reviewer@example.com", "role": "reviewer"},
    )
    assert rev_invite.status_code == 200
    rev_token = rev_invite.json()["token"]
    rev_id = rev_invite.json()["invite"]["invite_id"]

    create_user(settings, "reviewer@example.com", "secret-pass", "Reviewer")
    reviewer = _second_client(settings, "reviewer@example.com", "Reviewer", create=False)
    assert (
        reviewer.post(
            "/api/projects/invites/accept",
            json={"invite_id": rev_id, "token": rev_token},
        ).status_code
        == 200
    )

    assert (
        reviewer.post(
            f"/api/projects/{project_id}/reviews",
            json={"text": "Needs stronger citations in intro."},
        ).status_code
        == 200
    )
    assert reviewer.patch(f"/api/projects/{project_id}", json={"title": "Nope"}).status_code == 403
    assert reviewer.post(f"/api/projects/{project_id}/notes", json={"text": "Nope"}).status_code == 403

    owner2 = _relogin("collab.owner@example.com")
    view_invite = owner2.post(
        f"/api/projects/{project_id}/invites",
        json={"email": "viewer@example.com", "role": "viewer"},
    )
    v_token = view_invite.json()["token"]
    v_id = view_invite.json()["invite"]["invite_id"]
    create_user(settings, "viewer@example.com", "secret-pass", "Viewer")
    viewer = _second_client(settings, "viewer@example.com", "Viewer", create=False)
    assert (
        viewer.post(
            "/api/projects/invites/accept",
            json={"invite_id": v_id, "token": v_token},
        ).status_code
        == 200
    )
    assert viewer.get(f"/api/projects/{project_id}").status_code == 200
    assert viewer.post(f"/api/projects/{project_id}/notes", json={"text": "x"}).status_code == 403
    assert viewer.post(f"/api/projects/{project_id}/reviews", json={"text": "x"}).status_code == 403

    owner3 = _relogin("collab.owner@example.com")
    editor_id = str(editor_user["user_id"])
    role_change = owner3.patch(
        f"/api/projects/{project_id}/members/{editor_id}",
        json={"role": "viewer"},
    )
    assert role_change.status_code == 200
    assert any(
        m["user_id"] == editor_id and m["role"] == "viewer" for m in role_change.json()["members"]
    )
    removed = owner3.delete(f"/api/projects/{project_id}/members/{editor_id}")
    assert removed.status_code == 200
    assert not any(m["user_id"] == editor_id for m in removed.json()["members"])

    detail = owner3.get(f"/api/projects/{project_id}").json()
    types = {a["type"] for a in detail["activity"]}
    assert "member_invited" in types
    assert "member_joined" in types
    assert "note_added" in types or "review_added" in types


def test_wrong_email_cannot_accept_invite(settings, monkeypatch, tmp_path):
    owner = _auth_client(settings, monkeypatch, tmp_path, "own2@example.com", "Owner")
    project_id = owner.post("/api/projects/", json={"title": "Email lock"}).json()["project_id"]
    invite = owner.post(
        f"/api/projects/{project_id}/invites",
        json={"email": "intended@example.com", "role": "viewer"},
    ).json()
    other = _second_client(settings, "other@example.com", "Other")
    resp = other.post(
        "/api/projects/invites/accept",
        json={"invite_id": invite["invite"]["invite_id"], "token": invite["token"]},
    )
    assert resp.status_code == 400
    assert "email" in resp.json()["detail"].lower()


def test_invite_expires_and_cannot_accept(settings, monkeypatch, tmp_path):
    from datetime import datetime, timedelta, timezone

    owner = _auth_client(settings, monkeypatch, tmp_path, "exp.owner@example.com", "Owner")
    project_id = owner.post("/api/projects/", json={"title": "Expire me"}).json()["project_id"]
    invite = owner.post(
        f"/api/projects/{project_id}/invites",
        json={"email": "exp.guest@example.com", "role": "viewer"},
    ).json()
    invite_id = invite["invite"]["invite_id"]
    token = invite["token"]
    assert invite["invite"]["expires_at"]

    secret_path = (tmp_path / "projects" / "_invites" / f"{invite_id}.json")
    import json

    secret = json.loads(secret_path.read_text(encoding="utf-8"))
    secret["expires_at"] = (datetime.now(timezone.utc) - timedelta(days=1)).isoformat()
    secret_path.write_text(json.dumps(secret), encoding="utf-8")

    guest = _second_client(settings, "exp.guest@example.com", "Guest")
    resp = guest.post(
        "/api/projects/invites/accept",
        json={"invite_id": invite_id, "token": token},
    )
    assert resp.status_code == 400
    assert "expir" in resp.json()["detail"].lower()

    # Project public invite status must sync to expired.
    owner2 = _relogin("exp.owner@example.com")
    detail = owner2.get(f"/api/projects/{project_id}").json()
    assert any(
        i["invite_id"] == invite_id and i["status"] == "expired" for i in detail["invites"]
    )


def test_pending_inbox_includes_invited_by_email(settings, monkeypatch, tmp_path):
    owner = _auth_client(settings, monkeypatch, tmp_path, "inbox.owner@example.com", "Owner")
    project_id = owner.post("/api/projects/", json={"title": "Inbox"}).json()["project_id"]
    owner.post(
        f"/api/projects/{project_id}/invites",
        json={"email": "inbox.guest@example.com", "role": "viewer"},
    )
    guest = _second_client(settings, "inbox.guest@example.com", "Guest")
    pending = guest.get("/api/projects/invites/pending").json()
    assert pending["total"] >= 1
    match = next(i for i in pending["invites"] if i["project_id"] == project_id)
    assert match["invited_by_email"] == "inbox.owner@example.com"


def test_already_member_accept_is_idempotent(settings, monkeypatch, tmp_path):
    """If the user is already a member, accepting a pending invite returns the project."""
    import json

    from app.models.projects import ProjectMember
    from app.services.projects import store as project_store

    owner = _auth_client(settings, monkeypatch, tmp_path, "idem.owner@example.com", "Owner")
    project_id = owner.post("/api/projects/", json={"title": "Idem"}).json()["project_id"]
    invite = owner.post(
        f"/api/projects/{project_id}/invites",
        json={"email": "placeholder.idem@example.com", "role": "viewer"},
    ).json()

    guest = _second_client(settings, "idem.guest@example.com", "Guest")
    # Resolve guest user_id from session cookie via a project they own.
    guest_project = guest.post("/api/projects/", json={"title": "Guest own"}).json()
    guest_id = guest_project["owner_id"]

    project = project_store.get_project(project_id)
    assert project is not None
    project.members = [
        *list(project.members),
        ProjectMember(
            user_id=guest_id,
            email="idem.guest@example.com",
            name="Guest",
            role="viewer",
        ),
    ]
    with project_store._store_lock():  # noqa: SLF001
        project_store._write_raw(project)  # noqa: SLF001

    invite_id = invite["invite"]["invite_id"]
    secret_path = tmp_path / "projects" / "_invites" / f"{invite_id}.json"
    secret = json.loads(secret_path.read_text(encoding="utf-8"))
    secret["email"] = "idem.guest@example.com"
    secret_path.write_text(json.dumps(secret), encoding="utf-8")

    result = project_store.accept_invite(
        invite_id=invite_id,
        token=invite["token"],
        user_id=guest_id,
        user_email="idem.guest@example.com",
        user_name="Guest",
    )
    assert any(m.user_id == guest_id for m in result.members)
    assert any(i.invite_id == invite_id and i.status == "accepted" for i in result.invites)


def test_revoked_invite_cannot_accept(settings, monkeypatch, tmp_path):
    owner = _auth_client(settings, monkeypatch, tmp_path, "rev.owner@example.com", "Owner")
    project_id = owner.post("/api/projects/", json={"title": "Revoke me"}).json()["project_id"]
    invite = owner.post(
        f"/api/projects/{project_id}/invites",
        json={"email": "rev.guest@example.com", "role": "editor"},
    ).json()
    invite_id = invite["invite"]["invite_id"]
    token = invite["token"]
    revoked = owner.delete(f"/api/projects/{project_id}/invites/{invite_id}")
    assert revoked.status_code == 200
    assert any(
        i["invite_id"] == invite_id and i["status"] == "revoked"
        for i in revoked.json()["invites"]
    )

    guest = _second_client(settings, "rev.guest@example.com", "Guest")
    resp = guest.post(
        "/api/projects/invites/accept",
        json={"invite_id": invite_id, "token": token},
    )
    assert resp.status_code == 400
    assert "no longer valid" in resp.json()["detail"].lower() or "not found" in resp.json()["detail"].lower()


def test_invite_token_single_use(settings, monkeypatch, tmp_path):
    owner = _auth_client(settings, monkeypatch, tmp_path, "once.owner@example.com", "Owner")
    project_id = owner.post("/api/projects/", json={"title": "Once"}).json()["project_id"]
    invite = owner.post(
        f"/api/projects/{project_id}/invites",
        json={"email": "once.guest@example.com", "role": "reviewer"},
    ).json()
    body = {"invite_id": invite["invite"]["invite_id"], "token": invite["token"]}

    guest = _second_client(settings, "once.guest@example.com", "Guest")
    first = guest.post("/api/projects/invites/accept", json=body)
    assert first.status_code == 200
    assert any(m["email"] == "once.guest@example.com" for m in first.json()["members"])

    again = guest.post("/api/projects/invites/accept", json=body)
    assert again.status_code == 400


def test_invalid_invite_role_rejected(settings, monkeypatch, tmp_path):
    owner = _auth_client(settings, monkeypatch, tmp_path, "role.owner@example.com", "Owner")
    project_id = owner.post("/api/projects/", json={"title": "Roles"}).json()["project_id"]
    bad = owner.post(
        f"/api/projects/{project_id}/invites",
        json={"email": "x@example.com", "role": "owner"},
    )
    assert bad.status_code in (400, 422)


def test_bad_token_rejected(settings, monkeypatch, tmp_path):
    owner = _auth_client(settings, monkeypatch, tmp_path, "tok.owner@example.com", "Owner")
    project_id = owner.post("/api/projects/", json={"title": "Token"}).json()["project_id"]
    invite = owner.post(
        f"/api/projects/{project_id}/invites",
        json={"email": "tok.guest@example.com", "role": "viewer"},
    ).json()
    guest = _second_client(settings, "tok.guest@example.com", "Guest")
    resp = guest.post(
        "/api/projects/invites/accept",
        json={"invite_id": invite["invite"]["invite_id"], "token": "not-the-real-token"},
    )
    assert resp.status_code == 400
    assert "token" in resp.json()["detail"].lower()
