"""Write / Stories publish and public read tests."""

from __future__ import annotations

from fastapi.testclient import TestClient

from app.main import create_app
from app.services.auth.sessions import clear_sessions
from app.services.auth.users import create_user
from app.services.stories import store as story_store


def _auth_client(settings, monkeypatch, tmp_path, email: str, name: str = "Author"):
    monkeypatch.setenv("SELFRAG_AUTH_REQUIRED", "true")
    from app.config import get_settings

    get_settings.cache_clear()
    clear_sessions()
    root = tmp_path / "stories"
    monkeypatch.setattr(story_store, "_ROOT", root)
    create_user(settings, email, "secret-pass", name)
    app = create_app()
    client = TestClient(app)
    login = client.post("/api/auth/login", json={"email": email, "password": "secret-pass"})
    assert login.status_code == 200
    return client


def _second_client(settings, email: str, name: str = "Other"):
    clear_sessions()
    try:
        create_user(settings, email, "secret-pass", name)
    except ValueError:
        pass
    client = TestClient(create_app())
    login = client.post("/api/auth/login", json={"email": email, "password": "secret-pass"})
    assert login.status_code == 200
    return client


def test_create_publish_public_unpublish(settings, monkeypatch, tmp_path):
    client = _auth_client(settings, monkeypatch, tmp_path, "writer@example.com", "Writer")

    created = client.post(
        "/api/stories",
        json={
            "title": "What is an LLM?",
            "format": "ieee",
            "authors_line": "W. Writer",
            "affiliation": "Lexicon Lab",
            "sections": {
                "abstract": "An LLM is a large language model.",
                "introduction": "We study LLMs.",
            },
        },
    )
    assert created.status_code == 200, created.text
    story = created.json()
    assert story["status"] == "draft"
    assert story["format"] == "ieee"
    assert story["slug"] == "what-is-an-llm"
    assert "large language model" in story["body_md"]
    story_id = story["story_id"]

    mine = client.get("/api/stories/mine")
    assert mine.status_code == 200
    assert mine.json()["total"] == 1

    # Draft not on public feed
    anon = TestClient(create_app())
    feed = anon.get("/api/stories/public")
    assert feed.status_code == 200
    assert feed.json()["total"] == 0

    published = client.post(f"/api/stories/{story_id}/publish")
    assert published.status_code == 200
    assert published.json()["status"] == "published"
    assert published.json()["published_at"]

    feed = anon.get("/api/stories/public")
    assert feed.json()["total"] == 1
    assert feed.json()["stories"][0]["title"] == "What is an LLM?"

    by_slug = anon.get("/api/stories/public/what-is-an-llm")
    assert by_slug.status_code == 200
    assert "large language model" in by_slug.json()["body_md"]

    docx = anon.get("/api/stories/public/what-is-an-llm/docx")
    assert docx.status_code == 200
    assert docx.headers["content-type"].startswith(
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    )
    assert docx.content[:2] == b"PK"

    owner_docx = client.get(f"/api/stories/{story_id}/docx")
    assert owner_docx.status_code == 200
    assert owner_docx.content[:2] == b"PK"

    client.post(f"/api/stories/{story_id}/unpublish")
    assert anon.get("/api/stories/public").json()["total"] == 0
    assert anon.get("/api/stories/public/what-is-an-llm").status_code == 404


def test_patch_and_slug_uniqueness(settings, monkeypatch, tmp_path):
    client = _auth_client(settings, monkeypatch, tmp_path, "a@example.com")
    first = client.post(
        "/api/stories",
        json={"title": "Same Title", "sections": {"abstract": "One"}},
    ).json()
    second = client.post(
        "/api/stories",
        json={"title": "Same Title", "sections": {"abstract": "Two"}},
    ).json()
    assert first["slug"] == "same-title"
    assert second["slug"] == "same-title-2"

    patched = client.patch(
        f"/api/stories/{first['story_id']}",
        json={"title": "Updated", "sections": {"abstract": "New abstract", "introduction": "Hi"}},
    )
    assert patched.status_code == 200
    assert patched.json()["title"] == "Updated"
    assert patched.json()["slug"] == "updated"
    assert "New abstract" in patched.json()["body_md"]


def test_non_owner_cannot_edit(settings, monkeypatch, tmp_path):
    owner = _auth_client(settings, monkeypatch, tmp_path, "owner@example.com", "Owner")
    story_id = owner.post(
        "/api/stories",
        json={"title": "Private draft", "sections": {"abstract": "Secret notes"}},
    ).json()["story_id"]

    other = _second_client(settings, "other@example.com", "Other")
    assert other.get(f"/api/stories/{story_id}").status_code == 403
    assert other.patch(f"/api/stories/{story_id}", json={"title": "Hack"}).status_code == 403
    assert other.post(f"/api/stories/{story_id}/publish").status_code == 403
    assert other.delete(f"/api/stories/{story_id}").status_code == 403
    assert other.get(f"/api/stories/{story_id}/docx").status_code == 403


def test_mine_requires_auth(settings, monkeypatch, tmp_path):
    monkeypatch.setenv("SELFRAG_AUTH_REQUIRED", "true")
    from app.config import get_settings

    get_settings.cache_clear()
    monkeypatch.setattr(story_store, "_ROOT", tmp_path / "stories")
    clear_sessions()
    anon = TestClient(create_app())
    assert anon.get("/api/stories/mine").status_code == 401
    assert anon.get("/api/stories/public").status_code == 200


def test_delete_story(settings, monkeypatch, tmp_path):
    client = _auth_client(settings, monkeypatch, tmp_path, "del@example.com")
    story_id = client.post(
        "/api/stories",
        json={"title": "Temp", "sections": {"abstract": "Gone soon"}},
    ).json()["story_id"]
    assert client.delete(f"/api/stories/{story_id}").status_code == 204
    assert client.get(f"/api/stories/{story_id}").status_code == 404


def _login_client(settings, email: str, name: str = "User") -> TestClient:
    try:
        create_user(settings, email, "secret-pass", name)
    except ValueError:
        pass
    client = TestClient(create_app())
    login = client.post("/api/auth/login", json={"email": email, "password": "secret-pass"})
    assert login.status_code == 200, login.text
    return client


def test_story_collaborators_invite_edit_and_viewer(settings, monkeypatch, tmp_path):
    owner = _auth_client(settings, monkeypatch, tmp_path, "paper.owner@example.com", "Owner")
    guest = _login_client(settings, "paper.guest@example.com", "Guest")
    viewer = _login_client(settings, "paper.viewer@example.com", "Viewer")

    guest_rid = guest.get("/api/auth/me").json()["researcher_id"]
    viewer_rid = viewer.get("/api/auth/me").json()["researcher_id"]

    story = owner.post(
        "/api/stories",
        json={
            "title": "Team Paper",
            "sections": {"abstract": "Owner wrote the abstract."},
        },
    ).json()
    story_id = story["story_id"]

    invite = owner.post(
        f"/api/stories/{story_id}/invites",
        json={"researcher_id": guest_rid, "role": "editor"},
    )
    assert invite.status_code == 200, invite.text
    invite_id = invite.json()["invite_id"]

    pending = guest.get("/api/stories/invites/pending")
    assert pending.status_code == 200
    assert any(i["invite_id"] == invite_id for i in pending.json()["invites"])

    # Owner should not see their own invite in pending inbox.
    owner_pending = owner.get("/api/stories/invites/pending").json()
    assert not any(i["invite_id"] == invite_id for i in owner_pending.get("invites") or [])

    accepted = guest.post(f"/api/stories/invites/{invite_id}/accept")
    assert accepted.status_code == 200, accepted.text
    assert any(c["user_id"] for c in accepted.json()["collaborators"])

    shared = guest.get("/api/stories/shared")
    assert shared.status_code == 200
    assert any(s["story_id"] == story_id and s["my_role"] == "editor" for s in shared.json()["stories"])

    # Editor can read and edit sections, but not publish/delete.
    assert guest.get(f"/api/stories/{story_id}").status_code == 200
    patched = guest.patch(
        f"/api/stories/{story_id}",
        json={"sections": {"abstract": "Owner wrote the abstract.", "introduction": "Guest intro."}},
    )
    assert patched.status_code == 200, patched.text
    assert "Guest intro" in patched.json()["body_md"]
    assert guest.post(f"/api/stories/{story_id}/publish").status_code == 403
    assert guest.delete(f"/api/stories/{story_id}").status_code == 403
    assert guest.get(f"/api/stories/{story_id}/docx").status_code == 200

    # Viewer invite + accept: can read, cannot edit.
    v_invite = owner.post(
        f"/api/stories/{story_id}/invites",
        json={"researcher_id": viewer_rid, "role": "viewer"},
    )
    assert v_invite.status_code == 200
    v_id = v_invite.json()["invite_id"]
    assert viewer.post(f"/api/stories/invites/{v_id}/accept").status_code == 200
    assert viewer.get(f"/api/stories/{story_id}").status_code == 200
    viewed = viewer.get(f"/api/stories/{story_id}").json()
    assert viewed["my_role"] == "viewer"
    assert (
        viewer.patch(
            f"/api/stories/{story_id}",
            json={"sections": {"abstract": "Hacked"}},
        ).status_code
        == 403
    )
    assert viewer.post(f"/api/stories/{story_id}/publish").status_code == 403
    assert viewer.delete(f"/api/stories/{story_id}").status_code == 403
    # Editor role receives my_role editor and can patch.
    guest_view = guest.get(f"/api/stories/{story_id}").json()
    assert guest_view["my_role"] == "editor"

    # Decline path
    stranger = _login_client(settings, "paper.stranger@example.com", "Stranger")
    stranger_rid = stranger.get("/api/auth/me").json()["researcher_id"]
    d_invite = owner.post(
        f"/api/stories/{story_id}/invites",
        json={"researcher_id": stranger_rid, "role": "editor"},
    ).json()
    assert stranger.post(f"/api/stories/invites/{d_invite['invite_id']}/decline").status_code == 204
    assert stranger.get("/api/stories/shared").json()["total"] == 0
    assert stranger.post(f"/api/stories/invites/{d_invite['invite_id']}/accept").status_code == 400


def test_owner_can_close_and_reopen_work(settings, monkeypatch, tmp_path):
    owner = _auth_client(settings, monkeypatch, tmp_path, "close.owner@example.com", "Owner")
    guest = _login_client(settings, "close.guest@example.com", "Lex")
    story = owner.post("/api/stories", json={"title": "Closeable paper"}).json()
    story_id = story["story_id"]
    rid = guest.get("/api/auth/me").json()["researcher_id"]
    invite = owner.post(
        f"/api/stories/{story_id}/invites",
        json={"researcher_id": rid, "role": "editor"},
    )
    assert invite.status_code == 200, invite.text
    assert guest.post(f"/api/stories/invites/{invite.json()['invite_id']}/accept").status_code == 200

    closed = guest.post(f"/api/stories/{story_id}/work-status", json={"status": "completed"})
    assert closed.status_code == 200, closed.text
    assert closed.json()["work_status"] == "completed"
    assert closed.json()["work_closed_by_name"] == "Lex"
    owner_view = owner.get(f"/api/stories/{story_id}")
    assert owner_view.status_code == 200
    assert owner_view.json()["work_closed_by_name"] == "Lex"
    assert guest.patch(
        f"/api/stories/{story_id}",
        json={"sections": {"abstract": "Too late"}},
    ).status_code == 403

    reopened = owner.post(f"/api/stories/{story_id}/work-status", json={"status": "open"})
    assert reopened.status_code == 200
    assert reopened.json()["work_status"] == "open"
    history = reopened.json()["invites"]
    assert any(item["status"] == "accepted" and item["recipient_name"] == "Lex" for item in history)

    # Close again so both sides can delete the shared history lines.
    closed_again = guest.post(f"/api/stories/{story_id}/work-status", json={"status": "completed"})
    assert closed_again.status_code == 200
    cleared = owner.delete(f"/api/stories/{story_id}/history/close")
    assert cleared.status_code == 200, cleared.text
    assert cleared.json()["work_status"] == "completed"
    assert cleared.json()["work_closed_by_name"] == ""
    guest_view = guest.get(f"/api/stories/{story_id}")
    assert guest_view.json()["work_closed_by_name"] == ""
    invite_id = history[0]["invite_id"]
    removed = guest.delete(f"/api/stories/{story_id}/history/invites/{invite_id}")
    assert removed.status_code == 200, removed.text
    assert removed.json()["invites"] == []
    assert owner.get(f"/api/stories/{story_id}").json()["invites"] == []


def test_owner_can_invite_by_display_name(settings, monkeypatch, tmp_path):
    owner = _auth_client(settings, monkeypatch, tmp_path, "name.owner@example.com", "Owner")
    guest = _login_client(settings, "name.guest@example.com", "Lex")
    story = owner.post("/api/stories", json={"title": "Named invite"}).json()
    invite = owner.post(
        f"/api/stories/{story['story_id']}/invites",
        json={"researcher_id": "lex", "role": "editor"},
    )
    assert invite.status_code == 200, invite.text
    pending = guest.get("/api/stories/invites/pending")
    assert pending.status_code == 200
    assert any(i["story_id"] == story["story_id"] for i in pending.json()["invites"])
