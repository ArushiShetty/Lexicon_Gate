"""Phase 5 — submission readiness checklist."""

from __future__ import annotations

from types import SimpleNamespace

from fastapi.testclient import TestClient

from app.config import get_settings
from app.main import create_app
from app.services.auth.sessions import clear_sessions
from app.services.projects import manuscript_checklist as checklist_mod
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
    monkeypatch.setattr(project_store, "_MANUSCRIPTS_ROOT", root / "manuscripts")


def _client():
    return TestClient(create_app())


def _signup(client: TestClient, email: str, name: str = "User"):
    resp = client.post(
        "/api/auth/signup",
        json={"email": email, "password": "secret-pass", "name": name},
    )
    assert resp.status_code == 200, resp.text
    return resp.json()


def _invite_and_accept(owner, invitee, project_id: str, role: str, rid: str):
    created = owner.post(
        f"/api/projects/{project_id}/invites",
        json={"researcher_id": rid, "role": role},
    )
    assert created.status_code == 200, created.text
    assert (
        invitee.post(
            "/api/projects/invites/accept",
            json={"invite_id": created.json()["invite"]["invite_id"]},
        ).status_code
        == 200
    )


def test_checklist_init_update_refresh_auth(settings, monkeypatch, tmp_path):
    _prepare(settings, monkeypatch, tmp_path)
    monkeypatch.setattr(
        "app.api.projects.get_knowledge_base",
        lambda: SimpleNamespace(store=SimpleNamespace(get_document=lambda *_: None)),
    )

    owner = _client()
    viewer = _client()
    reviewer = _client()
    _signup(owner, "chk.owner@example.com", "Owner")
    viewer_sess = _signup(viewer, "chk.viewer@example.com", "Viewer")
    reviewer_sess = _signup(reviewer, "chk.reviewer@example.com", "Reviewer")
    project_id = owner.post("/api/projects/", json={"title": "Checklist"}).json()["project_id"]
    _invite_and_accept(owner, viewer, project_id, "viewer", viewer_sess["researcher_id"])
    _invite_and_accept(
        owner, reviewer, project_id, "reviewer", reviewer_sess["researcher_id"]
    )

    ms = owner.post(
        f"/api/projects/{project_id}/manuscript",
        json={
            "template": "ieee_research",
            "document_type": "ieee_research",
            "title": "Ready?",
            "authors": ["A. Author"],
        },
    )
    assert ms.status_code == 200

    got = owner.get(f"/api/projects/{project_id}/manuscript/checklist")
    assert got.status_code == 200, got.text
    body = got.json()
    keys = [i["key"] for i in body["checklist"]["items"]]
    for required in (
        "title",
        "abstract",
        "keywords",
        "required_sections",
        "citations",
        "references",
        "unsupported_claims",
        "figures_tables",
        "author_info",
        "formatting_review",
        "final_export_review",
    ):
        assert required in keys
    assert "acceptance" not in body["summary"]["headline"].lower() or "not an acceptance" in body[
        "summary"
    ]["headline"].lower()
    assert "compliance" in body["checklist"]["disclaimer"].lower()
    assert body["summary"]["incomplete"] == body["summary"]["total"]

    # Empty abstract should warn
    assert any(
        i["key"] == "abstract" and i.get("warning") for i in body["checklist"]["items"]
    )

    # Viewer can read
    assert viewer.get(f"/api/projects/{project_id}/manuscript/checklist").status_code == 200
    # Viewer cannot update
    assert (
        viewer.patch(
            f"/api/projects/{project_id}/manuscript/checklist/abstract",
            json={"status": "completed"},
        ).status_code
        == 403
    )

    updated = reviewer.patch(
        f"/api/projects/{project_id}/manuscript/checklist/abstract",
        json={"status": "in_progress"},
    )
    assert updated.status_code == 200, updated.text
    abs_item = next(i for i in updated.json()["checklist"]["items"] if i["key"] == "abstract")
    assert abs_item["status"] == "in_progress"
    assert abs_item["updated_by_name"]
    assert abs_item["updated_at"]
    assert updated.json()["summary"]["in_progress"] >= 1

    done = owner.patch(
        f"/api/projects/{project_id}/manuscript/checklist/author_info",
        json={"status": "completed"},
    )
    assert done.status_code == 200
    auth = next(i for i in done.json()["checklist"]["items"] if i["key"] == "author_info")
    assert auth["status"] == "completed"

    refreshed = owner.post(f"/api/projects/{project_id}/manuscript/checklist/refresh")
    assert refreshed.status_code == 200
    # Status preserved after refresh
    auth2 = next(i for i in refreshed.json()["checklist"]["items"] if i["key"] == "author_info")
    assert auth2["status"] == "completed"


def test_literature_review_checklist_keys():
    cl = checklist_mod.build_default_checklist("literature_review")
    keys = {i.key for i in cl.items}
    assert "search_strategy" in keys
    assert "synthesis" in keys
    assert "title" in keys
    assert "final_export_review" in keys
