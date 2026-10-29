"""Workspace recents tests."""

from __future__ import annotations

from fastapi.testclient import TestClient

from app.main import create_app
from app.models.query import AnswerStatus, PipelineName, PipelineResult
from app.models.workspace import AskChatTurn
from app.services.auth.sessions import clear_sessions
from app.services.auth.users import create_user
from app.services.store.history import history
from app.services.workspace import ask_chats, ask_queries, paper_searches


def _auth_client(settings, monkeypatch, tmp_path, email: str):
    monkeypatch.setenv("SELFRAG_AUTH_REQUIRED", "true")
    monkeypatch.setenv("SELFRAG_SUPABASE_URL", "")
    monkeypatch.setenv("SELFRAG_SUPABASE_SERVICE_KEY", "")
    from app.config import get_settings

    get_settings.cache_clear()
    settings.auth_required = True
    settings.supabase_url = ""
    settings.supabase_service_key = ""
    monkeypatch.setattr("app.config.settings", settings)
    monkeypatch.setattr("app.config.get_settings", lambda: settings)
    clear_sessions()
    monkeypatch.setattr(paper_searches, "_ROOT", tmp_path / "workspace")
    monkeypatch.setattr(ask_queries, "_ROOT", tmp_path / "workspace" / "ask")
    monkeypatch.setattr(ask_chats, "_ROOT", tmp_path / "workspace" / "chats")
    create_user(settings, email, "secret-pass", "Workspace")
    app = create_app()
    client = TestClient(app)
    login = client.post("/api/auth/login", json={"email": email, "password": "secret-pass"})
    assert login.status_code == 200
    return client


def test_workspace_paper_search_and_recents(settings, monkeypatch, tmp_path):
    client = _auth_client(settings, monkeypatch, tmp_path, "workspace.user@example.com")

    saved = client.post(
        "/api/workspace/paper-searches",
        json={
            "query": "self-rag verification",
            "provider": "semantic_scholar",
            "papers": [
                {
                    "paper_id": "p1",
                    "title": "Self-RAG",
                    "authors": ["A"],
                    "year": 2024,
                    "source": "semantic_scholar",
                    "url": "https://example.com/p1",
                }
            ],
        },
    )
    assert saved.status_code == 200
    body = saved.json()
    assert body["query"] == "self-rag verification"
    assert body["search_id"]
    assert len(body["papers"]) == 1

    detail = client.get(f"/api/workspace/paper-searches/{body['search_id']}")
    assert detail.status_code == 200
    assert detail.json()["papers"][0]["title"] == "Self-RAG"

    recents = client.get("/api/workspace/recents")
    assert recents.status_code == 200
    payload = recents.json()
    assert payload["total"] >= 1
    assert any(item["kind"] == "papers" and item["query"] == "self-rag verification" for item in payload["items"])


def test_ask_chat_thread_in_recents(settings, monkeypatch, tmp_path):
    client = _auth_client(settings, monkeypatch, tmp_path, "workspace.chat@example.com")

    first = client.post(
        "/api/workspace/ask-chats",
        json={
            "title": "What is RAG?",
            "turns": [
                {
                    "query": "What is RAG?",
                    "result": {
                        "query_id": "q1",
                        "pipeline": "enhanced",
                        "pipeline_label": "Enhanced",
                        "query": "What is RAG?",
                        "answer": "Retrieval Augmented Generation",
                        "status": "answered",
                    },
                },
                {
                    "query": "What is Self-RAG?",
                    "result": {
                        "query_id": "q2",
                        "pipeline": "enhanced",
                        "pipeline_label": "Enhanced",
                        "query": "What is Self-RAG?",
                        "answer": "Self-reflective RAG",
                        "status": "answered",
                    },
                },
            ],
        },
    )
    assert first.status_code == 200
    chat = first.json()
    assert chat["chat_id"]
    assert chat["title"] == "What is RAG?"
    assert len(chat["turns"]) == 2

    updated = client.post(
        "/api/workspace/ask-chats",
        json={
            "chat_id": chat["chat_id"],
            "title": "What is RAG?",
            "turns": chat["turns"]
            + [
                {
                    "query": "Give an example",
                    "result": {
                        "query_id": "q3",
                        "pipeline": "enhanced",
                        "pipeline_label": "Enhanced",
                        "query": "Give an example",
                        "answer": "Example",
                        "status": "answered",
                    },
                }
            ],
        },
    )
    assert updated.status_code == 200
    assert updated.json()["chat_id"] == chat["chat_id"]
    assert len(updated.json()["turns"]) == 3

    recents = client.get("/api/workspace/recents").json()["items"]
    ask_items = [item for item in recents if item["kind"] == "ask"]
    assert len(ask_items) == 1
    assert ask_items[0]["id"] == chat["chat_id"]
    assert ask_items[0]["query"] == "What is RAG?"
    assert ask_items[0]["meta"]["turn_count"] == 3

    loaded = client.get(f"/api/workspace/ask-chats/{chat['chat_id']}")
    assert loaded.status_code == 200
    assert len(loaded.json()["turns"]) == 3

    deleted = client.delete(f"/api/workspace/recents/ask/{chat['chat_id']}")
    assert deleted.status_code == 204
    assert client.get(f"/api/workspace/ask-chats/{chat['chat_id']}").status_code == 404


def test_workspace_delete_paper_and_ask_recents(settings, monkeypatch, tmp_path):
    client = _auth_client(settings, monkeypatch, tmp_path, "workspace.delete@example.com")

    saved = client.post(
        "/api/workspace/paper-searches",
        json={
            "query": "delete me papers",
            "provider": "openalex",
            "papers": [
                {
                    "paper_id": "p-del",
                    "title": "Delete Paper",
                    "authors": ["B"],
                    "year": 2023,
                    "source": "openalex",
                    "url": "https://example.com/p-del",
                }
            ],
        },
    )
    assert saved.status_code == 200
    search_id = saved.json()["search_id"]

    deleted = client.delete(f"/api/workspace/recents/papers/{search_id}")
    assert deleted.status_code == 204
    assert client.get(f"/api/workspace/paper-searches/{search_id}").status_code == 404
    assert client.delete(f"/api/workspace/recents/papers/{search_id}").status_code == 404

    ask_id = "ask-delete-test-id"
    result = PipelineResult(
        query_id=ask_id,
        pipeline=PipelineName.enhanced,
        pipeline_label="Enhanced",
        query="delete me ask",
        answer="gone",
        status=AnswerStatus.answered,
    )
    history.add(result)
    me = client.get("/api/auth/me")
    assert me.status_code == 200
    ask_queries.save_ask_query(me.json()["user_id"], result)
    ask_chats.save_chat(
        me.json()["user_id"],
        chat_id="chat-delete-test",
        title="delete me ask",
        turns=[AskChatTurn(query="delete me ask", result=result.model_dump(mode="json"))],
    )
    assert any(item["id"] == "chat-delete-test" for item in client.get("/api/workspace/recents").json()["items"])

    ask_deleted = client.delete("/api/workspace/recents/ask/chat-delete-test")
    assert ask_deleted.status_code == 204
    assert client.delete("/api/workspace/recents/ask/chat-delete-test").status_code == 404
    assert not any(
        item["kind"] == "ask" and item["id"] == "chat-delete-test"
        for item in client.get("/api/workspace/recents").json()["items"]
    )
    assert not any(
        item["kind"] == "papers" and item["id"] == search_id
        for item in client.get("/api/workspace/recents").json()["items"]
    )
