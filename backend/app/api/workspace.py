"""Workspace recents API."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query, Request

from .auth import current_user_from_request
from ..models.workspace import (
    AskChatRecord,
    AskChatSaveRequest,
    PaperSearchRecord,
    PaperSearchSaveRequest,
    WorkspaceRecentItem,
    WorkspaceRecentsResponse,
)
from ..services.store.history import history
from ..services.workspace import ask_chats, ask_queries, paper_searches

router = APIRouter(prefix="/workspace", tags=["workspace"])


def _require_user(request: Request) -> dict:
    user = current_user_from_request(request)
    if user is None or not user.get("user_id"):
        raise HTTPException(status_code=401, detail="Not signed in.")
    return user


@router.get("/recents", response_model=WorkspaceRecentsResponse)
def workspace_recents(
    request: Request,
    limit: int = Query(default=24, ge=1, le=50),
) -> WorkspaceRecentsResponse:
    user = _require_user(request)
    user_id = str(user["user_id"])
    ask_chats.migrate_legacy_queries(user_id)
    items: list[WorkspaceRecentItem] = []

    for chat in ask_chats.list_chats(user_id, limit=limit):
        items.append(
            WorkspaceRecentItem(
                kind="ask",
                id=chat.chat_id,
                query=chat.title or "New chat",
                created_at=chat.updated_at or chat.created_at,
                meta={"turn_count": len(chat.turns)},
            )
        )

    for record in paper_searches.list_paper_searches(user_id, limit=limit):
        items.append(
            WorkspaceRecentItem(
                kind="papers",
                id=record.search_id,
                query=record.query,
                created_at=record.created_at,
                provider=record.provider,
                paper_count=len(record.papers),
            )
        )

    items.sort(key=lambda item: item.created_at or "", reverse=True)
    merged = items[:limit]
    return WorkspaceRecentsResponse(items=merged, total=len(merged))


@router.post("/ask-chats", response_model=AskChatRecord)
def save_ask_chat(payload: AskChatSaveRequest, request: Request) -> AskChatRecord:
    user = _require_user(request)
    try:
        return ask_chats.save_chat(
            str(user["user_id"]),
            chat_id=payload.chat_id,
            title=payload.title,
            turns=payload.turns,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/ask-chats/{chat_id}", response_model=AskChatRecord)
def get_ask_chat(chat_id: str, request: Request) -> AskChatRecord:
    user = _require_user(request)
    record = ask_chats.get_chat(str(user["user_id"]), chat_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Chat not found.")
    return record


@router.post("/paper-searches", response_model=PaperSearchRecord)
def save_paper_search(payload: PaperSearchSaveRequest, request: Request) -> PaperSearchRecord:
    user = _require_user(request)
    try:
        return paper_searches.save_paper_search(
            str(user["user_id"]),
            query=payload.query,
            provider=payload.provider,
            papers=payload.papers,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/paper-searches/{search_id}", response_model=PaperSearchRecord)
def get_paper_search(search_id: str, request: Request) -> PaperSearchRecord:
    user = _require_user(request)
    record = paper_searches.get_paper_search(str(user["user_id"]), search_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Paper search not found.")
    return record


@router.delete("/recents/ask/{chat_id}", status_code=204)
def delete_ask_recent(chat_id: str, request: Request) -> None:
    user = _require_user(request)
    user_id = str(user["user_id"])
    cid = str(chat_id or "").strip()
    if not cid:
        raise HTTPException(status_code=404, detail="Ask recent not found.")

    record = ask_chats.get_chat(user_id, cid)
    query_ids: set[str] = {cid}
    if record is not None:
        for turn in record.turns:
            if isinstance(turn.result, dict):
                qid = str(turn.result.get("query_id") or "").strip()
                if qid:
                    query_ids.add(qid)

    chat_deleted = ask_chats.delete_chat(user_id, cid)
    mem_deleted = False
    query_deleted = False
    for qid in query_ids:
        if history.delete(qid):
            mem_deleted = True
        if ask_queries.delete_ask_query(user_id, qid):
            query_deleted = True

    if not chat_deleted and not mem_deleted and not query_deleted:
        raise HTTPException(status_code=404, detail="Ask recent not found.")


@router.delete("/recents/papers/{search_id}", status_code=204)
def delete_papers_recent(search_id: str, request: Request) -> None:
    user = _require_user(request)
    if not paper_searches.delete_paper_search(str(user["user_id"]), search_id):
        raise HTTPException(status_code=404, detail="Paper search not found.")
