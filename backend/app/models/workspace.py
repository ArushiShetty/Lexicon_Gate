"""Workspace recents models."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from .papers import PaperHit


class PaperSearchSaveRequest(BaseModel):
    query: str
    provider: str = ""
    papers: list[PaperHit] = Field(default_factory=list)


class PaperSearchRecord(BaseModel):
    search_id: str
    user_id: str
    query: str
    provider: str = ""
    papers: list[PaperHit] = Field(default_factory=list)
    created_at: str


class AskChatTurn(BaseModel):
    query: str
    result: dict[str, Any] | None = None
    error: str = ""


class AskChatSaveRequest(BaseModel):
    chat_id: str | None = None
    title: str = ""
    turns: list[AskChatTurn] = Field(default_factory=list)


class AskChatRecord(BaseModel):
    chat_id: str
    user_id: str
    title: str
    created_at: str
    updated_at: str
    turns: list[AskChatTurn] = Field(default_factory=list)


class WorkspaceRecentItem(BaseModel):
    kind: Literal["ask", "papers"]
    id: str
    query: str
    created_at: str | None = None
    status: str | None = None
    confidence: float | None = None
    provider: str | None = None
    paper_count: int | None = None
    meta: dict[str, Any] = Field(default_factory=dict)


class WorkspaceRecentsResponse(BaseModel):
    items: list[WorkspaceRecentItem] = Field(default_factory=list)
    total: int = 0
