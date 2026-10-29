"""Require a session cookie for product API routes."""

from __future__ import annotations

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from ..config import get_settings
from ..services.auth.sessions import get_session
from .auth import COOKIE_NAME

_PUBLIC_EXACT = frozenset({"/api/health", "/api/config", "/openapi.json", "/docs", "/redoc"})


class AuthGateMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next) -> Response:
        settings = get_settings()
        if not settings.auth_required:
            return await call_next(request)
        if request.method == "OPTIONS":
            return await call_next(request)
        path = request.url.path
        if (
            path in _PUBLIC_EXACT
            or path.startswith("/api/auth")
            or path.startswith("/docs")
            or path.startswith("/redoc")
            or (request.method == "GET" and path.startswith("/api/stories/public"))
        ):
            return await call_next(request)
        if path.startswith("/api/"):
            token = request.cookies.get(COOKIE_NAME)
            if not token:
                auth_header = request.headers.get("Authorization") or ""
                if auth_header.startswith("Bearer "):
                    token = auth_header[7:].strip()
            user = get_session(token) if token else None
            if user is None:
                return JSONResponse({"detail": "Not signed in."}, status_code=401)
            request.state.user = user
        return await call_next(request)
