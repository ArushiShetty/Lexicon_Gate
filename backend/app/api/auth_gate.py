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
            user = get_session(request.cookies.get(COOKIE_NAME))
            if user is None:
                return JSONResponse({"detail": "Not signed in."}, status_code=401)
            request.state.user = user
        return await call_next(request)
