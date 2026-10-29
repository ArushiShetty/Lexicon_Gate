"""FastAPI application factory."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .api.auth import router as auth_router
from .api.auth_gate import AuthGateMiddleware
from .api.events import router as events_router
from .api.projects import router as projects_router
from .api.routes import router
from .api.stories import router as stories_router
from .api.workspace import router as workspace_router
from .config import get_settings
from .services.store.knowledge_base import get_knowledge_base

logger = logging.getLogger("selfrag")


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    configure_logging(settings.log_level)
    kb = get_knowledge_base()
    stats = kb.stats()
    logger.info("Knowledge base ready: %s", stats)
    yield


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(
        title=settings.app_name,
        description=(
            "Document Q&A: upload sources, ask a question, and receive one answer "
            "only when retrieved evidence supports each claim."
        ),
        version="1.0.0",
        lifespan=lifespan,
    )
    # AuthGate is registered first so CORS (added last) still wraps 401 responses.
    app.add_middleware(AuthGateMiddleware)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(auth_router, prefix="/api")
    app.include_router(events_router, prefix="/api")
    app.include_router(workspace_router, prefix="/api")
    app.include_router(projects_router, prefix="/api")
    app.include_router(stories_router, prefix="/api")
    app.include_router(router, prefix="/api")
    return app


app = create_app()
