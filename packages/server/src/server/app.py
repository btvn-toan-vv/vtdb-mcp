"""ASGI app assembly — exen-mcp's app.py stripped to the core.

One FastMCP server over streamable-http at ``/mcp``, plus an unauthenticated
operational ``/health`` route (load balancers + the compose healthcheck),
served by a single uvicorn instance.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any

import uvicorn
from fastmcp import FastMCP
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from server import __version__
from server.api.tools import register_tools

logger = logging.getLogger(__name__)


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class ServerConfig:
    """Runtime configuration (from CLI flags / VTDB_* env vars)."""

    host: str = "127.0.0.1"
    port: int = 8000
    mcp_path: str = "/mcp"
    # Auto-reload the app process on source changes. Only meaningful in the dev
    # compose, which bind-mounts ./packages/server/src over /app/src.
    reload: bool = False

    @classmethod
    def from_env(cls) -> ServerConfig:
        """Build straight from VTDB_* env vars — used by the uvicorn factory
        in reload mode, where the respawned child never sees CLI flags."""
        return cls(
            host=os.environ.get("VTDB_HOST", "127.0.0.1"),
            port=int(os.environ.get("VTDB_PORT", "8000")),
            mcp_path=os.environ.get("VTDB_MCP_PATH", "/mcp"),
            reload=_env_flag("VTDB_RELOAD"),
        )


def create_app(config: ServerConfig) -> Starlette:
    """Build the ASGI app: FastMCP streamable-http at ``config.mcp_path``."""
    mcp = FastMCP(
        name="vtdb-mcp",
        instructions=(
            "vtdb-mcp — a minimal demo MCP server. "
            "Tools: `ping`, `echo`, `add`, `server_info`."
        ),
        version=__version__,
        # Return sanitized tool errors to clients; full tracebacks stay in the
        # server log (same posture as exen-mcp).
        mask_error_details=True,
    )
    register_tools(mcp)

    @mcp.custom_route("/health", methods=["GET"], include_in_schema=False)
    async def get_health(request: Request) -> Response:
        # Unauthenticated operational endpoint (same pattern as exen-mcp's
        # /health): no auth, answers 200 whenever the app is serving.
        payload: dict[str, Any] = {
            "status": "ok",
            "service": "vtdb-mcp",
            "version": __version__,
            "mcp_path": config.mcp_path,
        }
        return JSONResponse(payload)

    app = mcp.http_app(path=config.mcp_path, transport="streamable-http")
    logger.info(
        "MCP server mounted at %s (transport=streamable-http); health at /health",
        config.mcp_path,
    )
    return app


def app_from_env() -> Starlette:
    """Uvicorn ``--reload`` factory.

    The reloader respawns the app in a fresh child process (spawn context), so
    the import string is re-imported and CLI flags / the parent process's
    logging setup are lost. Rebuild both from the environment.
    """
    from server.log import configure_logging

    configure_logging(debug=_env_flag("VTDB_DEBUG"))
    return create_app(ServerConfig.from_env())


def run(config: ServerConfig) -> None:
    """Serve the app with uvicorn (single worker)."""
    if config.reload:
        # uvicorn's reloader re-imports the app in a child process, so it needs
        # an import string + factory; config is re-read from VTDB_* env vars by
        # app_from_env (the dev compose passes config via env, not flags).
        logger.info("Auto-reload enabled — source edits restart the app process")
        uvicorn.run(
            "server.app:app_from_env",
            factory=True,
            host=config.host,
            port=config.port,
            proxy_headers=True,
            forwarded_allow_ips="*",
            reload=True,
            # Skip uvicorn's built-in logging config so its loggers propagate
            # to the root handler set up by server.log.configure_logging().
            log_config=None,
        )
        return

    app = create_app(config)
    uvicorn.run(
        app,
        host=config.host,
        port=config.port,
        proxy_headers=True,
        forwarded_allow_ips="*",
        log_config=None,
    )
