"""ASGI app assembly — exen-mcp's app.py stripped to the core.

One FastMCP server over streamable-http at ``/mcp``, plus an unauthenticated
operational ``/health`` route (load balancers + the compose healthcheck),
served by a single uvicorn instance.
"""

from __future__ import annotations

import logging
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


@dataclass(frozen=True)
class ServerConfig:
    """Runtime configuration (from CLI flags / VTDB_* env vars)."""

    host: str = "127.0.0.1"
    port: int = 8000
    mcp_path: str = "/mcp"


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


def run(config: ServerConfig) -> None:
    """Serve ``create_app(config)`` with uvicorn (single worker)."""
    app = create_app(config)
    uvicorn.run(
        app,
        host=config.host,
        port=config.port,
        proxy_headers=True,
        forwarded_allow_ips="*",
        # Skip uvicorn's built-in logging config so its loggers propagate to
        # the root handler set up in server.__main__._configure_logging(),
        # unifying styles (same pattern as exen-mcp's run()).
        log_config=None,
    )
