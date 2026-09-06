"""Tool registration — the MCP server's tool surface lives here.

Mirrors exen-mcp's ``api/tools.py`` pattern: a single ``register_tools(mcp,
sym_ctx)`` called once from :func:`server.app.create_mcp`.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

from fastmcp import FastMCP

from server import __version__
from server.api.descriptions import render_query_description
from server.core.context import VtdbSymbolContext
from server.services import pipeline as pipeline_svc

logger = logging.getLogger(__name__)

_STARTED_AT = time.monotonic()


def register_tools(mcp: FastMCP, sym_ctx: VtdbSymbolContext | None = None) -> None:
    """Register the tools on ``mcp``.

    Tool calls log one line each so the Grafana Alloy -> Loki -> Grafana
    pipeline has a visible stream to tail (open the "vtdb-mcp · Logs"
    dashboard and call ``query`` a few times).
    """
    if sym_ctx is None:
        sym_ctx = VtdbSymbolContext()  # Phase 0: empty registry

    @mcp.tool(
        name="query",
        tags={"data"},
        # Jinja template: api/templates/query.md.j2 (see descriptions.py) —
        # the exen-mcp pattern; edit the markdown, not the Python.
        description=render_query_description(),
    )
    async def query(code: str, debug: bool = False) -> dict[str, Any]:
        # Worker thread like exen's pipeline_query (asyncio.to_thread): a slow
        # query never blocks the event loop.
        logger.info("tool query invoked (len=%d, debug=%s)", len(code), debug)
        return await asyncio.to_thread(pipeline_svc.execute_query, code, sym_ctx, debug)

    @mcp.tool(name="server_info", tags={"meta"})
    def server_info() -> dict[str, str | float]:
        """Return the server name, version, and uptime in seconds."""
        logger.info("tool server_info invoked")
        return {
            "name": "vtdb-mcp",
            "version": __version__,
            "uptime_seconds": round(time.monotonic() - _STARTED_AT, 3),
        }
