"""Tool registration — the MCP server's tool surface lives here.

Mirrors exen-mcp's ``api/tools.py`` pattern: a single ``register_tools(mcp,
...)`` called once from :func:`server.app.create_mcp`.
"""

from __future__ import annotations

import logging
import time

from fastmcp import FastMCP

from server import __version__

logger = logging.getLogger(__name__)

_STARTED_AT = time.monotonic()


def register_tools(mcp: FastMCP) -> None:
    """Register the tools on ``mcp``.

    Tool calls log one line each so the Grafana Alloy -> Loki -> Grafana
    pipeline has a visible stream to tail (open the "vtdb-mcp · Logs"
    dashboard and call ``query`` a few times).
    """

    @mcp.tool(name="query", tags={"data"})
    def query(code: str) -> str:
        """Run ``code`` against the vector database.

        Placeholder for the real executor (see the Qdrant ingest in
        server.ingest): for now it just echoes ``code`` back unchanged.
        """
        logger.info("tool query invoked (len=%d)", len(code))
        return code

    @mcp.tool(name="server_info", tags={"meta"})
    def server_info() -> dict[str, str | float]:
        """Return the server name, version, and uptime in seconds."""
        logger.info("tool server_info invoked")
        return {
            "name": "vtdb-mcp",
            "version": __version__,
            "uptime_seconds": round(time.monotonic() - _STARTED_AT, 3),
        }
