"""Tool registration — the MCP server's tool surface lives here.

Mirrors exen-mcp's ``api/tools.py`` pattern: a single ``register_tools(mcp,
...)`` called once from :func:`server.app.create_app`.
"""

from __future__ import annotations

import logging
import time

from fastmcp import FastMCP

from server import __version__

logger = logging.getLogger(__name__)

_STARTED_AT = time.monotonic()


def register_tools(mcp: FastMCP) -> None:
    """Register the demo tools on ``mcp``.

    Every tool logs one line per invocation so the Grafana Alloy -> Loki ->
    Grafana pipeline has a visible stream to tail (try calling ``ping`` a few
    times, then open the "vtdb-mcp · Logs" dashboard).
    """

    @mcp.tool(name="ping", tags={"meta"})
    def ping() -> str:
        """Liveness probe for MCP clients. Always answers "pong"."""
        logger.info("tool ping invoked")
        return "pong"

    @mcp.tool(name="echo", tags={"demo"})
    def echo(text: str) -> str:
        """Echo ``text`` back unchanged — the smallest possible round-trip."""
        logger.info("tool echo invoked (len=%d)", len(text))
        return text

    @mcp.tool(name="add", tags={"demo"})
    def add(a: float, b: float) -> float:
        """Add two numbers; returns the sum."""
        logger.info("tool add invoked (a=%s, b=%s)", a, b)
        return a + b

    @mcp.tool(name="server_info", tags={"meta"})
    def server_info() -> dict[str, str | float]:
        """Return the server name, version, and uptime in seconds."""
        logger.info("tool server_info invoked")
        return {
            "name": "vtdb-mcp",
            "version": __version__,
            "uptime_seconds": round(time.monotonic() - _STARTED_AT, 3),
        }
