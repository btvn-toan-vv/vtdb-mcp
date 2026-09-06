"""App surface: /health over HTTP, tool surface via the conftest fixtures.

Tool-call patterns for new tests: see test_client_template.py.
"""

import asyncio

from fastmcp import Client, FastMCP
from server import __version__
from server.app import ServerConfig, create_app
from starlette.testclient import TestClient

_CONFIG = ServerConfig()


def test_health_endpoint() -> None:
    app = create_app(_CONFIG)
    with TestClient(app) as client:  # context manager runs the MCP lifespan
        resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["service"] == "vtdb-mcp"
    assert body["version"] == __version__
    assert body["mcp_path"] == "/mcp"


def test_tool_registry(mcp: FastMCP) -> None:
    async def run() -> set[str]:
        async with Client(mcp) as client:
            return {t.name for t in await client.list_tools()}

    assert asyncio.run(run()) == {"query", "server_info"}


def test_query_executes_with_output_sink(call_tool) -> None:
    # The tool wraps the result dict as structured output; the contract itself
    # (output() semantics, errors, serialization) is covered in test_query.py.
    assert call_tool("query", {"code": "output({'sum': 40 + 2})"}).data == {
        "result": {"sum": 42}
    }


def test_server_info(call_tool) -> None:
    info = call_tool("server_info", {}).data
    assert info["name"] == "vtdb-mcp"
    assert info["version"] == __version__
    assert info["uptime_seconds"] >= 0.0
