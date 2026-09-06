"""App surface: /health over HTTP, tool surface via FastMCP's in-memory client."""

import asyncio

from fastmcp import Client
from starlette.testclient import TestClient

from server import __version__
from server.app import ServerConfig, create_app, create_mcp

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


def test_tool_registry() -> None:
    async def run() -> set[str]:
        async with Client(create_mcp(_CONFIG)) as client:
            return {t.name for t in await client.list_tools()}

    assert asyncio.run(run()) == {"ping", "echo", "add", "server_info"}


def test_ping_and_echo() -> None:
    async def run() -> tuple[str, str]:
        async with Client(create_mcp(_CONFIG)) as client:
            pong = (await client.call_tool("ping", {})).data
            echoed = (await client.call_tool("echo", {"text": "hello"})).data
            return pong, echoed

    assert asyncio.run(run()) == ("pong", "hello")


def test_add() -> None:
    async def run() -> float:
        async with Client(create_mcp(_CONFIG)) as client:
            return (await client.call_tool("add", {"a": 2.5, "b": 1.5})).data

    assert asyncio.run(run()) == 4.0


def test_server_info() -> None:
    async def run() -> dict:
        async with Client(create_mcp(_CONFIG)) as client:
            return (await client.call_tool("server_info", {})).data

    info = asyncio.run(run())
    assert info["name"] == "vtdb-mcp"
    assert info["version"] == __version__
    assert info["uptime_seconds"] >= 0.0
