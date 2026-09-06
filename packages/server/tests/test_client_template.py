"""TEMPLATE — how to test MCP tools with the in-memory FastMCP client.

Copy this file to ``test_<feature>.py``, rename, and replace the scenarios.
It is collected and executed like any other test module, so the examples below
are guaranteed to keep working (if one starts failing, the *helpers* broke,
not your tool).

Fixtures used (defined in conftest.py):
    mcp         fresh FastMCP server instance (no HTTP, no network)
    call_tool   sync bridge returning fastmcp.client.CallToolResult

What CallToolResult gives you:
    .data                  parsed structured result ("pong", 42.0, {...})
    .content               list[TextContent|...] — the MCP wire content
    .structured_content    raw structured dict
    .is_error              bool
"""

import pytest
from fastmcp.exceptions import ToolError


def test_tool_happy_path(call_tool):
    """Call a tool and assert on the parsed result."""
    result = call_tool("query", {"code": "output({'hello': 1 + 2.0})"})
    assert result.is_error is False
    assert result.data == {"result": {"hello": 3.0}}


def test_tool_result_surface(call_tool):
    result = call_tool("query", {"code": "output('cells!')"})
    assert result.data == {"result": "cells!"}
    # The full MCP content is there when you need the wire-level view:
    assert result.content[0].text


def test_tool_schema_error(call_tool):
    """Wrong/omitted arguments raise ToolError (mask_error_details=True means
    the message is a sanitized generic one — assert the failure, not text)."""
    with pytest.raises(ToolError):
        call_tool("query", {"code": 42})


def test_tool_metadata(mcp):
    """Inspect the registered surface: names, descriptions, input schemas."""
    import asyncio

    from fastmcp import Client

    async def _run():
        async with Client(mcp) as client:
            return {t.name: t for t in await client.list_tools()}

    tools = asyncio.run(_run())
    assert set(tools) == {"query", "server_info"}
    # MCP-side Tool objects: name / description / input_schema / annotations.
    # (input_schema was camelCase inputSchema pre MCP SDK v2; the alias only
    # exists as a deprecated shim — use the snake_case field.)
    assert tools["query"].description
    assert set(tools["query"].input_schema["properties"]) == {"code", "debug"}
    assert tools["query"].input_schema["properties"]["code"]["type"] == "string"


# ── your new tests go below this line ──────────────────────────────────────
# def test_my_tool(call_tool):
#     result = call_tool("my_tool", {"arg": "value"})
#     assert result.data == ...
