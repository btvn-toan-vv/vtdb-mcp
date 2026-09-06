"""The query tool's execution contract (services/pipeline.py via the tool).

Runs the real biocircle trace→seal→run spine against an empty
VtdbSymbolContext. Error payloads / sandbox rejections are part of the
contract, not accidents — deviating messages from the underlying libraries
update these tests.
"""

from fastmcp.client.client import CallToolResult


def test_query_simple_output(call_tool) -> None:
    # The Phase-0 sanity case from the design discussion.
    result: CallToolResult = call_tool("query", {"code": "output({'hello': 1 + 2.0})"})
    assert result.data == {"result": {"hello": 3.0}}


def test_query_missing_output_call(call_tool) -> None:
    # The scaffold's contract nudge: user code must finish with output(...).
    result = call_tool("query", {"code": "x = 40 + 2"})
    error = result.data["error"]
    assert error["type"] == "RuntimeError"
    assert "output() was not called" in error["message"]


def test_query_output_called_twice(call_tool) -> None:
    result = call_tool("query", {"code": "output(1)\noutput(2)"})
    error = result.data["error"]
    assert error["type"] == "RuntimeError"
    assert "exactly one" in error["message"]


def test_query_user_code_error(call_tool) -> None:
    result = call_tool("query", {"code": "1 / 0"})
    error = result.data["error"]
    assert error["type"] == "ZeroDivisionError"
    assert "traceback" not in error  # default: masked


def test_query_debug_includes_traceback(call_tool) -> None:
    result = call_tool("query", {"code": "1 / 0", "debug": True})
    error = result.data["error"]
    assert error["type"] == "ZeroDivisionError"
    assert "ZeroDivisionError" in error["traceback"]
    assert 'File "<user>", line 1' in error["traceback"]


def test_query_def_and_lambda_bodies(call_tool) -> None:
    code = "def f():\n    return 21 * 2\noutput({'v': f()})"
    assert call_tool("query", {"code": code}).data == {"result": {"v": 42}}


def test_query_sandbox_blocks_imports(call_tool) -> None:
    result = call_tool("query", {"code": "import numpy as np\noutput(1)"})
    error = result.data["error"]
    assert error["type"] == "SandboxError"
    assert "Imports are not allowed" in error["message"]
    assert error.get("lineno") == 1


def test_query_sandbox_blocks_class_definitions(call_tool) -> None:
    result = call_tool("query", {"code": "class C:\n    pass\noutput(C())"})
    error = result.data["error"]
    assert error["type"] == "SandboxError"
    assert "Class definitions are not allowed" in error["message"]


def test_query_undefined_symbol_lists_vocabulary(call_tool) -> None:
    result = call_tool("query", {"code": "output(search('cells', [0.0]))"})
    error = result.data["error"]
    assert error["type"] == "NameError"
    assert "'search' is not available in this pipeline" in error["message"]
    # The candidate list is the Phase-0 vocabulary: builtins + output()…
    assert "output" in error["message"]


def test_query_syntax_error(call_tool) -> None:
    result = call_tool("query", {"code": "output([1,"})
    assert result.data["error"]["type"] == "SyntaxError"


def test_query_non_finite_floats(call_tool) -> None:
    # The wire is strict JSON: NaN/Inf normalize to null on the client side.
    result = call_tool("query", {"code": "output({'nan': float('nan'), 'inf': 1e999})"})
    assert result.data == {"result": {"nan": None, "inf": None}}
