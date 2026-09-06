"""Query execution service — the heart of the ``query`` tool.

Mirror of exen-mcp's ``pipeline_query`` spine (services/pipeline.py), same
biocircle machinery:

1. **trace** the code in a sandboxed subprocess against mock stubs generated
   from the SymbolContext (biocircle ``Tracer``, PERMISSIVE mode);
2. **seal** the trace into an ``ExecutorSession`` plan;
3. **run** the plan;
4. **unwrap** the recorded result (Variable tree → real values).

Phase-0 simplifications, all deliberate:

- **stateless**: one fresh ``ExecutorSession(dag=False)`` per query — no
  cross-query persistence/caching (exen's SessionManager/DAG store lands when
  there is something worth caching);
- **single shared tracer subprocess** under a lock (exen runs a TracerPool for
  interleaved queries; one lock is correct if not parallel);
- **one ``output()`` payload only**; exactly-once is enforced by biocircle's
  scaffold and surfaces here as a regular error result.
"""

from __future__ import annotations

import atexit
import logging
import re
import threading
import traceback
from typing import Any

from biocircle.executor import unwrap_result
from biocircle.session import ExecutorSession
from biocircle.tracer.tracer import TraceMode, Tracer

from server.core.context import VtdbSymbolContext

logger = logging.getLogger(__name__)

_TRACE_TIMEOUT_S = 30.0

_tracer: Tracer | None = None
_tracer_lock = threading.Lock()
_tracer_init_lock = threading.Lock()


def _get_tracer() -> Tracer:
    """Process-wide tracer (persistent sandbox subprocess, created lazily).

    exen uses a TracerPool to trace interleaved queries concurrently; Phase 0
    serializes with ``_tracer_lock`` instead.
    """
    global _tracer
    with _tracer_init_lock:
        if _tracer is None:
            _tracer = Tracer(timeout=_TRACE_TIMEOUT_S)
            atexit.register(_tracer.close)
    return _tracer


def _error_result(
    exc_type: str, message: str, debug: bool, extra: dict[str, Any] | None = None
) -> dict[str, Any]:
    error: dict[str, Any] = {"type": exc_type, "message": message}
    if extra:
        error.update(extra)
    if debug:
        tb = traceback.format_exc()
        if tb.strip() != "NoneType: None":  # only when an exception is live
            error["traceback"] = tb
    return {"result": None, "error": error}


def execute_query(
    code: str,
    sym_ctx: VtdbSymbolContext,
    debug: bool = False,
    timeout: float | None = 30.0,
) -> dict[str, Any]:
    """Trace→seal→run ``code`` against ``sym_ctx``; return the payload dict.

    ``{"result": <output() payload>}`` on success (``None`` when the query
    never called ``output()``); ``{"result": None, "error": {...}}`` on
    failure. This dict is the user-facing surface — the MCP layer masks
    internals — so tracebacks only ride along with ``debug=True``.
    """
    try:
        with _tracer_lock:
            record = _get_tracer().trace(code, sym_ctx, mode=TraceMode.PERMISSIVE)
    except Exception as exc:  # noqa: BLE001 — the result dict IS the error surface
        logger.info("query trace failed: %s: %s", type(exc).__name__, exc)
        return _error_result(type(exc).__name__, str(exc), debug)

    if not record.ok:
        # Trace-phase failure (scaffold carried the subprocess traceback with
        # <user> frames): lineno always; traceback only when debugging.
        # Loose dict copy: ErrorInfo is a closed TypedDict and optional keys
        # (lineno) do exist at runtime on sandbox failures.
        err: dict[str, Any] = dict(record.error or {})
        extra: dict[str, Any] = {}
        if err.get("lineno"):
            extra["lineno"] = err["lineno"]
        if debug and err.get("traceback"):
            extra["traceback"] = err["traceback"]
        return _error_result(
            err.get("type", "Error") or "Error", err.get("message", ""), False, extra
        )

    session = ExecutorSession(dag=False)
    try:
        prepared = session.seal(record, sym_ctx)
        store = session.run_prepared(prepared, sym_ctx, timeout=timeout)
        value = unwrap_result(record.result, store)
    except Exception as exc:  # noqa: BLE001
        logger.info("query execution failed: %s: %s", type(exc).__name__, exc)
        # biocircle wraps inner errors as "InnerType: message (func=...) (line N)"
        # — peel the inner error type so DSL errors (UnknownFieldError etc.)
        # surface as first-class error types.
        err_type, message = type(exc).__name__, str(exc)
        m = re.match(r"^([A-Za-z_]\w*Error): (.*)$", message, re.DOTALL)
        if err_type == "ExecutionError" and m:
            err_type, message = m.group(1), m.group(2)
        return _error_result(err_type, message, debug)

    return {"result": value}
