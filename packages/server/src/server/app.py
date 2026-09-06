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
from server.core.context import VtdbSymbolContext

logger = logging.getLogger(__name__)


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _env_flag_default_on(name: str) -> bool:
    """Env flag defaulting to ON: only explicit 0/false/no/off disables."""
    raw = os.environ.get(name)
    if raw is None:
        return True
    return raw.strip().lower() not in {"0", "false", "no", "off"}


def _run_sync(coro: Any) -> None:
    """Run an async setup from sync build code, inside or outside a loop.

    App construction happens in _three_ contexts here: cli/boot (no loop),
    test clients (no loop), and uvicorn's reload child (loop already running).
    asyncio.run in the last one raises, so in that case run it in a one-shot
    thread with its own loop and join (setup is tiny and fully synchronous work).
    """
    import asyncio
    import threading

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        asyncio.run(coro)
        return
    holder: list[BaseException] = []

    def _target() -> None:
        try:
            asyncio.run(coro)
        except BaseException as exc:  # noqa: BLE001
            holder.append(exc)

    t = threading.Thread(target=_target, daemon=True, name="docs-setup")
    t.start()
    t.join()
    if holder:
        raise holder[0]


def _setup_docs(mcp: FastMCP, config: ServerConfig) -> None:
    """Swagger-style docs page at {mcp_path}/docs (fastmcp-docs), exen-style.

    The routes register via mcp.custom_route, so they join the served app when
    http_app() is built after this call (ordering matters).
    """
    from fastmcp_docs import FastMCPDocs, FastMCPDocsConfig

    host = "localhost" if config.host in ("0.0.0.0", "", "::") else config.host
    docs = FastMCPDocs(
        mcp,
        config=FastMCPDocsConfig(
            verbose=False,
            title="vtdb-mcp Tools",
            version=__version__,
            description=(
                "Sandboxed query DSL over the cells/images Qdrant collections"
                " — tool reference."
            ),
            docs_ui_route=f"{config.mcp_path}/docs",
            openapi_route=f"{config.mcp_path}/openapi.json",
            api_tools_route=f"{config.mcp_path}/api/tools",
            api_tool_detail_route=f"{config.mcp_path}/api/tools/{{tool_name}}",
            base_url=f"http://{host}:{config.port}",
        ),
    )
    _run_sync(docs.setup())
    logger.info("Docs UI at %s/docs", config.mcp_path)


@dataclass(frozen=True)
class ServerConfig:
    """Runtime configuration (from CLI flags / VTDB_* env vars)."""

    host: str = "127.0.0.1"
    port: int = 8000
    mcp_path: str = "/mcp"
    # Auto-reload the app process on source changes. Only meaningful in the dev
    # compose, which bind-mounts ./packages/server/src over /app/src.
    reload: bool = False
    # fastmcp-docs swagger-style page at /mcp/docs (off: VTDB_DOCS=0).
    enable_docs: bool = True

    @classmethod
    def from_env(cls) -> ServerConfig:
        """Build straight from VTDB_* env vars — used by the uvicorn factory
        in reload mode, where the respawned child never sees CLI flags."""
        return cls(
            host=os.environ.get("VTDB_HOST", "127.0.0.1"),
            port=int(os.environ.get("VTDB_PORT", "8000")),
            mcp_path=os.environ.get("VTDB_MCP_PATH", "/mcp"),
            reload=_env_flag("VTDB_RELOAD"),
            enable_docs=_env_flag_default_on("VTDB_DOCS"),
        )


def create_mcp(config: ServerConfig) -> FastMCP:
    """Build the FastMCP server: tools + the unauthenticated /health route.

    Split out of :func:`create_app` so tests can drive the tool surface via
    FastMCP's in-memory client without standing up HTTP.
    """
    mcp = FastMCP(
        name="vtdb-mcp",
        instructions=(
            "vtdb-mcp — MCP server over the subcellular-embeddings vector "
            "database. Tools: `query` (run code against the collections), "
            "`server_info`."
        ),
        version=__version__,
        # Return sanitized tool errors to clients; full tracebacks stay in the
        # server log (same posture as exen-mcp).
        mask_error_details=True,
    )
    # Registered on construction (Phase 1: filtering DSL — retrieval vocab).
    sym_ctx = VtdbSymbolContext()
    register_tools(mcp, sym_ctx)

    if config.enable_docs:
        _setup_docs(mcp, config)

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

    return mcp


def create_app(config: ServerConfig) -> Starlette:
    """Build the ASGI app: FastMCP streamable-http at ``config.mcp_path``."""
    app = create_mcp(config).http_app(path=config.mcp_path, transport="streamable-http")
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
