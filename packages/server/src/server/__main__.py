"""vtdb-mcp server — a minimal FastMCP server.

Usage::

    python -m server [--host 0.0.0.0] [--port 8000] [--mcp-path /mcp] [--debug]
    server          # same, via the console script

Defaults come from the environment: VTDB_HOST (127.0.0.1), VTDB_PORT (8000),
VTDB_MCP_PATH (/mcp), VTDB_DEBUG (off). The Docker image sets VTDB_HOST=0.0.0.0.

MCP runs over streamable-http at the MCP path; ``/health`` is an
unauthenticated operational endpoint for probes.
"""

import argparse
import logging
import os
import sys

from server import __version__


def _configure_logging(debug: bool = False) -> None:
    """Set up root logging — one unified style for app + uvicorn records.

    On an interactive terminal use rich's handler — colored level badges,
    dimmed timestamps, clickable ``file:line``, pretty tracebacks. When output
    is redirected (Docker, a log collector like Grafana Alloy) fall back to the
    plain single-line format so no ANSI escape codes leak into the logs.

    NOTE: the plain format ``%(asctime)s %(levelname)s %(name)s: %(message)s``
    is parsed by the Alloy pipeline into ``level`` / ``logger`` labels and the
    entry timestamp — keep it in sync with ``k8s/alloy/config.alloy``.
    """
    level = logging.DEBUG if debug else logging.INFO
    if sys.stderr.isatty():
        from rich.logging import RichHandler

        handler: logging.Handler = RichHandler(
            rich_tracebacks=True,
            tracebacks_show_locals=False,
            show_path=True,
            log_time_format="[%X]",
        )
        # RichHandler renders level/time/path itself; the message is all we
        # format.
        logging.basicConfig(
            level=level,
            format="%(name)s  %(message)s",
            datefmt="[%X]",
            handlers=[handler],
        )
    else:
        logging.basicConfig(
            level=level,
            format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        )

    # Uvicorn installs its own handlers and sets propagate=False on these
    # loggers, which produces mismatched "INFO:     ..." lines. Clear their
    # handlers and let records bubble up to the root handler configured above
    # so all logs share one style. Paired with log_config=None in
    # server.app.run().
    for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        uvicorn_logger = logging.getLogger(name)
        uvicorn_logger.handlers.clear()
        uvicorn_logger.propagate = True


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def main() -> None:
    parser = argparse.ArgumentParser(description="vtdb-mcp server")
    parser.add_argument(
        "--host",
        default=os.environ.get("VTDB_HOST", "127.0.0.1"),
        help="Bind host [env VTDB_HOST, default 127.0.0.1]",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.environ.get("VTDB_PORT", "8000")),
        help="Bind port [env VTDB_PORT, default 8000]",
    )
    parser.add_argument(
        "--mcp-path",
        default=os.environ.get("VTDB_MCP_PATH", "/mcp"),
        help="HTTP path for the MCP streamable-http endpoint [env VTDB_MCP_PATH]",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        default=_env_flag("VTDB_DEBUG"),
        help="DEBUG-level root logging [env VTDB_DEBUG]",
    )
    args = parser.parse_args()

    _configure_logging(debug=args.debug)
    logger = logging.getLogger(__name__)
    logger.info(
        "Starting vtdb-mcp %s on %s:%d (mcp path: %s)",
        __version__,
        args.host,
        args.port,
        args.mcp_path,
    )

    from server.app import ServerConfig, run

    run(ServerConfig(host=args.host, port=args.port, mcp_path=args.mcp_path))


if __name__ == "__main__":
    main()
