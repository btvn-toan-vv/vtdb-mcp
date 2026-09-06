"""vtdb-mcp server — a minimal FastMCP server.

Usage::

    python -m server [--host 0.0.0.0] [--port 8000] [--mcp-path /mcp] [--debug]
    server          # same, via the console script

Defaults come from the environment: VTDB_HOST (127.0.0.1), VTDB_PORT (8000),
VTDB_MCP_PATH (/mcp), VTDB_DEBUG (off), VTDB_RELOAD (off). The Docker image
sets VTDB_HOST=0.0.0.0.

MCP runs over streamable-http at the MCP path; ``/health`` is an
unauthenticated operational endpoint for probes.

``--reload`` (VTDB_RELOAD) enables uvicorn's auto-reload. Reload mode is
env-driven (the respawned child re-reads VTDB_*; CLI flags are dropped), so
prefer env vars when using it — see compose.yaml.
"""

import argparse
import logging
import os

from server import __version__
from server.log import configure_logging


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
    parser.add_argument(
        "--reload",
        action="store_true",
        default=_env_flag("VTDB_RELOAD"),
        help="Auto-reload on source changes (dev only; config from VTDB_* env)",
    )
    args = parser.parse_args()

    configure_logging(debug=args.debug)
    logger = logging.getLogger(__name__)
    logger.info(
        "Starting vtdb-mcp %s on %s:%d (mcp path: %s)%s",
        __version__,
        args.host,
        args.port,
        args.mcp_path,
        " [auto-reload]" if args.reload else "",
    )

    from server.app import ServerConfig, run

    run(
        ServerConfig(
            host=args.host,
            port=args.port,
            mcp_path=args.mcp_path,
            reload=args.reload,
        )
    )


if __name__ == "__main__":
    main()
