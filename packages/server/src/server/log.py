"""Root logging setup — one unified style for app + uvicorn records.

In its own module (not ``__main__``) because uvicorn's ``--reload`` respawns
the app in a fresh child process that never runs ``__main__`` — the
:func:`server.app.app_from_env` factory calls this again on behalf of the
child.
"""

import logging
import sys


def configure_logging(debug: bool = False) -> None:
    """Set up root logging.

    On an interactive terminal use rich's handler — colored level badges,
    dimmed timestamps, clickable ``file:line``, pretty tracebacks. When output
    is redirected (Docker, a log collector like Grafana Alloy) fall back to the
    plain single-line format so no ANSI escape codes leak into the logs.

    NOTE: the plain format ``%(asctime)s %(levelname)s %(name)s: %(message)s``
    is parsed by the Alloy pipeline into ``level`` / ``logger`` labels and the
    entry timestamp — keep it in sync with ``docker/alloy/config.alloy``.
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
