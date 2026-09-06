"""Jinja-rendered tool descriptions (same pattern as exen-mcp's
api/descriptions.py): markdown templates live in ``templates/`` next to this
file so long descriptions stay reviewable without touching tool code."""

from __future__ import annotations

from pathlib import Path

from jinja2 import Environment, FileSystemLoader

_TEMPLATES_DIR = Path(__file__).parent / "templates"
_ENV = Environment(loader=FileSystemLoader(_TEMPLATES_DIR), keep_trailing_newline=False)


def _render(name: str, **kwargs: object) -> str:
    return _ENV.get_template(name).render(**kwargs).strip()


def render_query_description() -> str:
    """Description for the ``query`` tool.

    Phase 0 has no config input; when the symbol context grows collection-aware
    functions, pass the live vocabulary in here (exen renders cohort topics /
    chart types from config the same way).
    """
    return _render(
        "query.md.j2",
        collections=["cells", "images"],
        example="output({'hello': 1 + 2.0})",
    )
