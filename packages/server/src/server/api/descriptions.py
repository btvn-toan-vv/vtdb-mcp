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

    Schema tables render from dsl/schema.py (payload field names/types) and
    ingest.ViewSpec.payload_indexes (the ⚡ marks) — the description can never
    drift from the registered DSL vocabulary. Anything new (e.g. more views)
    shows up on the next server start.
    """
    from server.dsl.schema import FIELDS
    from server.ingest import IMAGES, CELLS

    def view_row(sym) -> dict:
        indexed = set(sym.payload_indexes)
        return {
            "name": sym.collection,
            "dims": sym.dims,
            "desc": (
                "one row per segmented cell (payload: identity + labels)"
                if sym is CELLS
                else "one row per source HPA image"
            ),
            "note": (
                ""
                if sym is IMAGES
                else "8.2% of rows are the null-`cell_path` failed-extraction cohort"
                " — see the data-exam notebook."
            ),
            "fields": [
                {
                    "name": f,
                    "kind": t,
                    "indexed": f in indexed,
                }
                for f, t in sorted(FIELDS[sym.collection].items())
            ],
        }

    return _render("query.md.j2", views=[view_row(CELLS), view_row(IMAGES)])
