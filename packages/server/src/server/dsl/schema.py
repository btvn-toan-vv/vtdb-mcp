"""Payload schema per view — the single source of truth for field names/types.

Derived from the dataset metadata (packages/server ingest writes these as
payload keys; see ``_sanitize_key`` in services/../ingest.py — columns are
sanitized the same way: spaces → ``_``, lowercased).

Values are coarse python type names for user-facing validation messages:
``str`` (keyword/text), ``int``, ``float``.
"""

from __future__ import annotations

# collection name -> field -> python type name
FIELDS: dict[str, dict[str, str]] = {
    "cells": {
        "cell_id": "int",
        "if_plate_id": "int",
        "position": "str",
        "sample": "int",
        "cell_line": "str",
        "antibody": "str",
        "protein": "str",
        "ensembl_ids": "str",
        "gene_names": "str",
        "compartment": "str",
        "cell_path": "str",
        "time_ms": "float",
    },
    "images": {
        "file_prefix": "str",
        "cell_line": "str",
        "protein": "str",
        "antibody": "str",
        "genes": "str",
        "compartment": "str",
        "umap2d_x": "float",
        "umap2d_y": "float",
        "umap3d_x": "float",
        "umap3d_y": "float",
        "umap3d_z": "float",
    },
}

# database() accepts short and plural aliases.
VIEW_ALIASES: dict[str, str] = {
    "cell": "cells",
    "cells": "cells",
    "image": "images",
    "images": "images",
}
