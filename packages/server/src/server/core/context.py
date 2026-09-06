"""VtdbSymbolContext — the biocircle SymbolContext backing ``query()`` code.

Direct mirror of exen-mcp's ExenMcpSymbolContext: a biocircle SymbolContext
subclass that owns the DSL vocabulary. Phase 1 registers the filtering DSL
(``database`` / ``col`` + the FilterExpr classes, see dsl/registration.py);
query code can now run filters like::

    db = database("image")
    output({"num_ids": len(db.resolve_ids(col("cell line") == "U2OS"))})
"""

from __future__ import annotations

from biocircle import SymbolContext


class VtdbSymbolContext(SymbolContext):
    """biocircle SymbolContext for vtdb-mcp, with the DSL registered."""

    def __init__(self) -> None:
        super().__init__()
        # Deferred import: registration pulls in server.dsl.* which imports
        # this module's class for typing.
        from server.dsl.registration import register_dsl_symbols

        register_dsl_symbols(self)
