"""VtdbSymbolContext — the biocircle SymbolContext backing ``query()`` code.

Direct mirror of exen-mcp's ExenMcpSymbolContext: a biocircle SymbolContext
subclass that owns the DSL vocabulary. exen registers GPU/DataFrame primitives;
we will register Qdrant-backed functions (count/search over the cells/images
collections) when the executor grows.

Phase 0: nothing registered — query code traces against biocircle's safe
builtins only, so today's usable surface is exactly::

    output({"hello": 1 + 2.0})
"""

from __future__ import annotations

from biocircle import SymbolContext


class VtdbSymbolContext(SymbolContext):
    """biocircle SymbolContext for vtdb-mcp (Phase 0: no custom symbols).

    Registration goes through biocircle's contexts later, exen-style:
    ``ctx.function_context.registrar(func)`` for functions,
    ``ctx.module_context`` for namespaced helpers, ``ctx.macro_context`` for
    tracer-time macros.
    """
