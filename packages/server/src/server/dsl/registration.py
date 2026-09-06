"""Register the filtering DSL into a VtdbSymbolContext.

biocircle-coupling lives ONLY here (mirror of exen-mcp's
executor/registration/): functions via ``function_context.registrar``,
expression/handle classes via ``class_context.registrar`` (auto-discovery
registers the allowed dunder overloads + public methods).
"""

from __future__ import annotations

from server.core.context import VtdbSymbolContext
from server.dsl.expressions import FieldExpr, FilterExpr, StrExpr, col
from server.dsl.views import ViewHandle, database


def register_dsl_symbols(ctx: VtdbSymbolContext) -> None:
    """Phase 1 vocabulary: database/col + the expression classes."""
    for fn in (database, col):
        ctx.function_context.registrar(fn)
    for cls in (FieldExpr, StrExpr, FilterExpr, ViewHandle):
        # method_filter=None: discover all public methods + allowed dunders
        # (the default () registers NO methods — silent dead surface).
        ctx.class_context.registrar(method_filter=None, category="dsl")(cls)
