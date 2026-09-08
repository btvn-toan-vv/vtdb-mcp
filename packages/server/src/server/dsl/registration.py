"""Register the filtering DSL into a VtdbSymbolContext.

biocircle-coupling lives ONLY here (mirror of exen-mcp's
executor/registration/): functions via ``function_context.registrar``,
expression/handle classes via ``class_context.registrar`` (auto-discovery
registers the allowed dunder overloads + public methods).
"""

from __future__ import annotations

from server.core.context import VtdbSymbolContext
from server.dsl.aggregations import AggExpr, row_count
from server.dsl.expressions import FieldExpr, FilterExpr, StrExpr, col
from server.dsl.sorting import Rows
from server.dsl.views import FilteredView, GroupBy, ViewHandle, database


def register_dsl_symbols(ctx: VtdbSymbolContext) -> None:
    """DSL vocabulary: database/col/row_count + the expression/handle classes."""
    for fn in (database, col, row_count):
        ctx.function_context.registrar(fn)
    for cls in (
        FieldExpr,
        StrExpr,
        FilterExpr,
        AggExpr,
        ViewHandle,
        FilteredView,
        GroupBy,
        Rows,
    ):
        # method_filter=None: discover all public methods + allowed dunders
        # (the default () registers NO methods — silent dead surface).
        ctx.class_context.registrar(method_filter=None, category="dsl")(cls)
