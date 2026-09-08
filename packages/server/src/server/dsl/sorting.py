"""Chainable row collections for the DSL: ``Rows``.

``meta()``, ``group_by(...).agg(...)`` and ``search()`` return this
list-subclass, so post-processing hangs off the result:

    db.group_by("cell_line").agg(row_count()).sortby("count", descending=True)

Stable (Timsort): ties keep input order. Rows missing a ``by`` field sort
LAST in both directions (None isn't ordered against values).
"""

from __future__ import annotations

from typing import Any

from server.dsl.expressions import DSLUsageError


class Rows(list):
    """``list[dict]`` with chainable DSL helpers (sortby today; more later).

    Explicit indexing wrappers: biocircle discovers only OWN methods on a
    class, so the list dunders must be re-exposed here or ``rows[0]`` dies.
    """

    def __getitem__(self, key: Any) -> Any:
        out = list.__getitem__(self, key)
        return Rows(out) if isinstance(out, list) else out

    def __len__(self) -> int:
        return list.__len__(self)

    def sortby(self, *by: str | list[str], descending: bool = False) -> Rows:
        """Sort rows by field names: ``sortby("a", "b")`` (or one list of names).

        Later keys break ties; ties keep input order (stable). Rows missing a
        ``by`` field collect at the end in both directions.
        """
        keys: list[str] = []
        for b in by:
            if isinstance(b, str):
                keys.append(b)
            elif isinstance(b, list) and b and all(isinstance(k, str) for k in b):
                keys.extend(b)
            else:
                raise DSLUsageError(
                    "sortby() expects field names: sortby('a', 'b') or sortby(['a', 'b'])"
                )
        if not keys:
            raise DSLUsageError("sortby() needs at least one field name")
        if not isinstance(descending, bool):
            raise DSLUsageError(
                f"descending= expects bool, got {type(descending).__name__}"
            )
        if not self:
            return Rows()
        for r in self:
            if not isinstance(r, dict):
                raise DSLUsageError(
                    "sortby() only applies to list-of-dict rows (meta / "
                    "group_by().agg() / search()); this is a " + type(r).__name__
                )

        have = [r for r in self if all(k in r and r[k] is not None for k in keys)]
        missing = [r for r in self if r not in have]

        available = sorted({k for r in self for k in r})
        unknown = [k for k in keys if k not in available]
        if unknown:
            raise DSLUsageError(
                f"sortby(): unknown field(s) {', '.join(unknown)}. "
                f"Available in these rows: {', '.join(available)}"
            )

        ordered = sorted(
            have, key=lambda r: tuple(r[k] for k in keys), reverse=descending
        )
        return Rows(ordered + missing)


def as_rows(rows: list[dict[str, Any]]) -> Rows:
    """Wrap a plain list-of-dicts as Rows (idempotent — views call sites)."""
    return rows if isinstance(rows, Rows) else Rows(rows)
