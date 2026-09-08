"""REVIEW DRAFT — pairwise similarity rows: ``db.pairwise(...)``.

API being pinned (Row workflow; chainable like meta/agg/search):

    db = database("cells")
    db.pairwise([0, 3, 6])                          # ids → pairs
    db.pairwise(col("gene_names") == "G03")         # filter → all pairs within
    db.pairwise(None).sortby("score", desc)         # whole view (cap-bound)

Rows per pair: ``{"id_a", "id_b", "score"}`` — STRICT upper triangle
(id_a < id_b, no self-pairs), ordered by (id_a, id_b) ascending deterministically.
Score is cosine similarity — the collections' distance — as float.

Chainable: the return is ``Rows`` — ``.sortby("score", descending=True)`` works.

Trap guard: number of pair rows = C(n, 2); the cap counts PAIRS, not ids
(default 10_000; filter-derived sets hit it fast). Raise loudly with exact
guidance; missing anchor ids error by name. A 0/1-element selection is []
(pairs need two).
"""

import numpy as np
import pytest
from fastmcp.client.client import CallToolResult

from .conftest import dedent_code


def _pairs(call_tool, code: str) -> list[dict]:
    result: CallToolResult = call_tool("query", {"code": dedent_code(code)})
    return result.data["result"]["pairs"]


def _run(call_tool, code: str) -> CallToolResult:
    return call_tool("query", {"code": dedent_code(code)})


def _cos(a, b) -> float:
    # strict-json floats; numpy dot for f32 vectors
    den = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / den) if den else 0.0


# ---- shape / content ----------------------------------------------------------


def test_pairwise_ids_shape_upper_triangle(call_tool, hermetic_env) -> None:
    pairs = _pairs(
        call_tool, 'output({"pairs": database("cells").pairwise([0, 3, 6])})'
    )
    assert [(p["id_a"], p["id_b"]) for p in pairs] == [(0, 3), (0, 6), (3, 6)]
    assert all(p["id_a"] < p["id_b"] for p in pairs)  # no self/mirror pairs
    assert set(pairs[0]) == {"id_a", "id_b", "score"}


def test_pairwise_cosine_matches_ground_truth(call_tool, hermetic_env) -> None:
    vecs = hermetic_env["vectors"]["cells"]
    pairs = _pairs(
        call_tool, 'output({"pairs": database("cells").pairwise([0, 3, 6])})'
    )
    for p in pairs:
        gold = _cos(vecs[p["id_a"]], vecs[p["id_b"]])
        assert p["score"] == pytest.approx(gold, rel=1e-5)


def test_pairwise_filter_form(call_tool, hermetic_env) -> None:
    # gene G03 -> i%12==3 rows (no nulls in gene_names): 20 rows -> C(20,2)=190.
    pairs = _pairs(
        call_tool,
        'output({"pairs": database("cells").pairwise(col("gene_names") == "G03")})',
    )
    assert len(pairs) == 190
    assert all(p["id_a"] < p["id_b"] for p in pairs)


def test_pairwise_sorted_categories_call(call_tool, hermetic_env) -> None:
    # chains: sort highest similarity first, then read top row's ids.
    pairs = _pairs(
        call_tool,
        'output({"pairs": database("cells").pairwise(col("gene_names") == "G03").sortby("score", descending=True)})',
    )
    assert len(pairs) == 190
    scores = [p["score"] for p in pairs]
    assert scores == sorted(scores, reverse=True)


def test_pairwise_bounds_in_range(call_tool, hermetic_env) -> None:
    pairs = _pairs(
        call_tool, 'output({"pairs": database("cells").pairwise([0, 3, 6])})'
    )
    assert all(-1.0001 <= p["score"] <= 1.0001 for p in pairs)


# ---- edges + caps --------------------------------------------------------------


def test_pairwise_single_id_returns_empty(call_tool, hermetic_env) -> None:
    pairs = _pairs(call_tool, 'output({"pairs": database("cells").pairwise([3])})')
    assert pairs == []


def test_pairwise_empty_filter_returns_empty(call_tool, hermetic_env) -> None:
    pairs = _pairs(
        call_tool,
        'output({"pairs": database("cells").pairwise(col("gene_names") == "NOPE")})',
    )
    assert pairs == []


def test_pairwise_cap_count_pairs_not_ids(call_tool, hermetic_env) -> None:
    # C(20,2)=190 pairs is under the default cap (10000) but over limit=100.
    result = _run(
        call_tool,
        'output({"pairs": database("cells").pairwise(col("gene_names") == "G03", limit=100)})',
    )
    error = result.data["error"]
    assert error["type"] == "DSLUsageError"
    assert "pairs" in error["message"] and "limit" in error["message"]


def test_pairwise_limit_raise_ok(call_tool, hermetic_env) -> None:
    pairs = _pairs(
        call_tool,
        'output({"pairs": database("cells").pairwise(col("gene_names") == "G03", limit=200)})',
    )
    assert len(pairs) == 190  # cap raised past the pair count


def test_pairwise_missing_anchor_id(call_tool, hermetic_env) -> None:
    result = _run(
        call_tool, 'output({"pairs": database("cells").pairwise([0, 999_999])})'
    )
    error = result.data["error"]
    assert error["type"] == "DSLUsageError"
    assert "999999" in error["message"]


def test_pairwise_bad_type(call_tool, hermetic_env) -> None:
    result = _run(call_tool, 'output({"pairs": database("cells").pairwise("G03")})')
    assert result.data["error"]["type"] == "DSLUsageError"
