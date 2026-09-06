"""Shared fixtures for the server tests.

App surface (see tests/test_client_template.py for copy-paste examples):
    mcp        fresh FastMCP server, tools registered, no HTTP
    call_tool  sync bridge: call_tool("echo", {"text": "x"}).data

Qdrant ingest (see test_ingest.py):
    client     in-memory FakeQdrant mirroring the client methods ingest uses
    view       tiny ViewSpec (4-d)
    dataset    synthetic <view>/embeddings.npy + metadata.csv in tmp_path
"""

import asyncio
import time
from collections.abc import Callable
from pathlib import Path
from textwrap import dedent
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import polars as pl
import pytest
from fastmcp import Client, FastMCP
from fastmcp.client.client import CallToolResult
from qdrant_client import QdrantClient, models
from server.app import ServerConfig, create_mcp
from server.ingest import CELLS, IMAGES, ViewSpec, load_view

# ---- MCP tool surface -------------------------------------------------------


def dedent_code(s: str) -> str:
    """Indentation-friendly multi-line query strings: dedent + strip the
    surrounding blank lines, so a test can write code aligned with the call.

        call_tool("query", {"code": dedent_code(\"\"\"
            output({'hello': 1 + 2.0})
        \"\"\")})
    """
    return dedent(s).strip("\n")


@pytest.fixture
def mcp() -> FastMCP:
    """Fresh FastMCP server per test (tools + custom routes, no HTTP layer)."""
    return create_mcp(ServerConfig())


@pytest.fixture
def call_tool(mcp: FastMCP) -> Callable[..., CallToolResult]:
    """Synchronous tool call over the in-memory client: each call opens and
    closes its own session, so tests stay stateless and asyncio-free.

        result = call_tool("echo", {"text": "hi"})
        assert result.data == "hi"
    """

    def _call(name: str, arguments: dict[str, Any] | None = None) -> CallToolResult:
        async def _run() -> CallToolResult:
            async with Client(mcp) as client:
                return await client.call_tool(name, arguments or {})

        return asyncio.run(_run())

    return _call


class FakeQdrant:
    """In-memory stand-in for QdrantClient (the parts ingest.py uses)."""

    def __init__(self) -> None:
        self.collections: dict[str, dict] = {}
        self.calls: list[str] = []

    # -- schema ------------------------------------------------------------
    def collection_exists(self, name: str) -> bool:
        return name in self.collections

    def create_collection(
        self,
        collection_name: str,
        vectors_config: models.VectorParams,
        optimizers_config: models.OptimizersConfigDiff | None = None,
        **_: object,
    ) -> bool:
        self.calls.append("create_collection")
        self.collections[collection_name] = {
            "vectors": vectors_config,
            "optimizers": optimizers_config,
            "updated_optimizers": [],
            "points": {},
            "indexes": [],
        }
        return True

    def delete_collection(self, collection_name: str, **_: object) -> None:
        self.calls.append("delete_collection")
        del self.collections[collection_name]

    def create_payload_index(
        self,
        collection_name: str,
        field_name: str,
        field_schema: object = None,
        **_: object,
    ) -> None:
        self.collections[collection_name]["indexes"].append((field_name, field_schema))

    def get_collection(self, collection_name: str) -> SimpleNamespace:
        coll = self.collections[collection_name]
        return SimpleNamespace(
            config=SimpleNamespace(params=SimpleNamespace(vectors=coll["vectors"])),
            status=models.CollectionStatus.GREEN,
        )

    def get_collections(self) -> SimpleNamespace:
        return SimpleNamespace(
            collections=[SimpleNamespace(name=n) for n in self.collections]
        )

    def update_collection(
        self,
        collection_name: str,
        optimizers_config: models.OptimizersConfigDiff | None = None,
        **_: object,
    ) -> bool:
        self.collections[collection_name]["updated_optimizers"].append(
            optimizers_config
        )
        return True

    # -- points --------------------------------------------------------------
    def count(self, collection_name: str, **_: object) -> SimpleNamespace:
        return SimpleNamespace(count=len(self.collections[collection_name]["points"]))

    def upload_points(
        self, collection_name: str, points: list[models.PointStruct], **_: object
    ) -> None:
        self.calls.append("upload_points")
        for p in points:
            self.collections[collection_name]["points"][p.id] = p.payload

    # -- test helpers ----------------------------------------------------------
    def seed_points(self, collection_name: str, ids: range) -> None:
        """Pre-fill points as if a previous run uploaded rows ``ids``."""
        for i in ids:
            self.collections[collection_name]["points"][i] = {"seeded": True}

    @property
    def uploads(self) -> int:
        return self.calls.count("upload_points")


@pytest.fixture
def client() -> FakeQdrant:
    return FakeQdrant()


# ---- Hermetic integration stack (test-filtering tier 1) --------------------
#
# A real Qdrant container, a synthetic deterministic dataset loaded via the
# real ingest path, and the DSL pointed at it — zero dependence on the
# ~/data/subcellular_embeddings snapshot. Needs only local Docker; skips when
# the daemon is unreachable.

_TEST_QDRANT_PORT = 56333
_TEST_QDRANT_GRPC_PORT = 56334
_TEST_QDRANT_CONTAINER = "vtdb-mcp-test-qdrant"

_TEST_ENV = {
    "QDRANT_HOST": "127.0.0.1",
    "QDRANT_PORT": str(_TEST_QDRANT_PORT),
    "QDRANT_GRPC_PORT": str(_TEST_QDRANT_GRPC_PORT),
}


def _docker_usable() -> bool:
    import shutil
    import subprocess

    if shutil.which("docker") is None:
        return False
    try:
        subprocess.run(["docker", "info"], capture_output=True, check=True, timeout=15)
        return True
    except (subprocess.SubprocessError, FileNotFoundError):
        return False


def _write_synthetic_dataset(base: Path) -> dict[str, dict[str, int]]:
    """Deterministic views; returns {"cells": {...counts}, "images": {...}}.

    Payload columns mirror the real dataset's schema (sanitized at ingest).
    Both views get None coverage so is_null paths have real rows.
    """
    rng = np.random.default_rng(7)
    n_cells, n_images = 240, 120
    cell_line_cycle = ["U2OS", "HeLa", "MCF-7"]

    # ---- cells ----
    compartment_cycle = ["Nucleoplasm", "Cytosol", None, "Vesicles"]
    cell_meta = pl.DataFrame(
        {
            "cell_id": list(range(n_cells)),
            "if_plate_id": [807 + i % 3 for i in range(n_cells)],
            "position": [f"P{i % 96}" for i in range(n_cells)],
            "sample": [i % 4 for i in range(n_cells)],
            "cell line": [cell_line_cycle[i % 3] for i in range(n_cells)],
            "antibody": [f"AB{i % 17}" for i in range(n_cells)],
            "protein": [f"PROT{i % 23}" for i in range(n_cells)],
            "ensembl_ids": [f"ENSG{i:011d}" for i in range(n_cells)],
            "gene_names": [f"G{i % 12:02d}" for i in range(n_cells)],
            "compartment": [compartment_cycle[i % 4] for i in range(n_cells)],
            "cell_path": [
                None if i % 5 == 0 else f"807_P{i % 96}_{i % 4}_{i % 8}.png"
                for i in range(n_cells)
            ],
            "Time (ms)": [
                None if i % 7 == 0 else 20.0 + (i % 17) * 0.1 for i in range(n_cells)
            ],
        }
    )
    # ---- images ----
    img_meta = pl.DataFrame(
        {
            "file_prefix": [f"hpa/t{i:04d}" for i in range(n_images)],
            "cell line": [cell_line_cycle[i % 3] for i in range(n_images)],
            "protein": [f"PROT{i % 23}" for i in range(n_images)],
            "antibody": [f"AB{i % 17}" for i in range(n_images)],
            "genes": [f"G{i % 12:02d}" for i in range(n_images)],
            "compartment": [compartment_cycle[(i + 1) % 4] for i in range(n_images)],
            "umap2d x": [float(i % 17) - 8.0 for i in range(n_images)],
            "umap2d y": [float((i * 3) % 17) - 8.0 for i in range(n_images)],
            "umap3d x": [float((i * 5) % 17) - 8.0 for i in range(n_images)],
            "umap3d y": [float((i * 7) % 17) - 8.0 for i in range(n_images)],
            "umap3d z": [float((i * 11) % 17) - 8.0 for i in range(n_images)],
        }
    )
    for sub in ("cell_embedding", "image_embedding"):
        (base / sub).mkdir(parents=True, exist_ok=False)
    np.save(
        base / "cell_embedding" / "embeddings.npy",
        rng.random((n_cells, CELLS.dims), dtype=np.float32),
    )
    np.save(
        base / "image_embedding" / "embeddings.npy",
        rng.random((n_images, IMAGES.dims), dtype=np.float32),
    )

    def counts(df: pl.DataFrame, view: str) -> dict[str, int]:
        cl, comp = df["cell line"], df["compartment"]
        is_null = comp.is_null()
        out = {
            "rows": df.height,
            "u2os": int((cl == "U2OS").sum()),
            "compartment_null": int(is_null.sum()),
        }
        if view == "images":
            x = df["umap2d x"]
            out["u2os_pos_x"] = int(((cl == "U2OS") & (x > 0)).sum())
            out["is_in_pos_x"] = int((cl.is_in(["U2OS", "MCF-7"]) & (x > 0)).sum())
        else:
            g07 = df["gene_names"] == "G07"
            out["null_or_g07"] = int((is_null | g07).sum())
            out["path_null"] = int(df["cell_path"].is_null().sum())
        return out

    expected = {
        "cells": counts(cell_meta, "cells"),
        "images": counts(img_meta, "images"),
        "payload_of": PayloadOracle(cell_meta, img_meta),
    }
    cell_meta.write_csv(base / "cell_embedding" / "metadata.csv")
    img_meta.write_csv(base / "image_embedding" / "metadata.csv")
    return expected


class PayloadOracle:
    """payload_of(view, row) → the payload dict the DSL should return.

    Mirrors the ingest pipeline exactly: keys sanitized (spaces → `_`,
    lowercased), None values dropped. Built from the same frames that were
    written to the fixture CSVs, so expectations can never drift from data.
    """

    def __init__(self, cell_meta: pl.DataFrame, img_meta: pl.DataFrame) -> None:
        self._frames = {"cells": cell_meta, "images": img_meta}
        self._cache: dict[str, list[dict[str, Any]]] = {}

    def __call__(self, view: str, row: int) -> dict[str, Any]:
        import re

        if view not in self._cache:
            clean = re.compile(r"[^0-9A-Za-z]+")
            rows: list[dict[str, Any]] = []
            for record in self._frames[view].iter_rows(named=True):
                rows.append(
                    {
                        clean.sub("_", k).strip("_").lower(): v
                        for k, v in record.items()
                        if v is not None
                    }
                )
            self._cache[view] = rows
        return self._cache[view][row]


@pytest.fixture(scope="session")
def qdrant_test_stack(tmp_path_factory):
    """Spin a throwaway Qdrant + synthetic dataset; yield per-view facts.

    Skip conditions: no docker daemon, or the image missing (pull once:
    ``docker pull qdrant/qdrant:v1.19.0``).
    """
    import os
    import subprocess

    if not _docker_usable():
        pytest.skip("docker daemon unreachable")
    probe = subprocess.run(
        ["docker", "image", "inspect", "qdrant/qdrant:v1.19.0"],
        capture_output=True,
        check=False,
    )
    if probe.returncode != 0:
        subprocess.run(["docker", "pull", "qdrant/qdrant:v1.19.0"], check=True)

    subprocess.run(
        ["docker", "rm", "-f", _TEST_QDRANT_CONTAINER],
        capture_output=True,
        check=False,
    )
    subprocess.run(
        [
            "docker",
            "run",
            "-d",
            "--name",
            _TEST_QDRANT_CONTAINER,
            "-p",
            f"127.0.0.1:{_TEST_QDRANT_PORT}:6333",
            "-p",
            f"127.0.0.1:{_TEST_QDRANT_GRPC_PORT}:6334",
            "qdrant/qdrant:v1.19.0",
        ],
        check=True,
        capture_output=True,
    )

    from qdrant_client import QdrantClient

    test_client = QdrantClient(
        host="127.0.0.1",
        port=_TEST_QDRANT_PORT,
        grpc_port=_TEST_QDRANT_GRPC_PORT,
        prefer_grpc=True,
        timeout=120,
        check_compatibility=False,
    )
    import urllib.request

    for _ in range(60):
        try:
            with urllib.request.urlopen(
                f"http://127.0.0.1:{_TEST_QDRANT_PORT}/healthz", timeout=1
            ) as r:
                if r.status == 200:
                    break
        except OSError:
            time.sleep(0.5)
    else:
        subprocess.run(
            ["docker", "rm", "-f", _TEST_QDRANT_CONTAINER],
            capture_output=True,
            check=False,
        )
        pytest.fail("test qdrant container never became healthy")

    base = tmp_path_factory.mktemp("dataset")
    expected = _write_synthetic_dataset(base)
    load_view(test_client, CELLS, base, limit=0, force=False)
    load_view(test_client, IMAGES, base, limit=0, force=False)
    test_client.count("cells")  # post-condition sanity

    # Point the DSL's client factory at the test container for this session.
    old_env = {k: os.environ.get(k) for k in _TEST_ENV}
    os.environ.update(_TEST_ENV)
    from server.services.qdrant import reset_client

    reset_client()
    try:
        yield expected
    finally:
        for k, v in old_env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        reset_client()
        subprocess.run(
            ["docker", "rm", "-f", _TEST_QDRANT_CONTAINER],
            capture_output=True,
            check=False,
        )


@pytest.fixture
def hermetic_env(qdrant_test_stack):
    """Tier-1 per-test entry: repaint test-Qdrant env + drop the cached client.

    The session fixture sets the env once at spin-up, but a combined
    `-m ""` run may interleave modules that repaint (the live tier pops
    QDRANT_* to default to the dev stack). Repaint per test so tier 1 is
    immune to module cross-talk in any order. Yields the same expected-facts
    dict as the session fixture.
    """
    import os

    from server.services.qdrant import reset_client

    os.environ.update(_TEST_ENV)
    reset_client()
    yield qdrant_test_stack


def as_client(fake: FakeQdrant) -> QdrantClient:
    """Type-boundary cast: the fake is a structural stand-in for QdrantClient.

    Tests grab fake internals (``fake.collections``, ``fake.uploads``) AND hand
    the same object to ingest functions that declare ``QdrantClient`` — cast at
    the boundary rather than typing the fake as the real client (which would
    hide fake-only attributes from Pylance).
    """
    return cast(QdrantClient, fake)


@pytest.fixture
def view() -> ViewSpec:
    return ViewSpec(
        collection="test_cells",
        dir_name="test_view",
        dims=4,
        payload_indexes={
            "cell_line": models.PayloadSchemaType.KEYWORD,
            "if_plate_id": models.PayloadSchemaType.INTEGER,
        },
    )


@pytest.fixture
def dataset(tmp_path: Path, view: ViewSpec) -> Path:
    """5-row synthetic view; row 0 exercises the None-drop, row 1 a 'null' path."""
    view_dir = tmp_path / view.dir_name
    view_dir.mkdir()
    meta = pl.DataFrame(
        {
            "cell_id": [1, 2, 3, 4, 5],
            "cell line": ["HeLa", "U2OS", "HeLa", "MCF-7", "HeLa"],
            "if_plate_id": [807, 807, 874, 874, 943],
            "compartment": ["Nucleoplasm", None, "Cytosol", "Mitochondria", None],
            "Time (ms)": [21.1, 20.9, None, 22.3, 19.8],
        }
    )
    meta.write_csv(view_dir / "metadata.csv")
    rng = np.random.default_rng(0)
    np.save(view_dir / "embeddings.npy", rng.random((5, view.dims), dtype=np.float32))
    return tmp_path
