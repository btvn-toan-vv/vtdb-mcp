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
from server.ingest import ViewSpec

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
