"""Shared fixtures: a fake Qdrant client and a tiny synthetic dataset view.

The fake mirrors just the methods server.ingest exercises, backed by plain
dicts — no server, no network. The dataset view follows the real layout:
``<view>/embeddings.npy`` + ``metadata.csv`` (row-aligned).
"""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import polars as pl
import pytest
from qdrant_client import models

from server.ingest import ViewSpec


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
