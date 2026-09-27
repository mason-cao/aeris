from __future__ import annotations

from pathlib import Path

import chromadb
import numpy as np
import pytest

from app.retrieval.features import FEATURE_NAMES
from app.retrieval.store import (
    SERVER_DIR,
    IndexedEvent,
    StoreMismatchError,
    open_collection,
    query_neighbors,
    resolve_store_path,
    upsert_events,
)

DIM = len(FEATURE_NAMES)


def _event(i: int, vector: np.ndarray, *, metric: str = "no2", window_end: int = 0) -> IndexedEvent:
    return IndexedEvent(
        anomaly_id=f"a{i:04d}",
        vector=tuple(float(x) for x in vector),
        metric=metric,
        source="tceq",
        source_entity_id="S1",
        t_epoch=window_end - 36 * 3600,
        window_end_epoch=window_end,
        imputed=(),
    )


@pytest.fixture
def collection(tmp_path: Path):
    client = chromadb.PersistentClient(path=str(tmp_path / "store"))
    return open_collection(client, scaling_sha256="s" * 64)


def test_resolve_store_path() -> None:
    assert resolve_store_path("data/chromadb") == SERVER_DIR / "data" / "chromadb"
    assert resolve_store_path("/tmp/elsewhere") == Path("/tmp/elsewhere")


def test_reopening_with_other_scaling_is_refused(tmp_path: Path) -> None:
    client = chromadb.PersistentClient(path=str(tmp_path / "store"))
    open_collection(client, scaling_sha256="a" * 64)
    open_collection(client, scaling_sha256="a" * 64)
    with pytest.raises(StoreMismatchError):
        open_collection(client, scaling_sha256="b" * 64)


def test_query_matches_brute_force(collection) -> None:
    rng = np.random.default_rng(0)
    vectors = rng.normal(size=(400, DIM))
    metrics = rng.choice(["no2", "ozone"], size=400)
    ends = rng.integers(0, 10_000, size=400)
    upsert_events(collection, [_event(i, vectors[i], metric=str(metrics[i]), window_end=int(ends[i])) for i in range(400)])
    stored = collection.get(include=["embeddings"])
    by_id = dict(zip(stored["ids"], np.asarray(stored["embeddings"]), strict=True))
    for q in range(0, 400, 17):
        found = query_neighbors(collection, vectors[q], metric=str(metrics[q]), before_epoch=int(ends[q]), k=3)
        pool = [f"a{i:04d}" for i in range(400) if metrics[i] == metrics[q] and ends[i] <= ends[q]]
        exact = sorted(float(((by_id[p] - vectors[q]) ** 2).sum()) for p in pool)[:3]
        assert [n.squared_distance for n in found] == pytest.approx(exact, rel=1e-5, abs=1e-5)


def test_window_end_boundary_is_inclusive(collection) -> None:
    upsert_events(collection, [_event(1, np.zeros(DIM), window_end=100), _event(2, np.zeros(DIM), window_end=101)])
    found = query_neighbors(collection, np.zeros(DIM), metric="no2", before_epoch=100, k=3)
    assert [n.anomaly_id for n in found] == ["a0001"]


def test_other_metric_and_short_results(collection) -> None:
    upsert_events(collection, [_event(1, np.zeros(DIM), metric="ozone", window_end=0), _event(2, np.ones(DIM), window_end=0)])
    found = query_neighbors(collection, np.zeros(DIM), metric="no2", before_epoch=0, k=3)
    assert [n.anomaly_id for n in found] == ["a0002"]
    assert query_neighbors(collection, np.zeros(DIM), metric="co", before_epoch=0, k=3) == []


def test_metadata_holds_no_text_payload() -> None:
    meta = _event(1, np.zeros(DIM), window_end=0).metadata()
    assert set(meta) == {
        "metric", "source", "source_entity_id", "t_epoch", "window_end_epoch", "schema_version", "imputed_features",
    }


def test_wrong_dimension_is_refused(collection) -> None:
    bad = IndexedEvent("x", (0.0,), "no2", "tceq", "S1", 0, 0, ())
    with pytest.raises(ValueError):
        upsert_events(collection, [bad])
