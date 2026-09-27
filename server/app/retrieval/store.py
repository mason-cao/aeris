"""ChromaDB store for analog vectors.

Holds ids, vectors and small metadata only: never explanation text, claims,
scores or labels.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.retrieval.features import FEATURE_NAMES, SCHEMA_VERSION

COLLECTION_NAME = "event_analogs"
SERVER_DIR = Path(__file__).resolve().parents[2]

# l2 in ChromaDB is the squared Euclidean distance, which ranks like Euclidean.
_HNSW: dict[str, Any] = {"space": "l2", "ef_construction": 200, "ef_search": 200}


class StoreMismatchError(RuntimeError):
    """The store was built with another feature schema or scaling."""


@dataclass(frozen=True)
class IndexedEvent:
    anomaly_id: str
    vector: tuple[float, ...]
    metric: str
    source: str
    source_entity_id: str
    t_epoch: int
    window_end_epoch: int
    imputed: tuple[str, ...]

    def metadata(self) -> dict[str, str | int]:
        return {
            "metric": self.metric,
            "source": self.source,
            "source_entity_id": self.source_entity_id,
            "t_epoch": self.t_epoch,
            "window_end_epoch": self.window_end_epoch,
            "schema_version": SCHEMA_VERSION,
            "imputed_features": ",".join(self.imputed),
        }


@dataclass(frozen=True)
class Neighbor:
    anomaly_id: str
    squared_distance: float
    vector: tuple[float, ...]


def resolve_store_path(configured: str) -> Path:
    path = Path(configured)
    return path if path.is_absolute() else SERVER_DIR / path


def open_collection(client: Any, *, scaling_sha256: str) -> Any:
    """Create or reopen the collection, refusing one built for other features."""
    expected = {"schema_version": SCHEMA_VERSION, "scaling_sha256": scaling_sha256}
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        configuration={"hnsw": dict(_HNSW)},
        metadata=dict(expected),
        embedding_function=None,
    )
    stored = collection.metadata or {}
    found = {key: stored.get(key) for key in expected}
    if found != expected:
        raise StoreMismatchError(f"store holds {found}, this code expects {expected}")
    return collection


def upsert_events(
    collection: Any, events: Sequence[IndexedEvent], *, batch_size: int = 1000
) -> None:
    for event in events:
        if len(event.vector) != len(FEATURE_NAMES):
            raise ValueError(f"{event.anomaly_id}: vector has {len(event.vector)} dimensions")
    for start in range(0, len(events), batch_size):
        batch = events[start : start + batch_size]
        collection.upsert(
            ids=[event.anomaly_id for event in batch],
            embeddings=[list(event.vector) for event in batch],
            metadatas=[event.metadata() for event in batch],
        )


def query_neighbors(
    collection: Any,
    vector: Sequence[float],
    *,
    metric: str,
    before_epoch: int,
    k: int,
) -> list[Neighbor]:
    """Nearest events of ``metric`` whose evidence window closed by ``before_epoch``."""
    if k < 1:
        raise ValueError("k must be at least 1")
    result = collection.query(
        query_embeddings=[[float(x) for x in vector]],
        n_results=k,
        where={
            "$and": [
                {"metric": {"$eq": metric}},
                {"window_end_epoch": {"$lte": before_epoch}},
            ]
        },
        include=["distances", "embeddings"],
    )
    return [
        Neighbor(anomaly_id, float(distance), tuple(float(x) for x in embedding))
        for anomaly_id, distance, embedding in zip(
            result["ids"][0], result["distances"][0], result["embeddings"][0], strict=True
        )
    ]
