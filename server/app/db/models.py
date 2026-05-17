import uuid
from datetime import datetime

from sqlalchemy import (
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import JSON, UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class DataPoint(Base):
    __tablename__ = "data_points"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    # Composite primary key: TimescaleDB requires the partitioning column
    # (timestamp) to be part of any unique constraint on a hypertable.
    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), primary_key=True, nullable=False
    )
    lat: Mapped[float] = mapped_column(Float, nullable=False)
    lon: Mapped[float] = mapped_column(Float, nullable=False)
    metric: Mapped[str] = mapped_column(String(64), nullable=False)
    value: Mapped[float] = mapped_column(Float, nullable=False)
    unit: Mapped[str] = mapped_column(String(32), nullable=False)
    source: Mapped[str] = mapped_column(String(64), nullable=False)
    source_entity_id: Mapped[str] = mapped_column(String(128), nullable=False)
    raw_json: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    collected_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    __table_args__ = (
        Index("ix_data_points_timestamp", "timestamp"),
        Index("ix_data_points_source", "source"),
        Index("ix_data_points_metric", "metric"),
        Index("ix_data_points_source_metric_ts", "source", "metric", "timestamp"),
        Index("ix_data_points_source_entity", "source", "source_entity_id"),
        Index("ix_data_points_location", "lat", "lon"),
        UniqueConstraint(
            "source",
            "metric",
            "source_entity_id",
            "timestamp",
            name="uq_data_points_dedup",
        ),
    )

    def __repr__(self) -> str:
        return (
            f"<DataPoint {self.source}/{self.metric}={self.value} {self.unit} "
            f"at ({self.lat}, {self.lon}) @ {self.timestamp}>"
        )


class DataSource(Base):
    __tablename__ = "data_sources"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    name: Mapped[str] = mapped_column(String(64), unique=True, nullable=False)
    source_type: Mapped[str] = mapped_column(String(64), nullable=False)
    config: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    last_collected_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    status: Mapped[str] = mapped_column(String(32), default="inactive")
    error_count: Mapped[int] = mapped_column(Integer, default=0)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)

    def __repr__(self) -> str:
        return f"<DataSource {self.name} ({self.status})>"


class Anomaly(Base):
    __tablename__ = "anomalies"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    lat: Mapped[float] = mapped_column(Float, nullable=False)
    lon: Mapped[float] = mapped_column(Float, nullable=False)
    metric: Mapped[str] = mapped_column(String(64), nullable=False)
    source: Mapped[str] = mapped_column(String(64), nullable=False)
    value: Mapped[float] = mapped_column(Float, nullable=False)
    expected_value: Mapped[float | None] = mapped_column(Float, nullable=True)
    z_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    methods_triggered: Mapped[list[str]] = mapped_column(JSON, nullable=False)
    severity: Mapped[str] = mapped_column(String(16), nullable=False)
    detected_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    enrichment_records: Mapped[list["EnrichmentRecord"]] = relationship(
        back_populates="anomaly",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )

    __table_args__ = (
        Index("ix_anomalies_timestamp", "timestamp"),
        Index(
            "ix_anomalies_source_metric_ts", "source", "metric", "timestamp"
        ),
        Index("ix_anomalies_severity", "severity"),
        Index("ix_anomalies_location", "lat", "lon"),
    )

    def __repr__(self) -> str:
        return (
            f"<Anomaly {self.source}/{self.metric}={self.value} "
            f"sev={self.severity} @ {self.timestamp}>"
        )


class EnrichmentRecord(Base):
    __tablename__ = "enrichment_records"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    anomaly_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("anomalies.id", ondelete="CASCADE"),
        nullable=False,
    )
    context_window_start: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    context_window_end: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    cross_source_summary_json: Mapped[dict] = mapped_column(JSON, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    anomaly: Mapped["Anomaly"] = relationship(back_populates="enrichment_records")

    __table_args__ = (
        Index("ix_enrichment_records_anomaly_id", "anomaly_id"),
    )

    def __repr__(self) -> str:
        return (
            f"<EnrichmentRecord anomaly={self.anomaly_id} "
            f"window={self.context_window_start}..{self.context_window_end}>"
        )
