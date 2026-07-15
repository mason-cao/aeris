import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import (
    CHAR,
    JSON,
    Boolean,
    DateTime,
    Dialect,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    TypeDecorator,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from sqlalchemy.types import TypeEngine


class GUID(TypeDecorator[uuid.UUID]):
    """UUID column stored in a form SQLite cannot misread as a number.

    Postgres keeps the native UUID type. Every other dialect stores the
    dashed 36-char string: the bare ``UUID`` DDL keyword gets NUMERIC
    affinity on SQLite, and an undashed hex id made of digits plus a single
    ``e`` (~1 in 10^6 of them) is silently coerced to a REAL — one such id
    landed as ``Inf`` in the production data_points table and broke every
    read of it. Dashes make the value unparseable as a number, so existing
    NUMERIC-affinity tables are safe without a migration. Reads accept both
    the dashed and the legacy undashed form.
    """

    impl = CHAR(36)
    cache_ok = True

    def load_dialect_impl(self, dialect: Dialect) -> TypeEngine[Any]:
        if dialect.name == "postgresql":
            return dialect.type_descriptor(UUID(as_uuid=True))
        return dialect.type_descriptor(CHAR(36))

    def process_bind_param(
        self, value: uuid.UUID | str | None, dialect: Dialect
    ) -> uuid.UUID | str | None:
        if value is None or dialect.name == "postgresql":
            return value
        if isinstance(value, uuid.UUID):
            return str(value)
        return str(uuid.UUID(str(value)))

    def process_result_value(
        self, value: Any, dialect: Dialect
    ) -> uuid.UUID | None:
        if value is None or isinstance(value, uuid.UUID):
            return value
        return uuid.UUID(str(value))


class Base(DeclarativeBase):
    pass


class DataPoint(Base):
    __tablename__ = "data_points"

    id: Mapped[uuid.UUID] = mapped_column(
        GUID(), primary_key=True, default=uuid.uuid4
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
        GUID(), primary_key=True, default=uuid.uuid4
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
        GUID(), primary_key=True, default=uuid.uuid4
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
    explanations: Mapped[list["Explanation"]] = relationship(
        back_populates="anomaly",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )
    expert_labels: Mapped[list["ExpertLabel"]] = relationship(
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
        GUID(), primary_key=True, default=uuid.uuid4
    )
    anomaly_id: Mapped[uuid.UUID] = mapped_column(
        GUID(),
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


class Explanation(Base):
    __tablename__ = "explanations"

    id: Mapped[uuid.UUID] = mapped_column(
        GUID(), primary_key=True, default=uuid.uuid4
    )
    anomaly_id: Mapped[uuid.UUID] = mapped_column(
        GUID(),
        ForeignKey("anomalies.id", ondelete="CASCADE"),
        nullable=False,
    )
    model_name: Mapped[str] = mapped_column(String(64), nullable=False)
    model_version: Mapped[str | None] = mapped_column(String(64), nullable=True)
    reasoning_steps_json: Mapped[dict] = mapped_column(JSON, nullable=False)
    final_narrative: Mapped[str] = mapped_column(Text, nullable=False)
    stated_confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    latency_ms: Mapped[float | None] = mapped_column(Float, nullable=True)
    prompt_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    completion_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    anomaly: Mapped["Anomaly"] = relationship(back_populates="explanations")
    claims: Mapped[list["Claim"]] = relationship(
        back_populates="explanation",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )

    __table_args__ = (
        Index("ix_explanations_anomaly_id", "anomaly_id"),
        Index("ix_explanations_model_name", "model_name"),
    )

    def __repr__(self) -> str:
        return (
            f"<Explanation anomaly={self.anomaly_id} model={self.model_name} "
            f"conf={self.stated_confidence}>"
        )


class Claim(Base):
    __tablename__ = "claims"

    id: Mapped[uuid.UUID] = mapped_column(
        GUID(), primary_key=True, default=uuid.uuid4
    )
    explanation_id: Mapped[uuid.UUID] = mapped_column(
        GUID(),
        ForeignKey("explanations.id", ondelete="CASCADE"),
        nullable=False,
    )
    step_index: Mapped[int] = mapped_column(Integer, nullable=False)
    claim_type: Mapped[str] = mapped_column(String(48), nullable=False)
    # Every taxonomy type the classifier matched, primary first — needed to
    # detect headline-type drain (a claim matching a headline type but routed
    # to a descriptive primary) without re-running classification.
    matched_types: Mapped[list | None] = mapped_column(JSON, nullable=True)
    claim_text: Mapped[str] = mapped_column(Text, nullable=False)
    cited_sources: Mapped[list | None] = mapped_column(JSON, nullable=True)
    # B11 citation-reporting dimension. Legacy pre-migration rows remain null;
    # every claim emitted by explain.py records one of the declared outcomes.
    citation_outcome: Mapped[str | None] = mapped_column(
        String(16), nullable=True
    )
    # Phase 1 — retrieval-grounded factuality check (validate.py)
    grounding_verdict: Mapped[str] = mapped_column(String(16), nullable=False)
    grounding_evidence_ref: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    # The claim asserts a causal relation (validate._CAUSAL_RE). Metadata for
    # the grounded x causal analysis cut, never a scoring gate.
    causal: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    skipped_phase2: Mapped[bool] = mapped_column(
        Boolean, default=False, nullable=False
    )
    # Phase 2 — cross-source corroboration scorer (corroboration.py)
    corroboration_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    evidence_n: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    per_source_verdicts: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    # Channel-collapsed verdicts behind corroboration_score/evidence_n; stored
    # so the ablation and reporting need not re-derive the channel grouping.
    per_channel_verdicts: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    partial_verifiability: Mapped[bool] = mapped_column(
        Boolean, default=False, nullable=False
    )
    # Explicit analysis exclusion, separate from claim type and Phase-1/2
    # verdicts. B15 uses this for underpowered SO2 concentration claims.
    quantitative_exclusion_reason: Mapped[str | None] = mapped_column(
        String(48), nullable=True
    )
    low_corroboration_flag: Mapped[bool] = mapped_column(
        Boolean, default=False, nullable=False
    )

    explanation: Mapped["Explanation"] = relationship(back_populates="claims")

    __table_args__ = (
        Index("ix_claims_explanation_id", "explanation_id"),
        Index("ix_claims_claim_type", "claim_type"),
        Index("ix_claims_grounding_verdict", "grounding_verdict"),
    )

    def __repr__(self) -> str:
        return (
            f"<Claim type={self.claim_type} step={self.step_index} "
            f"grounding={self.grounding_verdict} corr={self.corroboration_score}>"
        )


class ExpertLabel(Base):
    __tablename__ = "expert_labels"

    id: Mapped[uuid.UUID] = mapped_column(
        GUID(), primary_key=True, default=uuid.uuid4
    )
    anomaly_id: Mapped[uuid.UUID] = mapped_column(
        GUID(),
        ForeignKey("anomalies.id", ondelete="CASCADE"),
        nullable=False,
    )
    labeler: Mapped[str] = mapped_column(String(64), nullable=False)
    true_cause: Mapped[str | None] = mapped_column(Text, nullable=True)
    claim_validations_json: Mapped[list | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    anomaly: Mapped["Anomaly"] = relationship(back_populates="expert_labels")

    __table_args__ = (
        Index("ix_expert_labels_anomaly_id", "anomaly_id"),
        Index("ix_expert_labels_labeler", "labeler"),
    )

    def __repr__(self) -> str:
        return f"<ExpertLabel anomaly={self.anomaly_id} labeler={self.labeler}>"
