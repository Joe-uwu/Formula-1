from datetime import datetime, date
from sqlalchemy import (
    String, Integer, Float, Date, Boolean, ForeignKey, UniqueConstraint,
    Index, DateTime, JSON,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


def source_cols():
    """Every ingested table carries where it came from and when."""
    return dict(
        source=mapped_column(String, nullable=False),
        ingested_at=mapped_column(DateTime, nullable=False, default=datetime.utcnow),
    )


class Season(Base):
    __tablename__ = "seasons"
    year: Mapped[int] = mapped_column(Integer, primary_key=True)
    url: Mapped[str | None] = mapped_column(String)
    source: Mapped[str] = mapped_column(String, nullable=False)
    ingested_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=datetime.utcnow)


class Circuit(Base):
    __tablename__ = "circuits"
    circuit_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    ref: Mapped[str] = mapped_column(String, nullable=False)
    name: Mapped[str] = mapped_column(String, nullable=False)
    location: Mapped[str | None] = mapped_column(String)
    country: Mapped[str | None] = mapped_column(String)
    lat: Mapped[float | None] = mapped_column(Float)
    lng: Mapped[float | None] = mapped_column(Float)
    alt: Mapped[float | None] = mapped_column(Float)
    source: Mapped[str] = mapped_column(String, nullable=False)
    ingested_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=datetime.utcnow)


class Driver(Base):
    __tablename__ = "drivers"
    driver_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    ref: Mapped[str] = mapped_column(String, nullable=False)
    number: Mapped[int | None] = mapped_column(Integer)
    code: Mapped[str | None] = mapped_column(String)
    forename: Mapped[str] = mapped_column(String, nullable=False)
    surname: Mapped[str] = mapped_column(String, nullable=False)
    dob: Mapped[date | None] = mapped_column(Date)
    nationality: Mapped[str | None] = mapped_column(String)
    source: Mapped[str] = mapped_column(String, nullable=False)
    ingested_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=datetime.utcnow)


class Constructor(Base):
    __tablename__ = "constructors"
    constructor_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    ref: Mapped[str] = mapped_column(String, nullable=False)
    name: Mapped[str] = mapped_column(String, nullable=False)
    nationality: Mapped[str | None] = mapped_column(String)
    source: Mapped[str] = mapped_column(String, nullable=False)
    ingested_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=datetime.utcnow)


class Race(Base):
    __tablename__ = "races"
    race_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    year: Mapped[int] = mapped_column(ForeignKey("seasons.year"), nullable=False)
    round: Mapped[int] = mapped_column(Integer, nullable=False)
    circuit_id: Mapped[int] = mapped_column(ForeignKey("circuits.circuit_id"), nullable=False)
    name: Mapped[str] = mapped_column(String, nullable=False)
    date: Mapped[date] = mapped_column(Date, nullable=False)
    time: Mapped[str | None] = mapped_column(String)
    source: Mapped[str] = mapped_column(String, nullable=False)
    ingested_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=datetime.utcnow)

    __table_args__ = (Index("ix_races_date", "date"),)


class Result(Base):
    __tablename__ = "results"
    result_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    race_id: Mapped[int] = mapped_column(ForeignKey("races.race_id"), nullable=False)
    driver_id: Mapped[int] = mapped_column(ForeignKey("drivers.driver_id"), nullable=False)
    constructor_id: Mapped[int] = mapped_column(ForeignKey("constructors.constructor_id"), nullable=False)
    grid: Mapped[int | None] = mapped_column(Integer)
    position: Mapped[int | None] = mapped_column(Integer)  # null = DNF/DNS
    position_order: Mapped[int | None] = mapped_column(Integer)
    points: Mapped[float | None] = mapped_column(Float)
    laps: Mapped[int | None] = mapped_column(Integer)
    status: Mapped[str | None] = mapped_column(String)
    milliseconds: Mapped[int | None] = mapped_column(Integer)
    fastest_lap_rank: Mapped[int | None] = mapped_column(Integer)
    source: Mapped[str] = mapped_column(String, nullable=False)
    ingested_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=datetime.utcnow)

    __table_args__ = (
        UniqueConstraint("race_id", "driver_id", name="uq_results_race_driver"),
        Index("ix_results_race_id", "race_id"),
        Index("ix_results_driver_race", "driver_id", "race_id"),
    )


class Qualifying(Base):
    __tablename__ = "qualifying"
    qualify_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    race_id: Mapped[int] = mapped_column(ForeignKey("races.race_id"), nullable=False)
    driver_id: Mapped[int] = mapped_column(ForeignKey("drivers.driver_id"), nullable=False)
    constructor_id: Mapped[int] = mapped_column(ForeignKey("constructors.constructor_id"), nullable=False)
    position: Mapped[int | None] = mapped_column(Integer)
    q1: Mapped[str | None] = mapped_column(String)
    q2: Mapped[str | None] = mapped_column(String)
    q3: Mapped[str | None] = mapped_column(String)
    q1_seconds: Mapped[float | None] = mapped_column(Float)
    q2_seconds: Mapped[float | None] = mapped_column(Float)
    q3_seconds: Mapped[float | None] = mapped_column(Float)
    source: Mapped[str] = mapped_column(String, nullable=False)
    ingested_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=datetime.utcnow)

    __table_args__ = (
        UniqueConstraint("race_id", "driver_id", name="uq_qualifying_race_driver"),
        Index("ix_qualifying_race_id", "race_id"),
    )


class Weather(Base):
    __tablename__ = "weather"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    race_id: Mapped[int] = mapped_column(ForeignKey("races.race_id"), nullable=False, unique=True)
    air_temp_avg: Mapped[float | None] = mapped_column(Float)
    track_temp_avg: Mapped[float | None] = mapped_column(Float)
    humidity_avg: Mapped[float | None] = mapped_column(Float)
    wind_speed_avg: Mapped[float | None] = mapped_column(Float)
    rainfall: Mapped[bool | None] = mapped_column(Boolean)
    source: Mapped[str] = mapped_column(String, nullable=False)
    ingested_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=datetime.utcnow)


class FeatureSet(Base):
    """One materialization of the training feature table for a given split config."""
    __tablename__ = "feature_sets"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    split_config: Mapped[dict] = mapped_column(JSON, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=datetime.utcnow)
    row_count: Mapped[int] = mapped_column(Integer, nullable=False)
    build_seconds: Mapped[float] = mapped_column(Float, nullable=False)


class FeatureRow(Base):
    __tablename__ = "feature_rows"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    feature_set_id: Mapped[int] = mapped_column(ForeignKey("feature_sets.id"), nullable=False)
    race_id: Mapped[int] = mapped_column(ForeignKey("races.race_id"), nullable=False)
    driver_id: Mapped[int] = mapped_column(ForeignKey("drivers.driver_id"), nullable=False)
    as_of_date: Mapped[date] = mapped_column(Date, nullable=False)
    features: Mapped[dict] = mapped_column(JSON, nullable=False)
    target_win: Mapped[int] = mapped_column(Integer, nullable=False)
    target_position: Mapped[int | None] = mapped_column(Integer)

    __table_args__ = (Index("ix_feature_rows_set_race", "feature_set_id", "race_id"),)


class Prediction(Base):
    __tablename__ = "predictions"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    model_version: Mapped[str] = mapped_column(String, nullable=False)
    config_hash: Mapped[str] = mapped_column(String, nullable=False)
    race_id: Mapped[int] = mapped_column(ForeignKey("races.race_id"), nullable=False)
    driver_id: Mapped[int] = mapped_column(ForeignKey("drivers.driver_id"), nullable=False)
    predicted_probability: Mapped[float] = mapped_column(Float, nullable=False)
    predicted_rank: Mapped[int] = mapped_column(Integer, nullable=False)
    actual_position: Mapped[int | None] = mapped_column(Integer)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=datetime.utcnow)

    __table_args__ = (
        Index("ix_predictions_model_race", "model_version", "config_hash", "race_id"),
    )
