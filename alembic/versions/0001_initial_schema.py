"""initial schema

Revision ID: 0001
Revises:
Create Date: 2026-08-24
"""
from alembic import op
import sqlalchemy as sa

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None


def source_cols():
    return [
        sa.Column("source", sa.String, nullable=False),
        sa.Column("ingested_at", sa.DateTime, nullable=False),
    ]


def upgrade() -> None:
    op.create_table(
        "seasons",
        sa.Column("year", sa.Integer, primary_key=True),
        sa.Column("url", sa.String),
        *source_cols(),
    )

    op.create_table(
        "circuits",
        sa.Column("circuit_id", sa.Integer, primary_key=True),
        sa.Column("ref", sa.String, nullable=False),
        sa.Column("name", sa.String, nullable=False),
        sa.Column("location", sa.String),
        sa.Column("country", sa.String),
        sa.Column("lat", sa.Float),
        sa.Column("lng", sa.Float),
        sa.Column("alt", sa.Float),
        *source_cols(),
    )

    op.create_table(
        "drivers",
        sa.Column("driver_id", sa.Integer, primary_key=True),
        sa.Column("ref", sa.String, nullable=False),
        sa.Column("number", sa.Integer),
        sa.Column("code", sa.String),
        sa.Column("forename", sa.String, nullable=False),
        sa.Column("surname", sa.String, nullable=False),
        sa.Column("dob", sa.Date),
        sa.Column("nationality", sa.String),
        *source_cols(),
    )

    op.create_table(
        "constructors",
        sa.Column("constructor_id", sa.Integer, primary_key=True),
        sa.Column("ref", sa.String, nullable=False),
        sa.Column("name", sa.String, nullable=False),
        sa.Column("nationality", sa.String),
        *source_cols(),
    )

    op.create_table(
        "races",
        sa.Column("race_id", sa.Integer, primary_key=True),
        sa.Column("year", sa.Integer, sa.ForeignKey("seasons.year"), nullable=False),
        sa.Column("round", sa.Integer, nullable=False),
        sa.Column("circuit_id", sa.Integer, sa.ForeignKey("circuits.circuit_id"), nullable=False),
        sa.Column("name", sa.String, nullable=False),
        sa.Column("date", sa.Date, nullable=False),
        sa.Column("time", sa.String),
        *source_cols(),
    )
    op.create_index("ix_races_date", "races", ["date"])

    op.create_table(
        "results",
        sa.Column("result_id", sa.Integer, primary_key=True),
        sa.Column("race_id", sa.Integer, sa.ForeignKey("races.race_id"), nullable=False),
        sa.Column("driver_id", sa.Integer, sa.ForeignKey("drivers.driver_id"), nullable=False),
        sa.Column("constructor_id", sa.Integer, sa.ForeignKey("constructors.constructor_id"), nullable=False),
        sa.Column("grid", sa.Integer),
        sa.Column("position", sa.Integer),
        sa.Column("position_order", sa.Integer),
        sa.Column("points", sa.Float),
        sa.Column("laps", sa.Integer),
        sa.Column("status", sa.String),
        sa.Column("milliseconds", sa.Integer),
        sa.Column("fastest_lap_rank", sa.Integer),
        *source_cols(),
        sa.UniqueConstraint("race_id", "driver_id", name="uq_results_race_driver"),
    )
    op.create_index("ix_results_race_id", "results", ["race_id"])
    op.create_index("ix_results_driver_race", "results", ["driver_id", "race_id"])

    op.create_table(
        "qualifying",
        sa.Column("qualify_id", sa.Integer, primary_key=True),
        sa.Column("race_id", sa.Integer, sa.ForeignKey("races.race_id"), nullable=False),
        sa.Column("driver_id", sa.Integer, sa.ForeignKey("drivers.driver_id"), nullable=False),
        sa.Column("constructor_id", sa.Integer, sa.ForeignKey("constructors.constructor_id"), nullable=False),
        sa.Column("position", sa.Integer),
        sa.Column("q1", sa.String),
        sa.Column("q2", sa.String),
        sa.Column("q3", sa.String),
        *source_cols(),
        sa.UniqueConstraint("race_id", "driver_id", name="uq_qualifying_race_driver"),
    )
    op.create_index("ix_qualifying_race_id", "qualifying", ["race_id"])

    op.create_table(
        "weather",
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column("race_id", sa.Integer, sa.ForeignKey("races.race_id"), nullable=False, unique=True),
        sa.Column("air_temp_avg", sa.Float),
        sa.Column("track_temp_avg", sa.Float),
        sa.Column("humidity_avg", sa.Float),
        sa.Column("wind_speed_avg", sa.Float),
        sa.Column("rainfall", sa.Boolean),
        *source_cols(),
    )

    op.create_table(
        "feature_sets",
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column("split_config", sa.JSON, nullable=False),
        sa.Column("created_at", sa.DateTime, nullable=False),
        sa.Column("row_count", sa.Integer, nullable=False),
        sa.Column("build_seconds", sa.Float, nullable=False),
    )

    op.create_table(
        "feature_rows",
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column("feature_set_id", sa.Integer, sa.ForeignKey("feature_sets.id"), nullable=False),
        sa.Column("race_id", sa.Integer, sa.ForeignKey("races.race_id"), nullable=False),
        sa.Column("driver_id", sa.Integer, sa.ForeignKey("drivers.driver_id"), nullable=False),
        sa.Column("as_of_date", sa.Date, nullable=False),
        sa.Column("features", sa.JSON, nullable=False),
        sa.Column("target_win", sa.Integer, nullable=False),
        sa.Column("target_position", sa.Integer),
    )
    op.create_index("ix_feature_rows_set_race", "feature_rows", ["feature_set_id", "race_id"])

    op.create_table(
        "predictions",
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column("model_version", sa.String, nullable=False),
        sa.Column("config_hash", sa.String, nullable=False),
        sa.Column("race_id", sa.Integer, sa.ForeignKey("races.race_id"), nullable=False),
        sa.Column("driver_id", sa.Integer, sa.ForeignKey("drivers.driver_id"), nullable=False),
        sa.Column("predicted_probability", sa.Float, nullable=False),
        sa.Column("predicted_rank", sa.Integer, nullable=False),
        sa.Column("actual_position", sa.Integer),
        sa.Column("created_at", sa.DateTime, nullable=False),
    )
    op.create_index("ix_predictions_model_race", "predictions", ["model_version", "config_hash", "race_id"])


def downgrade() -> None:
    op.drop_table("predictions")
    op.drop_table("feature_rows")
    op.drop_table("feature_sets")
    op.drop_table("weather")
    op.drop_table("qualifying")
    op.drop_table("results")
    op.drop_table("races")
    op.drop_table("constructors")
    op.drop_table("drivers")
    op.drop_table("circuits")
    op.drop_table("seasons")
