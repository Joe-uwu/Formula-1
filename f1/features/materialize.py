"""Materialize the training feature table once per split configuration.

Stored alongside the split parameters that produced it (`feature_sets.split_config`)
so a run is reproducible and the expensive per-race query set isn't recomputed
on every experiment. Call `get_or_build` — it reuses an existing feature_set
for an identical split_config instead of rebuilding.
"""
import hashlib
import json
import sys
import time
from datetime import date, datetime
from pathlib import Path

import pandas as pd
from sqlalchemy import select

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from f1.db.session import engine, Session
from f1.db.models import Race, FeatureSet, FeatureRow
from f1.features.queries import (
    driver_form, constructor_form, circuit_history, grid_and_qualifying,
    circuit_weather_history, qualifying_pace, teammate_quali_delta_history,
    constructor_reliability, circuit_overtaking_difficulty, constructor_season_pace,
)

# Entry 12's reverse ablation (development folds only) found six of the
# original features with no measurable contribution over grid+quali alone:
# driver_races_before, circuit_avg_finish, circuit_pass_rate_last5,
# hist_track_temp_avg, hist_wind_speed_avg, hist_rain_rate. Dropped.
#
# Entry 20's reverse ablation on the 9 new Stage 5 pace-magnitude features
# selected 5 of them by point estimate — but selected AND reported that
# score on the SAME development folds, which is selection bias. RETRACTED
# in Fix 3 (see REPORT.md): redone with nested selection (choose on
# 2017-2021 using a paired bootstrap CI that must exclude zero, then score
# the survivors fresh on held-out 2022-2023). Nothing survived. The honest
# feature set is batch-0 alone — 12 features, all pre-Stage-5.
#
# The 9 candidate columns (quali_gap_to_pole_norm, quali_gap_to_median_norm,
# teammate_quali_delta, teammate_quali_delta_avg5, grid_minus_quali_delta,
# constructor_dnf_rate_last10, circuit_overtaking_difficulty,
# circuit_overtaking_x_grid, constructor_season_pace_gap) are still computed
# by build_race_features below (cheap, and useful for future re-tests with
# more data) but excluded here.
FEATURE_COLUMNS = [
    "grid_position", "quali_position",
    "avg_quali_last5", "avg_finish_last5", "wins_last5",
    "avg_finish_last3", "wins_last3", "driver_points_cum",
    "constructor_points_cum", "constructor_wins_cum", "constructor_avg_finish_last3",
    "circuit_win_rate",
]


def _config_hash(split_config: dict) -> str:
    blob = json.dumps(split_config, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def build_race_features(conn, as_of: date) -> pd.DataFrame:
    """One row per driver for the given race date, all point-in-time-safe."""
    grid = grid_and_qualifying(conn, as_of)
    if grid.empty:
        return grid

    df = grid.merge(driver_form(conn, as_of), on="driver_id", how="left")
    df = df.merge(constructor_form(conn, as_of), on="constructor_id", how="left")
    df = df.merge(circuit_history(conn, as_of), on="driver_id", how="left")
    df = df.merge(qualifying_pace(conn, as_of), on="driver_id", how="left")
    df = df.merge(teammate_quali_delta_history(conn, as_of).drop(columns=["race_date"]),
                   on="driver_id", how="left")
    df = df.merge(constructor_reliability(conn, as_of).drop(columns=["race_date"]),
                   on="constructor_id", how="left")
    df["grid_minus_quali_delta"] = df["grid_position"] - df["quali_position"]

    circuit_ot = circuit_overtaking_difficulty(conn, as_of)
    df["circuit_overtaking_difficulty"] = circuit_ot["circuit_overtaking_difficulty"].iloc[0] if not circuit_ot.empty else None
    df["circuit_overtaking_x_grid"] = df["circuit_overtaking_difficulty"] * df["grid_position"]

    season_pace = constructor_season_pace(conn, as_of).drop(columns=["race_date"])
    df = df.merge(season_pace, on="constructor_id", how="left")

    for col in FEATURE_COLUMNS:
        if col not in df.columns:
            df[col] = None
    return df


def build_feature_table(split_config: dict) -> tuple[int, pd.DataFrame, dict]:
    """Runs the full query set once per race in [start_date, end_date), stores it,
    and returns (feature_set_id, dataframe, query_timings)."""
    start = date.fromisoformat(split_config["start_date"])
    end = date.fromisoformat(split_config["end_date"])

    with Session() as session:
        race_dates = session.execute(
            select(Race.date).where(Race.date >= start, Race.date < end).distinct().order_by(Race.date)
        ).scalars().all()

    t0 = time.perf_counter()
    per_race_seconds = []
    rows = []
    with engine.connect() as conn:
        for race_date in race_dates:
            rt0 = time.perf_counter()
            df = build_race_features(conn, race_date)
            per_race_seconds.append(time.perf_counter() - rt0)
            if df.empty:
                continue
            for _, r in df.iterrows():
                features = {c: (None if pd.isna(r[c]) else float(r[c])) for c in FEATURE_COLUMNS}
                rows.append(dict(
                    race_id=None,  # filled in below from race_date, once per race not per row
                    driver_id=int(r["driver_id"]),
                    as_of_date=race_date,
                    target_win=int(r["actual_position"] == 1) if pd.notna(r["actual_position"]) else 0,
                    target_position=int(r["actual_position"]) if pd.notna(r["actual_position"]) else None,
                    **features,
                ))
    build_seconds = time.perf_counter() - t0

    # race_id wasn't carried through build_race_features' driver-level merge output name
    # collisions; refetch it directly per as_of_date via a light join instead of threading
    # it through every merge above.
    with Session() as session:
        date_to_race_id = dict(session.execute(select(Race.date, Race.race_id)).all())
    for row in rows:
        row["race_id"] = date_to_race_id[row["as_of_date"]]

    with Session() as session:
        fs = FeatureSet(
            split_config=split_config,
            created_at=datetime.utcnow(),
            row_count=len(rows),
            build_seconds=build_seconds,
        )
        session.add(fs)
        session.flush()
        feature_set_id = fs.id
        for row in rows:
            db_row = {k: v for k, v in row.items() if k not in FEATURE_COLUMNS}
            db_row["features"] = {c: row[c] for c in FEATURE_COLUMNS}
            session.add(FeatureRow(feature_set_id=feature_set_id, **db_row))
        session.commit()

    timings = {
        "total_seconds": build_seconds,
        "races": len(race_dates),
        "avg_seconds_per_race": (sum(per_race_seconds) / len(per_race_seconds)) if per_race_seconds else 0,
        "max_seconds_per_race": max(per_race_seconds) if per_race_seconds else 0,
    }
    return feature_set_id, pd.DataFrame(rows), timings


def get_or_build(split_config: dict):
    """Reuse an existing materialization for this exact split_config if present."""
    with Session() as session:
        # JSON columns aren't equality-comparable in Postgres SQL; compare in Python.
        existing = next(
            (fs for fs in session.execute(select(FeatureSet)).scalars()
             if fs.split_config == split_config),
            None,
        )
        if existing:
            frs = session.execute(
                select(FeatureRow).where(FeatureRow.feature_set_id == existing.id)
            ).scalars().all()
            rows = [dict(race_id=f.race_id, driver_id=f.driver_id, as_of_date=f.as_of_date,
                          target_win=f.target_win, target_position=f.target_position, **f.features)
                    for f in frs]
            return existing.id, pd.DataFrame(rows), {"reused": True}

    return build_feature_table(split_config)
