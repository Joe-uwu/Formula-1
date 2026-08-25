"""Walk forward through test races one at a time, in date order, simulating
the real prediction setting: predict race N from race N's own grid/qualifying
plus point-in-time features, log the prediction, move to race N+1.

Note on "a database containing nothing after race N-1": the point-in-time
feature queries in f1/features/queries.py already make it structurally
impossible for a race's features to read rows dated on/after that race,
regardless of what else is in the database. So no separate DB truncation
step is needed to enforce this — it's enforced by every query's window frame.
"""
import sys
from datetime import datetime
from pathlib import Path
from typing import Callable

import pandas as pd
from sqlalchemy import delete, select, update
from sqlalchemy.dialects.postgresql import insert as pg_insert

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from f1.db.session import engine
from f1.db.models import Prediction, Result


def write_predictions(pred_df: pd.DataFrame, model_version: str, config_hash: str, race_id: int) -> None:
    """Replace this (model_version, config_hash, race_id)'s predictions, so a
    rerun (backtest or a live race re-predicted after quali) updates exactly
    that race instead of wiping unrelated ones sharing the same model/config."""
    if pred_df.empty:
        return
    now = datetime.utcnow()
    rows = [dict(model_version=model_version, config_hash=config_hash,
                  race_id=race_id, driver_id=int(r.driver_id),
                  predicted_probability=float(r.predicted_probability),
                  predicted_rank=int(r.predicted_rank),
                  actual_position=(None if pd.isna(r.actual_position) else int(r.actual_position)),
                  created_at=now)
            for r in pred_df.itertuples()]
    table = Prediction.__table__
    with engine.begin() as conn:
        conn.execute(delete(table).where(
            table.c.model_version == model_version, table.c.config_hash == config_hash,
            table.c.race_id == race_id,
        ))
        conn.execute(pg_insert(table).values(rows))


def run(feature_df: pd.DataFrame, predict_fn: Callable[[pd.DataFrame], pd.DataFrame],
        model_version: str, config_hash: str, write_to_db: bool = True) -> pd.DataFrame:
    """feature_df: materialized rows for the test split, one row per (race, driver).
    predict_fn: race_df -> race_df with predicted_probability, predicted_rank columns added.
    Returns the concatenated predictions across all test races, race by race, in date order."""
    all_preds = []
    for race_id, race_df in sorted(feature_df.groupby("race_id"),
                                     key=lambda kv: kv[1]["as_of_date"].iloc[0]):
        pred_df = predict_fn(race_df)
        pred_df["actual_position"] = race_df["target_position"]
        pred_df = pred_df[["race_id", "driver_id", "predicted_probability",
                             "predicted_rank", "actual_position"]]
        if write_to_db:
            write_predictions(pred_df, model_version, config_hash, int(race_id))
        all_preds.append(pred_df)

    return pd.concat(all_preds, ignore_index=True) if all_preds else pd.DataFrame(
        columns=["race_id", "driver_id", "predicted_probability", "predicted_rank", "actual_position"])


def backfill_actuals(model_version: str, config_hash: str) -> int:
    """Fill in actual_position for predictions made before a race happened,
    now that its result has been ingested. Returns rows updated."""
    table = Prediction.__table__
    with engine.begin() as conn:
        result = conn.execute(
            update(table)
            .values(actual_position=Result.position)
            .where(
                table.c.model_version == model_version, table.c.config_hash == config_hash,
                table.c.actual_position.is_(None),
                Result.race_id == table.c.race_id, Result.driver_id == table.c.driver_id,
            )
        )
        return result.rowcount


def fetch_predictions(model_version: str, config_hash: str) -> pd.DataFrame:
    """Read predictions back from the DB — this is what metrics should be
    computed from, so every reported number is traceable to an audited row."""
    table = Prediction.__table__
    with engine.connect() as conn:
        return pd.read_sql(
            select(table.c.race_id, table.c.driver_id, table.c.predicted_probability,
                   table.c.predicted_rank, table.c.actual_position)
            .where(table.c.model_version == model_version, table.c.config_hash == config_hash),
            conn,
        )
