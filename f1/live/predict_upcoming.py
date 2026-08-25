"""Predict a race that isn't in the materialized test set yet: resolve the
entry list and grid order live from FastF1 (requires qualifying to have
happened — see f1.live.lineup.NoGridOrderError), compute the same
point-in-time features used everywhere else, run the trained model, and log
the predictions so they can be scored once the race actually happens (see
backfill_and_score below).
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from f1.config import config_hash
from f1.db.session import engine
from f1.eval import walk_forward
from f1.features.materialize import FEATURE_COLUMNS
from f1.features.queries import driver_form, constructor_form, circuit_history, circuit_weather_history
from f1.live.lineup import ensure_race_and_lineup
from f1.models.train import WinModel

LIVE_CONFIG_HASH = config_hash() + "_live"


def predict_race(year: int, round_no: int) -> pd.DataFrame:
    race_id, as_of, lineup_df = ensure_race_and_lineup(year, round_no)

    with engine.connect() as conn:
        df = lineup_df.merge(driver_form(conn, as_of), on="driver_id", how="left")
        df = df.merge(constructor_form(conn, as_of), on="constructor_id", how="left")
        df = df.merge(circuit_history(conn, as_of), on="driver_id", how="left")
        weather = circuit_weather_history(conn, as_of)
        for col in ("hist_track_temp_avg", "hist_wind_speed_avg", "hist_rain_rate"):
            df[col] = weather[col].iloc[0] if not weather.empty else None

    for col in FEATURE_COLUMNS:
        if col not in df.columns:
            df[col] = None
    df["race_id"] = race_id
    df["actual_position"] = None  # unknown — the race hasn't happened yet

    model = WinModel.load(config_hash())
    result = model.predict_race(df)
    result = result.sort_values("predicted_probability", ascending=False).reset_index(drop=True)

    walk_forward.write_predictions(
        result[["race_id", "driver_id", "predicted_probability", "predicted_rank", "actual_position"]],
        model_version=model.version, config_hash=LIVE_CONFIG_HASH, race_id=race_id,
    )

    return result[["code", "full_name", "team", "predicted_rank", "predicted_probability"]]


def backfill_and_score(year: int, round_no: int) -> pd.DataFrame | None:
    """Call once the race has actually happened and its result is ingested
    (season backfill or re-running fastf1_results_ingest for this year):
    fills in actual_position on the logged live predictions."""
    model_version = WinModel.version
    updated = walk_forward.backfill_actuals(model_version, LIVE_CONFIG_HASH)
    if updated == 0:
        return None
    return walk_forward.fetch_predictions(model_version, LIVE_CONFIG_HASH)
