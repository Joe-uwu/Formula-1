"""HTTP wrapper around the existing f1 prediction pipeline.

No prediction, feature, or metrics logic lives here — every endpoint is
plumbing over f1.live / f1.eval / f1.models functions that already exist and
are exercised by the scripts/ CLIs and the test suite. This file only shapes
their output as JSON.

Run locally:  uvicorn api.main:app --reload --port 8000
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import select

from f1.config import config_hash
from f1.db.models import Constructor, Driver, Prediction, Race, Result
from f1.db.session import engine
from f1.eval import walk_forward
from f1.eval.metrics import RaceLevelStats
from f1.features.materialize import FEATURE_COLUMNS
from f1.live.lineup import NoGridOrderError, next_unscored_race
from f1.live.predict_upcoming import LIVE_CONFIG_HASH, predict_race
from f1.models.train import WinModel

app = FastAPI(title="F1 Race Winner Prediction API")

# ponytail: wide open for local dev — no user accounts/auth in this project,
# and nothing here is sensitive. Narrow allow_origins before a public deploy.
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)

CONFIG_HASH = config_hash()
# The headline xgb_v1-vs-pole_sitter comparison (README's "pooled
# rolling-origin, 193 races") is logged under config_hash()+"_rolling" — see
# scripts/fix1_baselines_9folds.py — not the bare config_hash, which has no
# predictions logged for either model. That's where "the model" (the same
# WinModel used for live predictions) has real, non-cherry-picked scored
# history; LIVE_CONFIG_HASH covers whatever this app has predicted since.
ROLLING_CONFIG_HASH = CONFIG_HASH + "_rolling"


def _clean(v):
    """NaN/NaT -> None so FastAPI's JSON encoder doesn't choke on it."""
    return None if (v is None or (isinstance(v, float) and np.isnan(v))) else v


def _clean_int(v):
    """Nullable int column (e.g. actual_position, unset before a race runs) -> int or None."""
    return None if pd.isna(v) else int(v)


def _not_ready(name: str, year: int, round_no: int) -> dict:
    return {
        "status": "qualifying_not_done", "next_race": name, "year": year, "round": round_no,
        "message": f"Qualifying for {name} hasn't happened yet. Check back after quali.",
    }


def _predictions_payload(df: pd.DataFrame) -> list[dict]:
    return [
        dict(driver_id=int(r.driver_id), code=r.code, full_name=r.full_name, team=r.team,
             predicted_rank=int(r.predicted_rank), predicted_probability=float(r.predicted_probability))
        for r in df.itertuples()
    ]


@app.get("/races/upcoming")
def races_upcoming():
    year, round_no, name, _ = next_unscored_race()
    try:
        result = predict_race(year, round_no)
    except NoGridOrderError:
        return _not_ready(name, year, round_no)
    return {"status": "ok", "next_race": name, "year": year, "round": round_no,
            "predictions": _predictions_payload(result)}


@app.get("/races/upcoming/{driver_id}/features")
def upcoming_driver_features(driver_id: int):
    year, round_no, name, _ = next_unscored_race()
    try:
        result = predict_race(year, round_no)
    except NoGridOrderError:
        return _not_ready(name, year, round_no)
    row = result.loc[result["driver_id"] == driver_id]
    if row.empty:
        raise HTTPException(404, f"No driver {driver_id} in the {name} entry list.")
    r = row.iloc[0]
    return {
        "status": "ok", "next_race": name, "year": year, "round": round_no,
        "driver_id": driver_id, "code": r.code, "full_name": r.full_name, "team": r.team,
        "predicted_rank": int(r.predicted_rank), "predicted_probability": float(r.predicted_probability),
        "features": {col: _clean(float(r[col])) if pd.notna(r[col]) else None for col in FEATURE_COLUMNS},
    }


def _fetch_all_predictions(model_version: str) -> pd.DataFrame:
    """Predictions for `model_version` across every config_hash this app reads
    from — the backtest evaluation (through ~2025) plus whatever this app has
    itself predicted and scored live (2026+, see the season backfill) — so
    "most recent" reflects the actual freshest scored races, not just the
    backtest's cutoff."""
    frames = [
        walk_forward.fetch_predictions(model_version, ROLLING_CONFIG_HASH),
        walk_forward.fetch_predictions(model_version, LIVE_CONFIG_HASH),
    ]
    return pd.concat([f for f in frames if not f.empty], ignore_index=True)


@app.get("/track-record")
def track_record(n: int = 10):
    """Hit@1 / Hit@3 for the model vs. the pole-sitter baseline, over the
    most recently completed `n` races that both have logged predictions for."""
    model_preds = _fetch_all_predictions(WinModel.version)
    pole_preds = _fetch_all_predictions("pole_sitter")
    scored_ids = model_preds.dropna(subset=["actual_position"])["race_id"].unique()
    if len(scored_ids) == 0:
        raise HTTPException(404, "No scored races logged yet.")

    with engine.connect() as conn:
        dates = pd.read_sql(select(Race.race_id, Race.date), conn)
    recent_ids = (
        dates[dates["race_id"].isin(scored_ids)]
        .sort_values("date", ascending=False)
        .head(n)["race_id"]
    )

    model_stats = RaceLevelStats.from_predictions(model_preds[model_preds["race_id"].isin(recent_ids)])
    pole_stats = RaceLevelStats.from_predictions(pole_preds[pole_preds["race_id"].isin(recent_ids)])
    return {
        "n_races": int(len(recent_ids)),
        "model": {"model_version": WinModel.version,
                   "hit_at_1": model_stats.value("hit_at_1"), "hit_at_3": model_stats.value("hit_at_3")},
        "pole_sitter": {"hit_at_1": pole_stats.value("hit_at_1"), "hit_at_3": pole_stats.value("hit_at_3")},
    }


def _race_predictions_with_actuals(conn, race_id: int) -> pd.DataFrame:
    """Predictions for one race, joined to driver/team names and the actual
    result, preferring the backtest config_hash and falling back to the live
    one (a race predicted ahead of time and later scored lives there)."""
    for chash in (ROLLING_CONFIG_HASH, LIVE_CONFIG_HASH):
        df = pd.read_sql(
            select(
                Prediction.driver_id, Prediction.predicted_probability, Prediction.predicted_rank,
                Prediction.actual_position, Driver.code, Driver.forename, Driver.surname,
                Constructor.name.label("team"),
            )
            .select_from(Prediction)
            .join(Driver, Driver.driver_id == Prediction.driver_id)
            .outerjoin(Result, (Result.race_id == Prediction.race_id) & (Result.driver_id == Prediction.driver_id))
            .outerjoin(Constructor, Constructor.constructor_id == Result.constructor_id)
            .where(Prediction.race_id == race_id, Prediction.model_version == WinModel.version,
                    Prediction.config_hash == chash)
            .order_by(Prediction.predicted_rank),
            conn,
        )
        if not df.empty:
            df["full_name"] = df["forename"] + " " + df["surname"]
            return df.drop(columns=["forename", "surname"])
    return pd.DataFrame()


@app.get("/races/history")
def races_history(year: int, round: int):
    with engine.connect() as conn:
        race = conn.execute(
            select(Race.race_id, Race.name, Race.date).where(Race.year == year, Race.round == round)
        ).first()
        if race is None:
            raise HTTPException(404, f"No race for {year} round {round}.")
        df = _race_predictions_with_actuals(conn, race.race_id)
    if df.empty:
        raise HTTPException(404, f"No logged predictions for {year} round {round} yet.")
    predictions = [
        dict(driver_id=int(r.driver_id), code=r.code, full_name=r.full_name, team=r.team,
             predicted_rank=int(r.predicted_rank), predicted_probability=float(r.predicted_probability),
             actual_position=_clean_int(r.actual_position))
        for r in df.itertuples()
    ]
    return {"race": {"race_id": race.race_id, "year": year, "round": round, "name": race.name,
                       "date": str(race.date)},
            "predictions": predictions}


@app.get("/races/recent")
def races_recent(n: int = 8):
    """The n most recently completed races with a quick predicted-vs-actual
    summary — same race selection as /track-record — for a history feed."""
    model_preds = _fetch_all_predictions(WinModel.version)
    scored_ids = model_preds.dropna(subset=["actual_position"])["race_id"].unique()
    if len(scored_ids) == 0:
        return {"races": []}

    with engine.connect() as conn:
        races = pd.read_sql(
            select(Race.race_id, Race.name, Race.date, Race.year, Race.round)
            .where(Race.race_id.in_([int(x) for x in scored_ids]))
            .order_by(Race.date.desc())
            .limit(n),
            conn,
        )
        out = []
        for r in races.itertuples():
            df = _race_predictions_with_actuals(conn, r.race_id)
            if df.empty:
                continue
            top = df.iloc[0]
            winner_rows = df[df["actual_position"] == 1]
            winner = None if winner_rows.empty else winner_rows.iloc[0]
            out.append({
                "race_id": int(r.race_id), "name": r.name, "date": str(r.date),
                "year": int(r.year), "round": int(r.round),
                "top_pick": {"code": top.code, "full_name": top.full_name, "team": top.team,
                             "predicted_probability": float(top.predicted_probability)},
                "actual_winner": None if winner is None else {
                    "code": winner.code, "full_name": winner.full_name,
                    "predicted_rank": int(winner.predicted_rank),
                },
                "hit": bool(winner is not None and winner.code == top.code),
            })
    return {"races": out}


@app.get("/races/{race_id}/share-card")
def race_share_card(race_id: int):
    with engine.connect() as conn:
        race = conn.execute(
            select(Race.year, Race.round, Race.name, Race.date).where(Race.race_id == race_id)
        ).first()
        if race is None:
            raise HTTPException(404, f"No race with id {race_id}.")
        df = _race_predictions_with_actuals(conn, race_id)
    if df.empty:
        raise HTTPException(404, f"No logged predictions for race {race_id} yet.")

    def _driver_dict(r) -> dict:
        return dict(code=r.code, full_name=r.full_name, team=r.team,
                     predicted_rank=int(r.predicted_rank), predicted_probability=float(r.predicted_probability),
                     actual_position=_clean_int(r.actual_position))

    top_pick = df.iloc[0]
    winner_rows = df[df["actual_position"] == 1]
    return {
        "race": {"race_id": race_id, "year": race.year, "round": race.round, "name": race.name,
                   "date": str(race.date)},
        "top_pick": _driver_dict(top_pick),
        "actual_winner": _driver_dict(winner_rows.iloc[0]) if not winner_rows.empty else None,
        "field": [_driver_dict(r) for r in df.itertuples()],
    }
