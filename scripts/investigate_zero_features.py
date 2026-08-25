"""Entry 28 support: null/distinct counts for the ten exact-zero-importance
features (Entries 11, 12, 24), over dev-fold race dates (2017-2023) and the
full race table. Read-only — computes each feature's underlying query
directly, independent of FEATURE_COLUMNS, since circuit_weather_history is
never even called from build_race_features."""
import sys
from pathlib import Path

import pandas as pd
from sqlalchemy import text

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from f1.db.session import engine
from f1.features.queries import circuit_history, qualifying_pace, constructor_reliability, \
    circuit_overtaking_difficulty, circuit_weather_history

TARGET_COLS = [
    "hist_track_temp_avg", "hist_wind_speed_avg", "hist_rain_rate",
    "circuit_avg_finish", "circuit_pass_rate_last5", "circuit_win_rate",
    "quali_gap_to_pole_norm", "grid_minus_quali_delta",
    "constructor_dnf_rate_last10", "circuit_overtaking_difficulty",
]


def collect(conn, race_dates):
    frames = []
    for d in race_dates:
        ch = circuit_history(conn, d)
        qp = qualifying_pace(conn, d)
        cr = constructor_reliability(conn, d)
        co = circuit_overtaking_difficulty(conn, d)
        cw = circuit_weather_history(conn, d)

        row = {}
        # per-driver frames: circuit_history and qualifying_pace give one row per driver.
        # For a simple per-column null/distinct scan, pool all driver rows for the date.
        for col in ["circuit_win_rate", "circuit_avg_finish", "circuit_pass_rate_last5"]:
            if col in ch.columns:
                row[col] = ch[[col]].assign(race_date=d)
        for col in ["quali_gap_to_pole_norm"]:
            if col in qp.columns:
                row[col] = qp[[col]].assign(race_date=d)
        if "constructor_dnf_rate_last10" in cr.columns:
            row["constructor_dnf_rate_last10"] = cr[["constructor_dnf_rate_last10"]].assign(race_date=d)
        if "circuit_overtaking_difficulty" in co.columns:
            row["circuit_overtaking_difficulty"] = co[["circuit_overtaking_difficulty"]].assign(race_date=d)
        for col in ["hist_track_temp_avg", "hist_wind_speed_avg", "hist_rain_rate"]:
            if col in cw.columns:
                row[col] = cw[[col]].assign(race_date=d)
        # grid_minus_quali_delta needs grid_and_qualifying; compute inline (grid - quali) per driver
        frames.append(row)
    return frames


def grid_minus_quali(conn, race_dates):
    rows = []
    for d in race_dates:
        r = conn.execute(text("""
            SELECT res.grid AS grid_position, q.position AS quali_position
            FROM results res
            JOIN races ra ON ra.race_id = res.race_id
            LEFT JOIN qualifying q ON q.race_id = res.race_id AND q.driver_id = res.driver_id
            WHERE ra.date = :d
        """), {"d": d}).mappings().all()
        for x in r:
            gp, qp = x["grid_position"], x["quali_position"]
            rows.append((gp - qp) if (gp is not None and qp is not None) else None)
    return pd.Series(rows, dtype="float64")


def report(name, race_dates):
    print(f"\n=== {name}: {len(race_dates)} races ===")
    with engine.connect() as conn:
        frames = collect(conn, race_dates)
        pooled = {}
        for col in TARGET_COLS:
            if col == "grid_minus_quali_delta":
                continue
            series_list = [f[col][col] for f in frames if col in f]
            if not series_list:
                pooled[col] = pd.Series([], dtype="float64")
            else:
                pooled[col] = pd.concat(series_list, ignore_index=True)
        pooled["grid_minus_quali_delta"] = grid_minus_quali(conn, race_dates)

    for col in TARGET_COLS:
        s = pooled[col]
        n = len(s)
        n_null = int(s.isna().sum())
        n_distinct = int(s.dropna().nunique())
        print(f"{col:32s} n={n:5d}  nulls={n_null:5d} ({(n_null/n*100 if n else 0):5.1f}%)  distinct={n_distinct}")


def main():
    with engine.connect() as conn:
        dev_dates = conn.execute(text(
            "SELECT date FROM races WHERE date >= '2017-01-01' AND date < '2024-01-01' ORDER BY date"
        )).scalars().all()
        all_dates = conn.execute(text(
            "SELECT date FROM races WHERE date >= '2015-01-01' AND date < '2025-12-31' ORDER BY date"
        )).scalars().all()

    with engine.connect() as conn:
        holdout_2025_dates = conn.execute(text(
            "SELECT date FROM races WHERE date >= '2025-01-01' AND date < '2026-01-01' ORDER BY date"
        )).scalars().all()

    report("Dev folds (2017-2023)", dev_dates)
    report("Full feature table range (2015-2025)", all_dates)
    report("Entry 11 permutation-importance test slice (2025 holdout, 24 races)", holdout_2025_dates)


if __name__ == "__main__":
    main()
