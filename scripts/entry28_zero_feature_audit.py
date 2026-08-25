"""Entry 28: null/distinct audit of the exact-zero-importance/delta features
flagged in Entries 11, 12, and 24. Read-only investigation, no model change.
Run once, logs one entry.
"""
import subprocess
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sqlalchemy import text
from f1.db.session import engine
from f1.features.queries import circuit_history, qualifying_pace, constructor_reliability, \
    circuit_overtaking_difficulty, circuit_weather_history
from f1 import report

TARGET_COLS = [
    "hist_track_temp_avg", "hist_wind_speed_avg", "hist_rain_rate",
    "circuit_avg_finish", "circuit_pass_rate_last5", "circuit_win_rate",
    "quali_gap_to_pole_norm", "grid_minus_quali_delta",
    "constructor_dnf_rate_last10", "circuit_overtaking_difficulty",
]


def git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]).decode().strip()
    except Exception:
        return "unknown"


def _grid_minus_quali(conn, race_dates):
    import pandas as pd
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


def _pool(conn, race_dates):
    import pandas as pd
    frames = {c: [] for c in TARGET_COLS}
    for d in race_dates:
        ch = circuit_history(conn, d)
        qp = qualifying_pace(conn, d)
        cr = constructor_reliability(conn, d)
        co = circuit_overtaking_difficulty(conn, d)
        cw = circuit_weather_history(conn, d)
        for col, src in [
            ("circuit_win_rate", ch), ("circuit_avg_finish", ch), ("circuit_pass_rate_last5", ch),
            ("quali_gap_to_pole_norm", qp),
            ("constructor_dnf_rate_last10", cr),
            ("circuit_overtaking_difficulty", co),
            ("hist_track_temp_avg", cw), ("hist_wind_speed_avg", cw), ("hist_rain_rate", cw),
        ]:
            if col in src.columns:
                frames[col].append(src[col])
    pooled = {c: (pd.concat(v, ignore_index=True) if v else pd.Series([], dtype="float64"))
              for c, v in frames.items()}
    pooled["grid_minus_quali_delta"] = _grid_minus_quali(conn, race_dates)
    return pooled


def _stats_rows(pooled):
    rows = []
    for col in TARGET_COLS:
        s = pooled[col]
        n = len(s)
        n_null = int(s.isna().sum())
        n_distinct = int(s.dropna().nunique())
        pct = (n_null / n * 100) if n else 0.0
        rows.append((col, n, n_null, pct, n_distinct))
    return rows


def _table_md(rows, title):
    lines = [f"**{title}:**\n", "| Feature | n | nulls | % null | distinct values |", "|---|---|---|---|---|"]
    for col, n, n_null, pct, n_distinct in rows:
        lines.append(f"| {col} | {n} | {n_null} | {pct:.1f}% | {n_distinct} |")
    return "\n".join(lines)


def main():
    with engine.connect() as conn:
        dev_dates = conn.execute(text(
            "SELECT date FROM races WHERE date >= '2017-01-01' AND date < '2024-01-01' ORDER BY date"
        )).scalars().all()
        full_dates = conn.execute(text(
            "SELECT date FROM races WHERE date >= '2015-01-01' AND date < '2025-12-31' ORDER BY date"
        )).scalars().all()
        holdout_2025 = conn.execute(text(
            "SELECT date FROM races WHERE date >= '2025-01-01' AND date < '2026-01-01' ORDER BY date"
        )).scalars().all()

        dev_rows = _stats_rows(_pool(conn, dev_dates))
        full_rows = _stats_rows(_pool(conn, full_dates))
        holdout_rows = _stats_rows(_pool(conn, holdout_2025))

        # circuit_id resolver check: does every 2025 race's circuit_id have zero prior races?
        circuit_check = conn.execute(text("""
            SELECT r25.race_id, r25.circuit_id, r25.name,
                   (SELECT count(*) FROM races prior WHERE prior.circuit_id = r25.circuit_id AND prior.date < r25.date) AS prior_races_same_circuit_id
            FROM races r25 WHERE r25.date >= '2025-01-01' ORDER BY r25.date
        """)).mappings().all()
        n_2025 = len(circuit_check)
        n_zero_prior = sum(1 for r in circuit_check if r["prior_races_same_circuit_id"] == 0)

    root_cause = (
        "**Root cause, by group:**\n\n"
        "1. **Dead code (3/10): hist_track_temp_avg, hist_wind_speed_avg, hist_rain_rate.** "
        "`circuit_weather_history()` is defined in f1/features/queries.py but `build_race_features()` in "
        "f1/features/materialize.py never calls it — these columns are unconditionally backfilled to `None` "
        "for every row, in every fold, always. Confirms the +0.0000/std-0.0000 readings in both Entry 11 and "
        "Entry 12: not uninformative, never fed real data at all. This is a fourth data-pipeline bug in the "
        "same family as Entries 7 and 13 (a silent wiring defect, not a crash).\n\n"
        "2. **Circuit-id cross-source resolution bug (4/10): circuit_avg_finish, circuit_pass_rate_last5, "
        "circuit_win_rate, circuit_overtaking_difficulty.** These have real, non-constant values in dev folds "
        "and the full table (see tables above). But f1/ingest/fastf1_results_ingest.py's circuit IdResolver "
        "is constructed with key `(r.name, r.location)` and then called with `(event['Location'], "
        "event['Location'])` — never matching an existing Kaggle circuit row (whose `name` and `location` "
        "differ), so every FastF1-sourced (2025) race mints a brand-new circuit_id with zero prior history. "
        f"Confirmed directly: all {n_2025} 2025 races checked, {n_zero_prior}/{n_2025} have zero prior rows "
        "under their assigned circuit_id. Every circuit-keyed feature is therefore 100% null for all 24 2025 "
        "races specifically — which is exactly the slice Entry 11's permutation importance was computed on "
        "(held out on 2025). This is a fifth data-pipeline bug in the same family as Entry 13: cross-source "
        "identity resolution failure, same mechanism (a natural-key mismatch between Kaggle and FastF1), "
        "different table. Entry 12's ablation ran on dev folds (2017-2023, 100% Kaggle-sourced) instead, "
        "which is why these four show real deltas there, not exact zeros.\n\n"
        "3. **Ablation-script wiring bug (3/10): quali_gap_to_pole_norm, grid_minus_quali_delta, "
        "constructor_dnf_rate_last10.** Real, populated data throughout — no data bug. "
        "scripts/fix3_nested_reverse_ablation.py's `score()` sets `drop_cols = [c for c in FEATURE_COLUMNS "
        "if c not in keep_cols]` and NaNs those out, intending that whatever remains reaches the model. But "
        "`WinModel.fit`/`predict_race` (f1/models/train.py) hardcode `df[FEATURE_COLUMNS]` — the module-level "
        "constant imported from materialize.py — ignoring `keep_cols` entirely. All four of Entry 24's "
        "exact-zero candidates (this group of three plus circuit_overtaking_difficulty, group 2 above) were "
        "already absent from FEATURE_COLUMNS before Fix 3 ran, so adding them to `keep_cols` was a no-op: the "
        "'with feature' and 'without feature' models were bit-identical, hence an exactly-zero, zero-width "
        "bootstrap CI on the log-loss delta — a measurement artifact of the ablation harness, not a null result."
    )

    metrics_md = "\n\n".join([
        _table_md(dev_rows, f"Dev folds, 2017-2023 pooled ({len(dev_dates)} races)"),
        _table_md(full_rows, f"Full feature-table range, 2015-2025 pooled ({len(full_dates)} races)"),
        _table_md(holdout_rows, f"2025 holdout slice only ({len(holdout_2025)} races) — the exact slice Entry 11's permutation importance was computed on"),
        root_cause,
    ])

    corrections = (
        "\n\n**Corrected readings:**\n\n"
        "- **Entry 24** (\"Selected 0/9 ... none\"): four of those nine — quali_gap_to_pole_norm, "
        "grid_minus_quali_delta, constructor_dnf_rate_last10, circuit_overtaking_difficulty — were never "
        "actually tested; the ablation harness silently never gave them to the model (bug 3 above). The "
        "other five (quali_gap_to_median_norm, teammate_quali_delta, teammate_quali_delta_avg5, "
        "circuit_overtaking_x_grid, constructor_season_pace_gap) were tested properly and legitimately failed "
        "to clear the CI-excludes-zero bar — that part of Entry 24's conclusion stands.\n"
        "- **Entry 23** (re-reading Entry 11's permutation importances against the noise floor): the "
        "\"within noise floor — indistinguishable from noise\" verdict for circuit_win_rate, "
        "hist_wind_speed_avg, hist_track_temp_avg, circuit_avg_finish, circuit_pass_rate_last5, and "
        "hist_rain_rate needs the same caveat — those six were measured on data that was structurally null "
        "(bugs 1 and 2 above), not noisy-but-real. \"Indistinguishable from noise\" is the wrong description; "
        "\"never measured\" is the right one. Entry 23's readings for the other features (driver_races_before, "
        "avg_finish_last5, wins_last5, driver_points_cum, wins_last3, constructor_avg_finish_last3, "
        "avg_finish_last3, constructor_points_cum) are unaffected — those columns are unrelated to circuit_id "
        "or weather and were fed real data."
    )
    metrics_md += corrections

    headline = (
        "3/10 permanently null (dead code, never wired in); 4/10 real everywhere except 100%-null on "
        "the 2025 FastF1 races (circuit-id resolver bug) — explains Entry 11's zeros; 3/10 real and "
        "populated throughout — Entry 24's zeros for these are an ablation-script wiring bug, not a "
        "null result. Full breakdown and root causes in the entry body."
    )

    verdict = "n/a (methodology-correction entry)"

    what_changed = (
        "Ran null-count / distinct-value-count checks directly against the underlying feature queries "
        "(bypassing FEATURE_COLUMNS, since two of the three mechanisms found here are wiring bugs, not "
        "literal all-null source data) for the ten columns Entries 11, 12, and 24 reported as exactly "
        "+0.0000 importance/delta with a zero-width CI or std. Checked dev folds (2017-2023), the full "
        "feature-table date range (2015-2025), and — since Entry 11's permutation importance ran on the "
        "2025 holdout specifically, not dev folds — that 2025 slice in isolation. Also checked whether "
        "every 2025 race's circuit_id links back to that track's pre-2025 (Kaggle) race history."
    )
    hypothesis = (
        "Bootstrap resampling cannot produce a zero-width interval on real data, so a reported "
        "+0.0000/[0,0] almost certainly means the underlying column was constant or absent over the rows "
        "actually measured — either because the data pipeline never populated it, or because a bug kept "
        "it out of the model being scored."
    )
    config_diff = "No split/model change. Read-only queries against f1/features/queries.py functions and the races/weather tables."
    next_steps = (
        "None — this phase is closing. If a future phase revisits Stage 5, fix f1/ingest/fastf1_results_ingest.py's "
        "circuit IdResolver key (should key get_or_create the same way the cache is keyed: (name, location), not "
        "(Location, Location)) and wire circuit_weather_history() into build_race_features(), before trusting any "
        "circuit- or weather-based feature's importance again. Do not rerun the locked holdout (Entry 27) — "
        "circuit_win_rate is in the final model's feature set and is null for all 24 2025 holdout races under this "
        "bug, but Entry 27 is a spent one-shot evaluation and this note is a record, not a rerun trigger."
    )

    report.append_entry(
        title=f"Entry {report.next_entry_number()} — Null/distinct audit of Entry 11/12/24's exact-zero features",
        commit_sha=git_sha(),
        what_changed=what_changed,
        hypothesis=hypothesis,
        config_diff=config_diff,
        metrics_table_md=metrics_md,
        headline=headline,
        verdict=verdict,
        next_steps=next_steps,
    )
    print("REPORT.md updated.")
    print(headline)


if __name__ == "__main__":
    main()
