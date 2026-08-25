"""Stage 5: pace-magnitude features, added in small batches, each measured
on DEVELOPMENT FOLDS ONLY (2017-2023) so a metric change can be attributed
to a specific batch and the locked holdout (2024-2025) stays untouched.

Each batch is cumulative (batch N includes everything from batch < N).
"BATCH_0" is the surviving 12 features from before Stage 5 (after Entry 12
dropped the 6 dead ones). A batch's own columns are added to whatever the
previous batch already had; columns not yet "unlocked" are set to NaN so
the model can't see them, using the same ablation trick as Entry 11/12.
"""
import subprocess
import sys

sys.stdout.reconfigure(encoding="utf-8")
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from f1.config import SPLIT_CONFIG
from f1.features.materialize import get_or_build, FEATURE_COLUMNS
from f1.eval import metrics as M
from f1.eval.rolling_origin import run_rolling_origin, DEV_TEST_YEARS
from f1.models.train import WinModel
from f1 import report

BATCH_0 = [
    "grid_position", "quali_position", "avg_quali_last5", "avg_finish_last5", "wins_last5",
    "avg_finish_last3", "wins_last3", "driver_points_cum",
    "constructor_points_cum", "constructor_wins_cum", "constructor_avg_finish_last3",
    "circuit_win_rate",
]

BATCHES = [
    ("Batch 1: qualifying pace gap (gap to pole, gap to median, both z-scored within session)",
     ["quali_gap_to_pole_norm", "quali_gap_to_median_norm"]),
    ("Batch 2: teammate qualifying delta (this race + trailing-5 average)",
     ["teammate_quali_delta", "teammate_quali_delta_avg5"]),
    ("Batch 3: grid-minus-qualifying delta (penalty signal)",
     ["grid_minus_quali_delta"]),
    ("Batch 4: constructor reliability (trailing DNF rate, last 10 races)",
     ["constructor_dnf_rate_last10"]),
    ("Batch 5: circuit overtaking difficulty + interaction with grid position",
     ["circuit_overtaking_difficulty", "circuit_overtaking_x_grid"]),
    ("Batch 6: season-relative constructor pace (rolling qualifying gap-to-pole, current season)",
     ["constructor_season_pace_gap"]),
]


def git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]).decode().strip()
    except Exception:
        return "unknown"


def score(feature_df: pd.DataFrame, keep_cols: list[str], version: str) -> M.RaceLevelStats:
    df = feature_df.copy()
    drop_cols = [c for c in FEATURE_COLUMNS if c not in keep_cols]
    df[drop_cols] = float("nan")
    preds = run_rolling_origin(df, lambda: WinModel(), model_version=version,
                                 config_hash="stage5_dev", test_years=DEV_TEST_YEARS)
    return M.RaceLevelStats.from_predictions(preds)


def main():
    feature_set_id, feature_df, _ = get_or_build(SPLIT_CONFIG)
    feature_df["as_of_date"] = pd.to_datetime(feature_df["as_of_date"])

    cumulative_cols = list(BATCH_0)
    prev_stats = score(feature_df, cumulative_cols, "stage5_batch0")
    prev_metrics = {name: prev_stats.value(name) for name in M.HEADLINE_METRIC_NAMES}

    for title, new_cols in BATCHES:
        cumulative_cols = cumulative_cols + new_cols
        version = f"stage5_{'_'.join(new_cols)}"[:60]
        stats = score(feature_df, cumulative_cols, version)

        rows = ["| Metric | Before this batch | After this batch | Δ | Verdict |", "|---|---|---|---|---|"]
        for name in M.HEADLINE_METRIC_NAMES:
            before = prev_metrics[name]
            after = stats.value(name)
            _, dlo, dhi = M.paired_bootstrap_ci(stats, prev_stats, name)
            v = M.verdict(dlo, dhi, metric=name)
            rows.append(f"| {name} | {before:.4f} | {after:.4f} | {after - before:+.4f} | {v} |")
        table = "\n".join(rows)

        report.append_entry(
            title=f"Entry {report.next_entry_number()} — Stage 5, {title}",
            commit_sha=git_sha(),
            what_changed=f"Added columns: {', '.join(new_cols)}. Cumulative feature set now: {cumulative_cols}",
            hypothesis=f"Measured on development folds only (2017-2023, {len(stats.race_ids)} races) so the locked holdout stays untouched.",
            config_diff=f"feature_columns += {new_cols}",
            metrics_table_md=table,
            headline=f"log_loss: {prev_metrics['log_loss']:.4f} -> {stats.value('log_loss'):.4f}, hit_at_1: {prev_metrics['hit_at_1']:.4f} -> {stats.value('hit_at_1'):.4f}",
            verdict=M.verdict(*M.paired_bootstrap_ci(stats, prev_stats, "log_loss")[1:], metric="log_loss"),
            next_steps="Next batch." if (title, new_cols) != BATCHES[-1] else "All batches added — run the final reverse-ablation pass to decide what stays, respecting the ~20-feature ceiling.",
        )
        print(f"{title}: {table}")
        prev_stats = stats
        prev_metrics = {name: stats.value(name) for name in M.HEADLINE_METRIC_NAMES}

    print("REPORT.md updated with all Stage 5 batches.")


if __name__ == "__main__":
    main()
