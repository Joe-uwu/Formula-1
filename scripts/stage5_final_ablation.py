"""Stage 5 close-out: reverse ablation on the 9 new Stage 5 columns
specifically (grid+quali+batch-0 baseline plus one Stage 5 feature at a
time), development folds only. Decides final inclusion under the ~20-feature
ceiling — drop anything that doesn't measurably beat the batch-0 baseline.
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
from scripts.stage5_features import BATCH_0, score

STAGE5_COLS = [
    "quali_gap_to_pole_norm", "quali_gap_to_median_norm",
    "teammate_quali_delta", "teammate_quali_delta_avg5",
    "grid_minus_quali_delta",
    "constructor_dnf_rate_last10",
    "circuit_overtaking_difficulty", "circuit_overtaking_x_grid",
    "constructor_season_pace_gap",
]


def git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]).decode().strip()
    except Exception:
        return "unknown"


def main():
    feature_set_id, feature_df, _ = get_or_build(SPLIT_CONFIG)
    feature_df["as_of_date"] = pd.to_datetime(feature_df["as_of_date"])

    baseline_stats = score(feature_df, BATCH_0, "stage5_ablation_base")
    baseline_ll = baseline_stats.value("log_loss")
    baseline_hit1 = baseline_stats.value("hit_at_1")

    rows = [f"Batch-0 baseline (12 features): log_loss={baseline_ll:.4f}, hit_at_1={baseline_hit1:.4f}\n",
             "| Feature added | Log loss | Δ log loss | Hit@1 | Δ Hit@1 | Verdict |",
             "|---|---|---|---|---|---|"]
    keep = []
    for feat in STAGE5_COLS:
        stats = score(feature_df, BATCH_0 + [feat], f"stage5_abl_{feat}"[:60])
        ll, hit1 = stats.value("log_loss"), stats.value("hit_at_1")
        d_ll, d_hit1 = ll - baseline_ll, hit1 - baseline_hit1
        alive = d_ll < -0.0005 or d_hit1 > 0.005
        if alive:
            keep.append(feat)
        rows.append(f"| {feat} | {ll:.4f} | {d_ll:+.4f} | {hit1:.4f} | {d_hit1:+.4f} | {'keep' if alive else 'drop'} |")
    table = "\n".join(rows)

    final_cols = BATCH_0 + keep
    final_stats = score(feature_df, final_cols, "stage5_final")
    final_ll, final_hit1 = final_stats.value("log_loss"), final_stats.value("hit_at_1")

    report.append_entry(
        title=f"Entry {report.next_entry_number()} — Stage 5 close-out: reverse ablation, final feature set",
        commit_sha=git_sha(),
        what_changed=f"Reverse ablation on the 9 Stage 5 candidates (development folds, batch-0 baseline + one feature at a time). Kept: {keep if keep else 'none'}.",
        hypothesis="Respect the ~20-feature ceiling for ~4700 training rows — drop anything that doesn't measurably beat the batch-0 baseline on its own.",
        config_diff=f"Final feature set ({len(final_cols)} columns): {final_cols}",
        metrics_table_md=table + f"\n\n**Final feature set score:** log_loss={final_ll:.4f} (vs batch-0 {baseline_ll:.4f}), hit_at_1={final_hit1:.4f} (vs batch-0 {baseline_hit1:.4f})",
        headline=f"Kept {len(keep)}/9 Stage 5 features. Final feature count: {len(final_cols)}.",
        verdict="n/a (feature-selection entry)",
        next_steps="Update f1/features/materialize.py's FEATURE_COLUMNS to the final set and proceed to Stage 6.",
    )
    print("REPORT.md updated.")
    print(table)
    print("keep:", keep)


if __name__ == "__main__":
    main()
