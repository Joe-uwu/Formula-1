"""Fix 3: Entry 20's reverse ablation selected features by point estimate on
the development folds, then reported the SAME selected set's combined score
on those SAME folds — classic selection bias (the reported 0.5793 Hit@1 was
both selected on and evaluated on the same data). Retracted.

Honest version: nested selection.
- SELECTION folds: 2017-2021 (5 dev folds). Each Stage-5 candidate is scored
  against the batch-0 baseline with a PAIRED bootstrap CI on the log-loss
  delta (race-level resampling). Kept only if the CI excludes zero (real,
  not point-estimate-only).
- VALIDATION folds: 2022-2023 (2 dev folds, held out from selection). The
  FINAL selected set (batch-0 + whatever survived) is scored here, once,
  with its own paired CI against batch-0 — this is the honest number that
  replaces 0.5793. If nothing survives selection, report that and keep
  batch-0 alone; no fallback fudging.
Neither split touches the locked holdout (2024-2025).
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
from f1.eval.rolling_origin import run_rolling_origin
from f1.models.train import WinModel
from f1 import report

SELECTION_FOLDS = range(2017, 2022)   # test years 2017-2021
VALIDATION_FOLDS = range(2022, 2024)  # test years 2022-2023 — held out from selection

BATCH_0 = [
    "grid_position", "quali_position", "avg_quali_last5", "avg_finish_last5", "wins_last5",
    "avg_finish_last3", "wins_last3", "driver_points_cum",
    "constructor_points_cum", "constructor_wins_cum", "constructor_avg_finish_last3",
    "circuit_win_rate",
]
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


def score(feature_df: pd.DataFrame, keep_cols: list[str], version: str, test_years) -> M.RaceLevelStats:
    df = feature_df.copy()
    drop_cols = [c for c in FEATURE_COLUMNS if c not in keep_cols]
    df[drop_cols] = float("nan")
    preds = run_rolling_origin(df, lambda: WinModel(), model_version=version,
                                 config_hash="fix3_nested", test_years=test_years)
    return M.RaceLevelStats.from_predictions(preds)


def main():
    feature_set_id, feature_df, _ = get_or_build(SPLIT_CONFIG)
    feature_df["as_of_date"] = pd.to_datetime(feature_df["as_of_date"])

    # --- selection phase: SELECTION_FOLDS only ---
    sel_baseline = score(feature_df, BATCH_0, "fix3_sel_base", SELECTION_FOLDS)
    sel_rows = ["**Selection phase (2017-2021, paired bootstrap CI on log-loss delta vs batch-0):**\n",
                 "| Feature | Δ log loss | 95% CI | Selected |", "|---|---|---|---|"]
    selected = []
    for feat in STAGE5_COLS:
        stats = score(feature_df, BATCH_0 + [feat], f"fix3_sel_{feat}"[:60], SELECTION_FOLDS)
        point, dlo, dhi = M.paired_bootstrap_ci(stats, sel_baseline, "log_loss")
        # log_loss: lower is better, so "excludes zero" in the improving direction means dhi < 0
        excludes_zero = dhi < 0 or dlo > 0
        keep = dhi < 0  # CI entirely on the "log loss went down" side
        if keep:
            selected.append(feat)
        sel_rows.append(f"| {feat} | {point:+.4f} | [{dlo:+.4f}, {dhi:+.4f}] | {'yes' if keep else 'no'} |")
    selection_table = "\n".join(sel_rows)

    # --- validation phase: VALIDATION_FOLDS only, selected set vs batch-0, evaluated fresh ---
    val_baseline = score(feature_df, BATCH_0, "fix3_val_base", VALIDATION_FOLDS)
    final_cols = BATCH_0 + selected
    val_final = score(feature_df, final_cols, "fix3_val_final", VALIDATION_FOLDS)
    point, dlo, dhi = M.paired_bootstrap_ci(val_final, val_baseline, "hit_at_1")
    ll_point, ll_dlo, ll_dhi = M.paired_bootstrap_ci(val_final, val_baseline, "log_loss")

    val_table = (
        f"\n**Validation phase (2022-2023, held out from selection) — honest generalization estimate:**\n\n"
        f"batch-0 alone: hit_at_1={val_baseline.value('hit_at_1'):.4f}, log_loss={val_baseline.value('log_loss'):.4f}\n\n"
        f"final set ({'batch-0 + ' + str(selected) if selected else 'batch-0 only, nothing survived selection'}): "
        f"hit_at_1={val_final.value('hit_at_1'):.4f}, log_loss={val_final.value('log_loss'):.4f}\n\n"
        f"Δ hit_at_1 = {point:+.4f} [{dlo:+.4f}, {dhi:+.4f}] ({M.verdict(dlo, dhi)})\n\n"
        f"Δ log_loss = {ll_point:+.4f} [{ll_dlo:+.4f}, {ll_dhi:+.4f}] ({M.verdict(ll_dlo, ll_dhi, metric='log_loss')})"
    )

    headline = (f"Selected {len(selected)}/9 on selection folds (CI excludes zero): {selected if selected else 'none'}. "
                 f"On held-out validation folds: Δhit@1={point:+.4f} [{dlo:+.4f},{dhi:+.4f}].")

    report.append_entry(
        title=f"Entry {report.next_entry_number()} — Fix 3: nested reverse ablation, retracts Entry 20's 0.5793",
        commit_sha=git_sha(),
        what_changed=(
            "RETRACTS Entry 20's headline figure (hit_at_1=0.5793): it was selected on and reported on the "
            "same development folds, which is selection bias, not a measured result. Redone with nested "
            "selection: features chosen on 2017-2021 using paired bootstrap CIs on the log-loss delta (kept "
            "only if the CI excludes zero, i.e. reliably better, not just a better point estimate), then the "
            "selected set is scored fresh on 2022-2023 (held out from selection) for the honest number."
        ),
        hypothesis="A feature set chosen by looking at a metric should never be reported on that same metric/data — nested selection is the fix.",
        config_diff=f"selection_folds={list(SELECTION_FOLDS)}, validation_folds={list(VALIDATION_FOLDS)} (both within development folds; locked holdout 2024-2025 untouched)",
        metrics_table_md=selection_table + "\n" + val_table,
        headline=headline,
        verdict=M.verdict(dlo, dhi),
        next_steps="Stage 6 (skip Stage 7 per instruction): PyTorch conditional logit / Plackett-Luce.",
    )
    print("REPORT.md updated.")
    print(selection_table)
    print(val_table)


if __name__ == "__main__":
    main()
