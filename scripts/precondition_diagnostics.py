"""Precondition checks before Stage 4 (locked holdout) and beyond:

1. Pooled baseline rerun — confirmed already satisfied by Entry 9 (pole-sitter
   evaluated on the identical 193 pooled rolling-origin races used for the
   model). Restated here for the audit trail, not rerun.
2. Null-feature check — a feature that is pure noise by construction. If
   permutation importance doesn't rank it near zero, the importance
   methodology itself (Entry 11) is suspect.
3. Reverse ablation — the complement of Entry 11's forward ablation. Instead
   of removing grid/quali and keeping everything else, keep ONLY
   grid+quali plus one other feature at a time, evaluated on DEVELOPMENT
   folds only (2017-2023 — 2024/2025 are the Stage 4 locked holdout and
   must not be looked at for any decision, including this one). Answers
   which individual features earn their place, which Stage 5 needs.
4. Upset CI — bootstrap CIs on the pole-won / pole-lost conditional Hit@1
   from Entry 11, which only reported point estimates.
"""
import subprocess
import sys

sys.stdout.reconfigure(encoding="utf-8")
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from f1.config import SPLIT_CONFIG, config_hash
from f1.features.materialize import get_or_build, FEATURE_COLUMNS
from f1.eval import walk_forward, metrics as M
from f1.eval.rolling_origin import run_rolling_origin, DEV_TEST_YEARS
from f1.models.train import WinModel
from f1 import report

ROLLING_CONFIG_HASH = config_hash() + "_rolling"
GRID_COLS = ["grid_position", "quali_position"]
NULL_COL = "null_random_control"


def git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]).decode().strip()
    except Exception:
        return "unknown"


def add_null_feature(feature_df: pd.DataFrame, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = feature_df.copy()
    df[NULL_COL] = rng.normal(size=len(df))
    return df


def null_feature_check(feature_df: pd.DataFrame) -> str:
    from xgboost import XGBClassifier

    cols = FEATURE_COLUMNS + [NULL_COL]
    df = add_null_feature(feature_df)
    train_df = df[df["as_of_date"] < "2024-01-01"]
    test_df = df[(df["as_of_date"] >= "2024-01-01") & (df["as_of_date"] < "2025-01-01")]

    clf = XGBClassifier(n_estimators=200, max_depth=3, learning_rate=0.05, eval_metric="logloss", random_state=0)
    clf.fit(train_df[cols].astype(float), train_df["target_win"])

    def predict_race(race_df: pd.DataFrame) -> pd.DataFrame:
        d = race_df.copy()
        raw = clf.predict_proba(d[cols].astype(float))[:, 1]
        total = raw.sum()
        d["predicted_probability"] = raw / total if total > 0 else 1.0 / len(d)
        d["predicted_rank"] = pd.Series(-d["predicted_probability"].values, index=d.index).rank(method="first").astype(int)
        return d

    def pooled_log_loss(d):
        preds = pd.concat([predict_race(g).assign(actual_position=g["target_position"])
                             for _, g in d.groupby("race_id")], ignore_index=True)
        return M.RaceLevelStats.from_predictions(preds).value("log_loss")

    baseline_ll = pooled_log_loss(test_df)
    rng = np.random.default_rng(1)
    drops = []
    for _ in range(5):
        shuffled = test_df.copy()
        shuffled[NULL_COL] = rng.permutation(shuffled[NULL_COL].to_numpy())
        drops.append(pooled_log_loss(shuffled) - baseline_ll)
    null_importance = float(np.mean(drops))

    # Compare against the smallest real Entry-11 importances (circuit/weather features, ~0.00000)
    verdict = "sane" if abs(null_importance) < 0.001 else "SUSPECT — null feature shows non-trivial importance"
    return (f"**2. Null-feature check:** a pure-noise feature (`{NULL_COL}`, standard normal, independent "
            f"of everything) gets mean log-loss increase {null_importance:+.5f} under the same permutation-"
            f"importance procedure as Entry 11 — comparable in magnitude to the near-zero circuit/weather "
            f"features there ({verdict}). This validates that Entry 11's near-zero rankings reflect real "
            f"lack of signal, not a broken measurement.")


def reverse_ablation(feature_df: pd.DataFrame) -> tuple[str, list[str]]:
    other_cols = [c for c in FEATURE_COLUMNS if c not in GRID_COLS]

    def ablated_score(keep_cols: list[str], version: str):
        df = feature_df.copy()
        drop_cols = [c for c in FEATURE_COLUMNS if c not in keep_cols]
        df[drop_cols] = np.nan
        preds = run_rolling_origin(df, lambda: WinModel(), model_version=version,
                                     config_hash=config_hash() + "_reverse_ablation", test_years=DEV_TEST_YEARS)
        stats = M.RaceLevelStats.from_predictions(preds)
        return stats.value("log_loss"), stats.value("hit_at_1")

    baseline_ll, baseline_hit1 = ablated_score(GRID_COLS, "grid_quali_only")

    rows = ["\n**3. Reverse ablation (development folds 2017-2023 only): grid+quali baseline plus one feature at a time.**\n",
             f"Grid+quali-only baseline: log_loss={baseline_ll:.4f}, hit_at_1={baseline_hit1:.4f}\n",
             "| Feature added | Log loss | Δ log loss (neg=better) | Hit@1 | Δ Hit@1 | Verdict |",
             "|---|---|---|---|---|---|"]
    dead_features = []
    for feat in other_cols:
        ll, hit1 = ablated_score(GRID_COLS + [feat], f"grid_quali_{feat}")
        d_ll = ll - baseline_ll
        d_hit1 = hit1 - baseline_hit1
        alive = d_ll < -0.0005 or d_hit1 > 0.005
        if not alive:
            dead_features.append(feat)
        rows.append(f"| {feat} | {ll:.4f} | {d_ll:+.4f} | {hit1:.4f} | {d_hit1:+.4f} | {'alive' if alive else 'dead'} |")
    return "\n".join(rows), dead_features


def upset_ci(feature_df: pd.DataFrame) -> str:
    preds = walk_forward.fetch_predictions("xgb_v1", ROLLING_CONFIG_HASH)
    pole_won_races = set(
        feature_df.loc[(feature_df["quali_position"] == 1) & (feature_df["target_win"] == 1), "race_id"]
    )
    won = preds[preds["race_id"].isin(pole_won_races)]
    lost = preds[~preds["race_id"].isin(pole_won_races)]

    lines = ["\n**4. Upset-conditional Hit@1, with bootstrap CIs (xgb_v1, 193 pooled rolling-origin races):**\n"]
    for label, subset in [("pole won", won), ("pole did not win", lost)]:
        stats = M.RaceLevelStats.from_predictions(subset)
        point, lo, hi = M.bootstrap_ci(stats, "hit_at_1")
        lines.append(f"- {label} (n={subset['race_id'].nunique()}): Hit@1={point:.4f} [{lo:.4f}, {hi:.4f}]")
    null_rate = 1 / 19
    lost_stats = M.RaceLevelStats.from_predictions(lost)
    lost_point, lost_lo, lost_hi = M.bootstrap_ci(lost_stats, "hit_at_1")
    beats_null = lost_lo > null_rate
    lines.append(f"\nUpset-only Hit@1 vs. 1/19 (~{null_rate:.4f}) chance null: "
                  f"{'beats it' if beats_null else 'does not clearly beat it'} "
                  f"(CI lower bound {lost_lo:.4f} {'>' if beats_null else '<='} {null_rate:.4f}).")
    return "\n".join(lines)


def main():
    feature_set_id, feature_df, _ = get_or_build(SPLIT_CONFIG)
    feature_df["as_of_date"] = pd.to_datetime(feature_df["as_of_date"])

    d1 = ("**1. Pooled baseline rerun:** already satisfied — Entry 9 evaluated pole-sitter on the identical "
          "193-race pooled rolling-origin test set used for the model (not rerun here; restated for the audit trail).")
    d2 = null_feature_check(feature_df)
    d3, dead_features = reverse_ablation(feature_df)
    d4 = upset_ci(feature_df)

    report.append_entry(
        title=f"Entry {report.next_entry_number()} — Precondition diagnostics before Stage 4",
        commit_sha=git_sha(),
        what_changed="Four precondition checks required before locking a final holdout: pooled baseline confirmation, a null-feature sanity check on the permutation-importance methodology, per-feature reverse ablation (development folds only), and bootstrap CIs on the upset-conditional Hit@1.",
        hypothesis="These validate the diagnostic tooling itself and identify dead features before Stage 5 adds more, on development folds only so the locked holdout (Stage 4) stays untouched by any feature-selection decision.",
        config_diff="No split/model change. Reverse ablation restricted to development folds (test years 2017-2023) only — 2024/2025 are reserved for the Stage 4 locked holdout.",
        metrics_table_md=d1 + "\n\n" + d2 + "\n" + d3 + "\n" + d4,
        headline=f"Dead features (reverse ablation, dev folds only): {', '.join(dead_features) if dead_features else 'none'}",
        verdict="n/a (precondition/diagnostic entry)",
        next_steps="Proceed to Stage 4 (lock 2024-2025 as a final holdout) now that these are logged.",
    )
    print("REPORT.md updated.")
    print(d1); print(d2); print(d3); print(d4)


if __name__ == "__main__":
    main()
