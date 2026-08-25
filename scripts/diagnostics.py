"""Diagnostics, run after Stage 3. Explain the result regardless of which
model wins:

1. Spearman(predicted_probability, grid_position), pooled — if this is
   near 1, the model is a grid lookup with noise.
2. Permutation importance on the held-out 2025 fold (not training data).
3. Ablation: retrain with grid_position/quali_position removed entirely.
4. Upset breakdown (Hit@1 | pole won vs. didn't), pooled across all 193
   rolling-origin races instead of the original 24.
"""
import subprocess
import sys

sys.stdout.reconfigure(encoding="utf-8")
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from f1.config import SPLIT_CONFIG, config_hash
from f1.features.materialize import get_or_build, FEATURE_COLUMNS
from f1.eval import walk_forward, metrics as M
from f1.eval.rolling_origin import run_rolling_origin, FOLD_TEST_YEARS
from f1.models.train import WinModel
from f1.models.ranker import RankerModel
from f1 import report

ROLLING_CONFIG_HASH = config_hash() + "_rolling"
ABLATION_CONFIG_HASH = config_hash() + "_rolling_no_grid"
GRID_COLS = ["grid_position", "quali_position"]


def git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]).decode().strip()
    except Exception:
        return "unknown"


def diag1_prob_vs_grid(feature_df, preds_by_system) -> str:
    lines = ["**1. Spearman(predicted probability, grid position), pooled:**\n"]
    grid_lookup = feature_df.set_index(["race_id", "driver_id"])["grid_position"]
    for name, preds in preds_by_system.items():
        merged = preds.set_index(["race_id", "driver_id"]).join(grid_lookup, how="inner").reset_index()
        merged = merged.dropna(subset=["grid_position"])
        rho, _ = spearmanr(merged["predicted_probability"], -merged["grid_position"])  # lower grid = better = higher prob
        flag = " <- essentially a grid lookup" if abs(rho) > 0.9 else ""
        lines.append(f"- {name}: {rho:.4f}{flag}")
    return "\n".join(lines)


def _permutation_importance(model, test_df: pd.DataFrame, n_repeats: int = 5, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    def pooled_log_loss(df):
        preds = pd.concat([model.predict_race(g).assign(actual_position=g["target_position"])
                             for _, g in df.groupby("race_id")], ignore_index=True)
        return M.RaceLevelStats.from_predictions(preds).value("log_loss")

    baseline = pooled_log_loss(test_df)
    rows = []
    for col in FEATURE_COLUMNS:
        drops = []
        for _ in range(n_repeats):
            shuffled = test_df.copy()
            shuffled[col] = rng.permutation(shuffled[col].to_numpy())
            drops.append(pooled_log_loss(shuffled) - baseline)  # positive = log loss got worse = feature mattered
        rows.append(dict(feature=col, mean_log_loss_increase=float(np.mean(drops)), std=float(np.std(drops))))
    return pd.DataFrame(rows).sort_values("mean_log_loss_increase", ascending=False)


def diag2_permutation_importance(feature_df) -> str:
    train_df = feature_df[feature_df["as_of_date"] < "2025-01-01"]
    test_df = feature_df[(feature_df["as_of_date"] >= "2025-01-01") & (feature_df["as_of_date"] < "2026-01-01")]
    model = WinModel().fit(train_df)
    imp = _permutation_importance(model, test_df)
    lines = ["\n**2. Permutation importance (xgb_v1, held out on 2025, log-loss increase when a feature is shuffled):**\n",
              "| Feature | Mean log-loss increase | Std |", "|---|---|---|"]
    for r in imp.itertuples():
        lines.append(f"| {r.feature} | {r.mean_log_loss_increase:+.5f} | {r.std:.5f} |")
    return "\n".join(lines)


def diag3_ablation(feature_df) -> tuple[str, dict]:
    ablated = feature_df.copy()
    ablated[GRID_COLS] = np.nan

    classifier_preds = walk_forward.fetch_predictions("xgb_v1", ROLLING_CONFIG_HASH)
    classifier_stats = M.RaceLevelStats.from_predictions(classifier_preds)

    ablated_preds = run_rolling_origin(
        ablated, lambda: WinModel(), model_version="xgb_v1_no_grid", config_hash=ABLATION_CONFIG_HASH,
    )
    ablated_stats = M.RaceLevelStats.from_predictions(ablated_preds)

    lines = ["\n**3. Ablation — grid_position and quali_position removed entirely (xgb_v1 classifier):**\n",
              "| Metric | With grid/quali | Without | Drop |", "|---|---|---|---|"]
    drops = {}
    for name in M.HEADLINE_METRIC_NAMES:
        with_v = classifier_stats.value(name)
        without_v = ablated_stats.value(name)
        drops[name] = with_v - without_v
        lines.append(f"| {name} | {with_v:.4f} | {without_v:.4f} | {with_v - without_v:+.4f} |")
    return "\n".join(lines), drops


def diag4_upset_breakdown(feature_df, preds_by_system) -> str:
    pole_won_races = set(
        feature_df.loc[(feature_df["quali_position"] == 1) & (feature_df["target_win"] == 1), "race_id"]
    )
    lines = ["\n**4. Upset breakdown, pooled across all rolling-origin races:**\n"]
    for name, preds in preds_by_system.items():
        all_races = set(preds["race_id"])
        won = preds[preds["race_id"].isin(pole_won_races)]
        lost = preds[preds["race_id"].isin(all_races - pole_won_races)]
        lines.append(f"- {name}: pole won (n={won['race_id'].nunique()}) Hit@1={M.hit_at_k(won, 1):.3f} | "
                      f"pole lost (n={lost['race_id'].nunique()}) Hit@1={M.hit_at_k(lost, 1):.3f}")
    return "\n".join(lines)


def main():
    feature_set_id, feature_df, _ = get_or_build(SPLIT_CONFIG)
    feature_df["as_of_date"] = pd.to_datetime(feature_df["as_of_date"])

    preds_by_system = {
        "xgb_v1 (classifier)": walk_forward.fetch_predictions("xgb_v1", ROLLING_CONFIG_HASH),
        "xgb_rank_pairwise": walk_forward.fetch_predictions("xgb_rank_pairwise", ROLLING_CONFIG_HASH),
        "xgb_rank_ndcg": walk_forward.fetch_predictions("xgb_rank_ndcg", ROLLING_CONFIG_HASH),
        "pole_sitter": walk_forward.fetch_predictions("pole_sitter", ROLLING_CONFIG_HASH),
    }

    d1 = diag1_prob_vs_grid(feature_df, preds_by_system)
    d2 = diag2_permutation_importance(feature_df)
    d3, ablation_drops = diag3_ablation(feature_df)
    d4 = diag4_upset_breakdown(feature_df, preds_by_system)

    hit1_drop = ablation_drops["hit_at_1"]
    grid_independent_signal = "yes" if hit1_drop < 0.03 else "some" if hit1_drop < 0.08 else "little to none"

    report.append_entry(
        title=f"Entry {report.next_entry_number()} — Diagnostics",
        commit_sha=git_sha(),
        what_changed="Four diagnostics on the Stage 2/3 rolling-origin models: predicted-probability-vs-grid correlation, permutation importance, a grid/quali ablation, and the pooled upset breakdown.",
        hypothesis="These explain *why* the models perform as they do, regardless of which one wins on the headline metrics.",
        config_diff="No split/model change — read-only analysis over Entry 9/10's already-logged rolling-origin predictions, plus one new ablation run (grid_position, quali_position set to null).",
        metrics_table_md=d1 + "\n" + d2 + "\n" + d3 + "\n" + d4,
        headline=f"Hit@1 drop when grid/quali removed: {hit1_drop:+.4f} -> non-grid features carry {grid_independent_signal} independent signal.",
        verdict="n/a (diagnostic entry, not a model comparison)",
        next_steps="If independent signal is weak, the honest path forward is either richer non-grid features (pit strategy, tyre degradation, weather forecast at prediction time) or accepting grid-order-plus-noise as the ceiling for this feature set.",
    )
    print("REPORT.md updated.")
    print(d1); print(d2); print(d3); print(d4)


if __name__ == "__main__":
    main()
