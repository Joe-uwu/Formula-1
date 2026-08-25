"""Stage 2: replace the single 24-race 2025 holdout with rolling-origin
walk-forward — train on all seasons <= Y, test on season Y+1, for Y in
2016..2024 (test years 2017-2025, ~193 races pooled), retraining fresh each
fold. Bootstrap CIs are recomputed on the pooled set, race as the resampling
unit. Also reports per-fold Hit@1 so era effects are visible.
"""
import subprocess
import sys

sys.stdout.reconfigure(encoding="utf-8")
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from f1.config import SPLIT_CONFIG, config_hash
from f1.features.materialize import get_or_build
from f1.eval import walk_forward, metrics as M
from f1.eval.rolling_origin import run_rolling_origin, FOLD_TEST_YEARS
from f1.models.baseline import pole_sitter_predictions
from f1.models.train import WinModel
from f1 import report

ROLLING_CONFIG_HASH = config_hash() + "_rolling"


def git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]).decode().strip()
    except Exception:
        return "unknown"


def main():
    feature_set_id, feature_df, timings = get_or_build(SPLIT_CONFIG)
    feature_df["as_of_date"] = pd.to_datetime(feature_df["as_of_date"])

    model_preds = run_rolling_origin(
        feature_df, lambda: WinModel(), model_version="xgb_v1", config_hash=ROLLING_CONFIG_HASH,
    )

    pooled_start = pd.Timestamp(f"{min(FOLD_TEST_YEARS)}-01-01")
    pooled_end = pd.Timestamp(f"{max(FOLD_TEST_YEARS) + 1}-01-01")
    pooled_test_df = feature_df[(feature_df["as_of_date"] >= pooled_start) & (feature_df["as_of_date"] < pooled_end)]
    pole_preds = walk_forward.run(pooled_test_df, pole_sitter_predictions,
                                    model_version="pole_sitter", config_hash=ROLLING_CONFIG_HASH)

    n_races = model_preds["race_id"].nunique()

    stats = M.RaceLevelStats.from_predictions(model_preds)
    baseline_stats = M.RaceLevelStats.from_predictions(pole_preds)

    rows = ["| Metric | Value | 95% CI | Δ vs pole-sitter | Δ 95% CI | Verdict |",
            "|---|---|---|---|---|---|"]
    for name in M.HEADLINE_METRIC_NAMES:
        point, lo, hi = M.bootstrap_ci(stats, name)
        _, dlo, dhi = M.paired_bootstrap_ci(stats, baseline_stats, name)
        v = M.verdict(dlo, dhi, metric=name)
        rows.append(f"| {name} | {point:.4f} | [{lo:.4f}, {hi:.4f}] | "
                     f"{point - baseline_stats.value(name):+.4f} | [{dlo:+.4f}, {dhi:+.4f}] | {v} |")
    table = "\n".join(rows) + "\n"

    # per-fold Hit@1
    per_fold = ["\n**Per-fold Hit@1 (test season Y+1, trained on everything <= Y):**\n"]
    for year, g in model_preds.groupby("fold_test_year"):
        per_fold.append(f"- {year}: {M.hit_at_k(g, 1):.3f} (n={g['race_id'].nunique()} races)")
    per_fold_md = "\n".join(per_fold)

    # Hit@3 ceiling check (spec's specific claim: 23/24 in 2025)
    preds_2025 = model_preds[model_preds["fold_test_year"] == 2025]
    hit3_2025 = M.hit_at_k(preds_2025, 3) if not preds_2025.empty else float("nan")
    n_2025 = preds_2025["race_id"].nunique() if not preds_2025.empty else 0
    ceiling_note = (
        f"\n\n**Hit@3 ceiling:** {hit3_2025:.4f} in 2025 ({round(hit3_2025 * n_2025)}/{n_2025} races) — the "
        f"winner is in the top-3 grid slots almost every race, so Hit@3 has almost no headroom left to "
        f"discriminate between models. Reported in the table above but shouldn't be leaned on."
    )

    headline_hit1 = stats.value("hit_at_1")

    report.append_entry(
        title=f"Entry {report.next_entry_number()} — Stage 2: rolling-origin evaluation ({n_races} races pooled)",
        commit_sha=git_sha(),
        what_changed=(
            f"Replaced the single 2025 holdout (24 races) with rolling-origin walk-forward: retrain on all "
            f"seasons <= Y, test on season Y+1, for Y=2016..2024 (test seasons 2017-2025), pooling all "
            f"{n_races} test races for the headline bootstrap CIs. Pole-sitter baseline evaluated on the "
            f"identical pooled race set."
        ),
        hypothesis=(
            "A 24-race single holdout gives ~24 Bernoulli trials for Hit@1, which is why every Entry 6 "
            "ranking metric was 'inconclusive' with CIs spanning [0.42, 0.79] — that's an uninformative test, "
            "not a null result. ~193 pooled races should narrow the CIs enough to actually distinguish "
            "'no evidence of a difference' from 'evidence of no difference.'"
        ),
        config_diff=f"rolling-origin folds: test years {list(FOLD_TEST_YEARS)}, retrained per fold; base split config unchanged: {SPLIT_CONFIG}",
        metrics_table_md=table + per_fold_md + ceiling_note,
        headline=f"Pooled Hit@1={headline_hit1:.3f} over {n_races} races",
        verdict=M.verdict(*M.paired_bootstrap_ci(stats, baseline_stats, "hit_at_1")[1:], metric="hit_at_1"),
        next_steps=(
            "Stage 3: the model's Spearman correlation against actual finishing order is a statistically real "
            "loss vs. just using grid order (pole-sitter's implicit full-field ranking). Binary Win-target "
            "training has no incentive to order P8-P15 correctly; try a learning-to-rank objective "
            "(rank:pairwise, rank:ndcg) and see if that regression disappears."
        ),
    )
    print("REPORT.md updated.")
    print(f"pooled races: {n_races}")
    print(table)


if __name__ == "__main__":
    main()
