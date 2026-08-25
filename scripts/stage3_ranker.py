"""Stage 3: fix the objective mismatch. Trains XGBoost rankers (rank:pairwise
and rank:ndcg, grouped by race, graded relevance from finishing position) via
the same rolling-origin evaluation as Stage 2, and reports them side by side
with the Entry 9 classifier and pole-sitter on every metric. The specific
question: does the Spearman regression against grid order (Entry 9: model
0.651 vs. pole-sitter's implicit full-field grid ordering) disappear?
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
from f1.models.ranker import RankerModel
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

    pairwise_preds = run_rolling_origin(
        feature_df, lambda: RankerModel("rank:pairwise"),
        model_version="xgb_rank_pairwise", config_hash=ROLLING_CONFIG_HASH,
    )
    ndcg_preds = run_rolling_origin(
        feature_df, lambda: RankerModel("rank:ndcg"),
        model_version="xgb_rank_ndcg", config_hash=ROLLING_CONFIG_HASH,
    )

    # classifier + pole-sitter already logged by Stage 2 under this same config_hash
    classifier_preds = walk_forward.fetch_predictions("xgb_v1", ROLLING_CONFIG_HASH)
    pole_preds = walk_forward.fetch_predictions("pole_sitter", ROLLING_CONFIG_HASH)

    systems = [
        ("xgb_v1 (classifier, binary Win target)", classifier_preds),
        ("xgb_rank_pairwise", pairwise_preds),
        ("xgb_rank_ndcg", ndcg_preds),
        ("pole_sitter", pole_preds),
    ]

    n_races = classifier_preds["race_id"].nunique()
    rows = ["| System | Hit@1 | Hit@3 | MRR | NDCG@5 | Spearman | Log loss | Brier |",
            "|---|---|---|---|---|---|---|---|"]
    stats_by_system = {}
    for name, preds in systems:
        stats = M.RaceLevelStats.from_predictions(preds)
        stats_by_system[name] = stats
        rows.append(
            f"| {name} | {stats.value('hit_at_1'):.4f} | {stats.value('hit_at_3'):.4f} | "
            f"{stats.value('mrr'):.4f} | {stats.value('ndcg_at_5'):.4f} | {stats.value('spearman'):.4f} | "
            f"{stats.value('log_loss'):.4f} | {stats.value('brier_score'):.4f} |"
        )
    table = "\n".join(rows) + "\n"

    # Spearman CIs specifically, since that's the question this stage answers
    spearman_rows = ["\n**Spearman vs. actual finishing order, with CIs (this is the question Stage 3 asks):**\n",
                       "| System | Spearman | 95% CI | Δ vs pole-sitter | Δ 95% CI | Verdict |", "|---|---|---|---|---|---|"]
    pole_stats = stats_by_system["pole_sitter"]
    for name, stats in stats_by_system.items():
        if name == "pole_sitter":
            continue
        point, lo, hi = M.bootstrap_ci(stats, "spearman")
        _, dlo, dhi = M.paired_bootstrap_ci(stats, pole_stats, "spearman")
        v = M.verdict(dlo, dhi, metric="spearman")
        spearman_rows.append(f"| {name} | {point:.4f} | [{lo:.4f}, {hi:.4f}] | "
                               f"{point - pole_stats.value('spearman'):+.4f} | [{dlo:+.4f}, {dhi:+.4f}] | {v} |")
    spearman_md = "\n".join(spearman_rows)

    best_ranker_name = "xgb_rank_ndcg" if stats_by_system["xgb_rank_ndcg"].value("spearman") > stats_by_system["xgb_rank_pairwise"].value("spearman") else "xgb_rank_pairwise"
    best_ranker_spearman = stats_by_system[best_ranker_name].value("spearman")
    classifier_spearman = stats_by_system["xgb_v1 (classifier, binary Win target)"].value("spearman")
    pole_spearman = pole_stats.value("spearman")
    resolved = best_ranker_spearman >= pole_spearman - 0.02  # within noise of the baseline

    headline = (f"Spearman: classifier={classifier_spearman:.4f}, {best_ranker_name}={best_ranker_spearman:.4f}, "
                 f"pole_sitter={pole_spearman:.4f}. Regression {'disappears' if resolved else 'does NOT disappear'} "
                 f"with a ranking objective.")

    report.append_entry(
        title=f"Entry {report.next_entry_number()} — Stage 3: rank:pairwise / rank:ndcg vs. binary classifier ({n_races} races pooled)",
        commit_sha=git_sha(),
        what_changed=(
            "Trained XGBRanker with rank:pairwise and rank:ndcg objectives, graded relevance from finishing "
            "position (field_size - position + 1, DNF/unknown = 0), grouped by race_id, same rolling-origin "
            "folds and features as Entry 9's classifier."
        ),
        hypothesis=(
            "Entry 9's classifier is trained on a binary Win target, so it has no incentive to order the "
            "mid-field correctly — only to separate P1 from the rest. Spearman scores the whole permutation, "
            "so a ranking objective should close that gap even if Hit@1 doesn't move."
        ),
        config_diff=f"objective: binary:logistic -> rank:pairwise / rank:ndcg; everything else unchanged from Entry 9: {SPLIT_CONFIG}",
        metrics_table_md=table + spearman_md,
        headline=headline,
        verdict=("improved" if resolved else "worsened") + " (Spearman-specific; see full table for other metrics)",
        next_steps="Run diagnostics: predicted-probability-vs-grid-position correlation, permutation importance, grid/quali ablation, pooled upset breakdown.",
    )
    print("REPORT.md updated.")
    print(table)
    print(spearman_md)


if __name__ == "__main__":
    main()
