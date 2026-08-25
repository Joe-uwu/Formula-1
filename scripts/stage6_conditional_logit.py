"""Stage 6: conditional logit (win-only) and Plackett-Luce (full-field),
compared against the existing xgb_v1 classifier and xgb_rank_ndcg on
DEVELOPMENT FOLDS ONLY (2017-2023), same discipline as everything since
Stage 4. Uses the honest batch-0 feature set (12 features) established by
Fix 3 — nothing from Stage 5 survived nested selection.
"""
import subprocess
import sys

sys.stdout.reconfigure(encoding="utf-8")
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from f1.config import SPLIT_CONFIG
from f1.eval import metrics as M
from f1.eval.rolling_origin import run_rolling_origin, DEV_TEST_YEARS
from f1.features.materialize import get_or_build
from f1.models.baseline import pole_sitter_predictions
from f1.models.conditional_logit import ConditionalLogitModel, PlackettLuceModel
from f1.models.ranker import RankerModel
from f1.models.train import WinModel
from f1 import report
from f1.eval import walk_forward

STAGE6_CONFIG_HASH = "stage6_dev"


def git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]).decode().strip()
    except Exception:
        return "unknown"


class _PoleModel:
    def fit(self, train_df, val_df=None):
        return self

    def predict_race(self, race_df):
        return pole_sitter_predictions(race_df)


def main():
    feature_set_id, feature_df, _ = get_or_build(SPLIT_CONFIG)
    feature_df["as_of_date"] = pd.to_datetime(feature_df["as_of_date"])

    systems_defs = [
        ("pole_sitter", lambda: _PoleModel()),
        ("xgb_v1", lambda: WinModel()),
        ("xgb_rank_ndcg", lambda: RankerModel("rank:ndcg")),
        ("conditional_logit", lambda: ConditionalLogitModel()),
        ("plackett_luce", lambda: PlackettLuceModel()),
    ]

    preds_by_system = {}
    for name, factory in systems_defs:
        preds = run_rolling_origin(feature_df, factory, model_version=name,
                                     config_hash=STAGE6_CONFIG_HASH, test_years=DEV_TEST_YEARS)
        preds_by_system[name] = preds
        print(f"{name}: {preds['race_id'].nunique()} races")

    n_races = preds_by_system["xgb_v1"]["race_id"].nunique()
    pole_stats = M.RaceLevelStats.from_predictions(preds_by_system["pole_sitter"])

    rows = ["| System | Hit@1 | Hit@3 | MRR | NDCG@5 | Spearman | Log loss | Brier |",
            "|---|---|---|---|---|---|---|---|"]
    stats_by_system = {}
    for name, preds in preds_by_system.items():
        stats = M.RaceLevelStats.from_predictions(preds)
        stats_by_system[name] = stats
        rows.append(f"| {name} | {stats.value('hit_at_1'):.4f} | {stats.value('hit_at_3'):.4f} | "
                     f"{stats.value('mrr'):.4f} | {stats.value('ndcg_at_5'):.4f} | {stats.value('spearman'):.4f} | "
                     f"{stats.value('log_loss'):.4f} | {stats.value('brier_score'):.4f} |")
    table = "\n".join(rows)

    # CIs on the two new models vs pole-sitter, on hit_at_1 and log_loss (the two axes that matter)
    ci_rows = ["\n**Paired CIs vs pole-sitter (development folds):**\n",
                "| System | Metric | Δ | 95% CI | Verdict |", "|---|---|---|---|---|"]
    for name in ("conditional_logit", "plackett_luce"):
        for metric in ("hit_at_1", "log_loss"):
            point, dlo, dhi = M.paired_bootstrap_ci(stats_by_system[name], pole_stats, metric)
            v = M.verdict(dlo, dhi, metric=metric)
            ci_rows.append(f"| {name} | {metric} | {point:+.4f} | [{dlo:+.4f}, {dhi:+.4f}] | {v} |")
    ci_md = "\n".join(ci_rows)

    cl_spear = stats_by_system["conditional_logit"].value("spearman")
    pl_spear = stats_by_system["plackett_luce"].value("spearman")
    classifier_spear = stats_by_system["xgb_v1"].value("spearman")

    report.append_entry(
        title=f"Entry {report.next_entry_number()} — Stage 6: conditional logit / Plackett-Luce vs. classifier and ranker ({n_races} dev-fold races)",
        commit_sha=git_sha(),
        what_changed=(
            "Implemented a conditional logit (win-only cross-entropy, softmax within race) and its Plackett-Luce "
            "extension (trained on the full observed finishing order, softmax over successively smaller "
            "remaining fields) in PyTorch. Compared against xgb_v1 (binary classifier) and xgb_rank_ndcg on the "
            "honest batch-0 feature set (12 features), development folds only."
        ),
        hypothesis=(
            "A race is a discrete choice among ~20 alternatives where exactly one wins; the binary classifier's "
            "loss doesn't encode that. The conditional logit's loss directly optimizes what's being measured "
            "(within-race softmax cross-entropy against the winner); Plackett-Luce extends that to the full "
            "field, which should help Spearman the same way the ranking objective did in Stage 3."
        ),
        config_diff=f"feature_columns=batch-0 (12, post Fix-3), dev folds={list(DEV_TEST_YEARS)}",
        metrics_table_md=table + ci_md,
        headline=(f"Spearman: classifier={classifier_spear:.4f}, conditional_logit={cl_spear:.4f}, "
                    f"plackett_luce={pl_spear:.4f}."),
        verdict="see per-metric CIs above — mixed by design, reported as such",
        next_steps="Stage 8 (skipping Stage 7 per instruction): calibration — single temperature on the within-race softmax, fit on development folds by log loss; then blend against grid-only.",
    )
    print("REPORT.md updated.")
    print(table)
    print(ci_md)


if __name__ == "__main__":
    main()
