"""Entry 29: documents the Entry 27 model-selection error. No new modeling,
no rerun of the locked holdout — that would destroy the one-shot property
that makes Entry 27 meaningful. This is a record of an undocumented
decision, not a correction of Entry 27's numbers.
"""
import subprocess
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from f1 import report


def git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]).decode().strip()
    except Exception:
        return "unknown"


def main():
    metrics_md = (
        "**Entry 25's development-fold comparison (145 races), the only evidence available at "
        "selection time:**\n\n"
        "| System | Hit@1 | Hit@3 | MRR | NDCG@5 | Spearman | Log loss |\n"
        "|---|---|---|---|---|---|---|\n"
        "| pole_sitter | 0.5241 | 0.8759 | 0.6946 | 0.8560 | 0.7620 | 1.6436 |\n"
        "| xgb_v1 (classifier) | 0.5379 | 0.8759 | 0.7096 | 0.8580 | 0.6887 | 0.1072 |\n"
        "| **xgb_rank_ndcg** | **0.5586** | **0.8897** | **0.7261** | **0.8734** | 0.7833 | **0.1039** |\n"
        "| conditional_logit | 0.5103 | 0.8966 | 0.7053 | 0.8637 | 0.7592 | 0.1113 |\n"
        "| **plackett_luce (chosen for Entry 27)** | 0.4897 | 0.8828 | 0.6928 | 0.8603 | **0.7939** | 0.1174 |\n\n"
        "xgb_rank_ndcg leads on Hit@1, Hit@3, MRR, NDCG@5, and log loss — every metric except Spearman, "
        "where plackett_luce leads (0.7939 vs 0.7833) by construction: it's the only one of the five "
        "trained on full-field ranking loss rather than a win-only signal. plackett_luce's Hit@1 (0.4897) "
        "is the WORST of all five systems tested in Entry 25, including the pole-sitter baseline (0.5241) "
        "and the plain classifier (0.5379) — 0.0689 below pole-sitter, 0.0482 below the plain classifier.\n\n"
        "Entry 26 (Stage 8 calibration) fits a temperature and a grid-only blend weight for plackett_luce "
        "and conditional_logit — it does not calibrate or even mention xgb_rank_ndcg, and records no reason "
        "plackett_luce was the model advanced to calibration and the locked holdout instead of the model "
        "that led on 5 of 6 dev-fold metrics. No selection criterion appears anywhere in Entries 25 or 26."
    )

    headline = (
        "The locked holdout (Entry 27) evaluated plackett_luce_calibrated_blend, which had the worst "
        "dev-fold Hit@1 of the five systems compared in Entry 25 (0.4897, vs 0.5241 pole-sitter and "
        "0.5379 classifier), while xgb_rank_ndcg led on Hit@1/Hit@3/MRR/NDCG@5/log loss. No selection "
        "criterion was recorded before the choice was made."
    )

    body = (
        "\n\n**What the criterion should have been:** a single metric, fixed and written down BEFORE Stage 8 "
        "(calibration) touched any model, evaluated only on development folds. This project's own framing "
        "(see README, and Entry 1's reason for existing at all) treats Hit@1 as the headline decision "
        "metric — top-1 predictive accuracy is what 'who wins' means. Under that criterion, Entry 25 already "
        "contained the answer: xgb_rank_ndcg (0.5586), not plackett_luce (0.4897). Spearman — the metric "
        "plackett_luce does lead on — measures agreement with the FULL finishing order, which is a "
        "reasonable thing to optimize for but is not the project's stated target and was never proposed as "
        "the selection criterion anywhere in the log. Choosing the model with the best full-ranking "
        "correlation and then locking it in as the Hit@1 holdout answer is a metric mismatch between the "
        "selection step and the question the holdout is being asked to answer.\n\n"
        "**Where this should have been fixed:** before Stage 8. Entry 26's calibration step operated on "
        "plackett_luce and conditional_logit only, already having silently excluded xgb_rank_ndcg from "
        "contention. The selection criterion needed to be fixed and applied at the end of Entry 25, using "
        "only development-fold evidence, before any model was calibrated or handed to the locked holdout.\n\n"
        "**What this entry does NOT do:** rerun the locked holdout on xgb_rank_ndcg or any other model. "
        "Entry 27 already spent the one 2024-2025 evaluation this project gets — that is what makes its "
        "0.4792 Hit@1 an honest generalization estimate rather than a cherry-picked one. Rerunning it on a "
        "different model, even the one that should have been selected, would make the new number just as "
        "unprincipled as the original selection was. Entry 27 stands as the honest generalization estimate "
        "for the model that was actually evaluated (plackett_luce_calibrated_blend) — not for the model "
        "that development-fold evidence says should have been chosen. The README states this distinction "
        "explicitly rather than presenting Entry 27's number as if it were an evaluation of the project's "
        "best model."
    )

    report.append_entry(
        title=f"Entry {report.next_entry_number()} — Model-selection error: Entry 27 locked the worst dev-fold Hit@1 performer",
        commit_sha=git_sha(),
        what_changed=(
            "Documentation entry, no new modeling and no rerun of the locked holdout. Reviews Entry 25's "
            "development-fold comparison against the model Entry 27 actually evaluated."
        ),
        hypothesis=(
            "Entry 26 records no selection criterion for choosing plackett_luce_calibrated_blend over the "
            "other four systems Entry 25 compared; checking Entry 25's own table against that choice."
        ),
        config_diff="No split/model/config change. Read-only review of Entries 25-27's already-logged numbers.",
        metrics_table_md=metrics_md + body,
        headline=headline,
        verdict="n/a (methodology-documentation entry)",
        next_steps="None — this phase is closing. See README for how this and Entry 27 are reported together.",
    )
    print("REPORT.md updated.")
    print(headline)


if __name__ == "__main__":
    main()
