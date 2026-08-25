"""Stage 1 of the post-Entry-6 evaluation fix: honest probabilistic baselines.

Same single holdout as Entry 6 (train<=2023, val=2024, test=2025) — this
stage is about what we're comparing against, not the evaluation design.
Reuses the already-logged xgb_v1 and pole-sitter predictions from the DB
(same model, same test races) rather than retraining, and adds two new
baselines: uniform (1/N) and grid-only logistic regression.
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
from f1.models.baseline import pole_sitter_predictions, uniform_predictions
from f1.models.grid_logistic import GridLogisticModel
from f1 import report

CLIP_EPS = 1e-15


def git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]).decode().strip()
    except Exception:
        return "unknown"


def main():
    feature_set_id, feature_df, timings = get_or_build(SPLIT_CONFIG)
    val_start = pd.Timestamp(SPLIT_CONFIG["val_start_date"])
    test_start = pd.Timestamp(SPLIT_CONFIG["test_start_date"])
    feature_df["as_of_date"] = pd.to_datetime(feature_df["as_of_date"])
    train_df = feature_df[feature_df["as_of_date"] < val_start]
    test_df = feature_df[feature_df["as_of_date"] >= test_start]

    chash = config_hash()

    # xgb_v1 and pole_sitter are already logged from Entry 6 — same test set,
    # same model, reuse the audited rows instead of rerunning.
    model_preds = walk_forward.fetch_predictions("xgb_v1", chash)
    pole_preds = walk_forward.fetch_predictions("pole_sitter", chash)

    uniform_preds = walk_forward.run(test_df, uniform_predictions,
                                       model_version="uniform", config_hash=chash)
    grid_logistic = GridLogisticModel().fit(train_df)
    grid_preds = walk_forward.run(test_df, grid_logistic.predict_race,
                                    model_version="grid_logistic", config_hash=chash)

    systems = [
        ("xgb_v1 (model)", model_preds),
        ("pole_sitter", pole_preds),
        ("uniform", uniform_preds),
        ("grid_logistic", grid_preds),
    ]

    rows = ["| System | Log loss | Brier | ECE | Brier Skill Score* |", "|---|---|---|---|---|"]
    values = {}
    for name, preds in systems:
        stats = M.RaceLevelStats.from_predictions(preds)
        ll = stats.value("log_loss")
        brier = stats.value("brier_score")
        ece, _ = M.expected_calibration_error(preds)
        bss = M.brier_skill_score(preds, pole_preds)
        values[name] = dict(log_loss=ll, brier=brier, ece=ece, bss=bss)
        rows.append(f"| {name} | {ll:.4f} | {brier:.4f} | {ece:.4f} | {bss:.4f} |")
    table = "\n".join(rows)
    footnote = (f"\n\n*BSS is computed against the pole-sitter baseline (pole-sitter's own BSS is 0 by "
                 f"construction). Log-loss probabilities are clipped to [{CLIP_EPS:g}, 1-{CLIP_EPS:g}] before "
                 f"averaging — pole-sitter's true log loss is infinite on every race it doesn't win (predicted "
                 f"probability 0 for the actual winner); the {CLIP_EPS:g} clip is what turns that into a finite "
                 f"number, so pole-sitter's log-loss column here is an artifact of the clipping constant, not a "
                 f"real probabilistic score. Its uniform-1.0/0.0 predictions make it uncomparable on log loss by "
                 f"design — that's exactly why this stage adds uniform and grid-only as the real comparison points.")

    model_vs_grid = "beats" if values["xgb_v1 (model)"]["log_loss"] < values["grid_logistic"]["log_loss"] else "does NOT beat"
    verdict_line = (f"xgb_v1 {model_vs_grid} grid_logistic on log loss "
                     f"({values['xgb_v1 (model)']['log_loss']:.4f} vs {values['grid_logistic']['log_loss']:.4f}). ")
    if model_vs_grid.startswith("does NOT"):
        verdict_line += "The feature set beyond starting position is contributing nothing measurable here."
    else:
        verdict_line += "The feature set beyond starting position is contributing something measurable here."

    report.append_entry(
        title=f"Entry {report.next_entry_number()} — Stage 1: honest probabilistic baselines",
        commit_sha=git_sha(),
        what_changed=(
            "Added uniform (1/N) and grid-only logistic regression baselines, both evaluated on the same "
            "2025 test set as Entry 6's xgb_v1 and pole-sitter predictions (reused from the DB, not rerun). "
            "Reports log loss, Brier, ECE, and Brier Skill Score for all four side by side."
        ),
        hypothesis=(
            "The Entry 6 log-loss/Brier 'improvement' over pole-sitter is suspected to be a clipping artifact, "
            "since pole-sitter's true log loss is infinite whenever pole doesn't win. Uniform and grid-only give "
            "honest, non-degenerate probability baselines to compare against instead."
        ),
        config_diff=f"{SPLIT_CONFIG}\n(same split as Entry 6 — this stage only adds baselines, no split change)",
        metrics_table_md=table + footnote,
        headline=verdict_line,
        verdict="n/a (baseline-comparison entry, not a model change)",
        next_steps="Stage 2: the 24-race single holdout gives ~24 Bernoulli trials for Hit@1 — replace it with rolling-origin evaluation before drawing conclusions from ranking metrics.",
    )
    print("REPORT.md updated.")
    print(table)
    print(footnote)


if __name__ == "__main__":
    main()
