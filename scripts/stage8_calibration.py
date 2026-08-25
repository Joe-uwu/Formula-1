"""Stage 8: calibration.

One temperature parameter T on the within-race softmax, fit on development
folds by log loss — replacing the old pre-rebuild project's three
parameters (grid_alpha, temperature, form_boost_weight) tuned by
differential evolution against Hit@1 on fifteen races. That contrast is
worth its own paragraph in the README: one honestly-fit parameter on one
proper metric vs. three parameters hand-fit against a noisy metric on a
sample too small to support them.

Temperature-scaling softmax(s/T) is mathematically exact as a power
transform on already-computed probabilities — softmax(s/T)_i =
p_i^(1/T) / sum_j p_j^(1/T) where p = softmax(s) — so this calibrates the
already-logged Stage 6 predictions directly, no retraining needed. Applied
to plackett_luce (Stage 6's best Spearman) and conditional_logit.

Then blends the calibrated model against the grid-only logistic baseline
with a weight also fit on development folds by log loss. If the optimal
weight lands near 1.0 (all baseline, no model), that's reported as a
finding, not hidden.
"""
import subprocess
import sys

sys.stdout.reconfigure(encoding="utf-8")
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from f1.config import config_hash
from f1.eval import metrics as M
from f1.eval import walk_forward
from f1.eval.rolling_origin import DEV_TEST_YEARS
from f1 import report

STAGE6_CONFIG_HASH = "stage6_dev"
ROLLING_CONFIG_HASH = config_hash() + "_rolling"


def git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]).decode().strip()
    except Exception:
        return "unknown"


def temperature_scale(preds: pd.DataFrame, T: float) -> pd.DataFrame:
    df = preds.copy()
    powered = df["predicted_probability"].clip(lower=1e-12) ** (1.0 / T)
    df["predicted_probability"] = powered / df.groupby("race_id")["predicted_probability"].transform(
        lambda s: (s.clip(lower=1e-12) ** (1.0 / T)).sum()
    )
    return df


def fit_temperature(preds: pd.DataFrame) -> float:
    def neg_score(T):
        scaled = temperature_scale(preds, T)
        return M.RaceLevelStats.from_predictions(scaled).value("log_loss")

    result = minimize_scalar(neg_score, bounds=(0.05, 10.0), method="bounded")
    return float(result.x)


def blend(preds_a: pd.DataFrame, preds_b: pd.DataFrame, w: float) -> pd.DataFrame:
    """w * a + (1-w) * b, aligned on (race_id, driver_id)."""
    merged = preds_a.merge(preds_b, on=["race_id", "driver_id"], suffixes=("_a", "_b"))
    merged["predicted_probability"] = w * merged["predicted_probability_a"] + (1 - w) * merged["predicted_probability_b"]
    merged["predicted_rank"] = merged.groupby("race_id")["predicted_probability"].rank(ascending=False, method="first").astype(int)
    merged["actual_position"] = merged["actual_position_a"]
    return merged[["race_id", "driver_id", "predicted_probability", "predicted_rank", "actual_position"]]


def fit_blend_weight(preds_a: pd.DataFrame, preds_b: pd.DataFrame) -> float:
    def neg_score(w):
        blended = blend(preds_a, preds_b, w)
        return M.RaceLevelStats.from_predictions(blended).value("log_loss")

    result = minimize_scalar(neg_score, bounds=(0.0, 1.0), method="bounded")
    return float(result.x)


def main():
    pl_preds = walk_forward.fetch_predictions("plackett_luce", STAGE6_CONFIG_HASH)
    cl_preds = walk_forward.fetch_predictions("conditional_logit", STAGE6_CONFIG_HASH)
    grid_preds_all = walk_forward.fetch_predictions("grid_logistic", ROLLING_CONFIG_HASH)
    dev_races = set(pl_preds["race_id"])  # plackett_luce dev-fold races define the comparison set
    grid_preds = grid_preds_all[grid_preds_all["race_id"].isin(dev_races)]

    rows = ["**Temperature fit (development folds, minimizing pooled log loss):**\n",
             "| Model | T | Log loss before | Log loss after |", "|---|---|---|---|"]
    calibrated = {}
    temperatures = {}
    for name, preds in [("plackett_luce", pl_preds), ("conditional_logit", cl_preds)]:
        before = M.RaceLevelStats.from_predictions(preds).value("log_loss")
        T = fit_temperature(preds)
        scaled = temperature_scale(preds, T)
        after = M.RaceLevelStats.from_predictions(scaled).value("log_loss")
        calibrated[name] = scaled
        temperatures[name] = T
        rows.append(f"| {name} | {T:.3f} | {before:.4f} | {after:.4f} |")
    temp_table = "\n".join(rows)

    blend_rows = ["\n**Blend weight vs. grid-only logistic (development folds, minimizing pooled log loss):**\n",
                   "| Model | w (model weight) | Log loss (model alone) | Log loss (grid alone) | Log loss (blended) |",
                   "|---|---|---|---|---|"]
    blend_results = {}
    for name, preds in calibrated.items():
        common = set(preds["race_id"]) & set(grid_preds["race_id"])
        a = preds[preds["race_id"].isin(common)]
        b = grid_preds[grid_preds["race_id"].isin(common)]
        w = fit_blend_weight(a, b)
        blended = blend(a, b, w)
        ll_model = M.RaceLevelStats.from_predictions(a).value("log_loss")
        ll_grid = M.RaceLevelStats.from_predictions(b).value("log_loss")
        ll_blend = M.RaceLevelStats.from_predictions(blended).value("log_loss")
        blend_results[name] = (w, ll_model, ll_grid, ll_blend)
        blend_rows.append(f"| {name} | {w:.3f} | {ll_model:.4f} | {ll_grid:.4f} | {ll_blend:.4f} |")
    blend_table = "\n".join(blend_rows)

    near_one = [name for name, (w, *_rest) in blend_results.items() if w > 0.9]
    near_zero = [name for name, (w, *_rest) in blend_results.items() if w < 0.1]
    finding = ""
    if near_zero:
        finding = f"\n\n**Finding:** optimal blend weight for {near_zero} landed near 0 — the calibrated model alone beats any blend with grid-only, i.e. grid-only isn't adding anything the model doesn't already have."
    elif near_one:
        finding = f"\n\n**Finding:** optimal blend weight for {near_one} landed near 1.0 (all model, ~0 grid-only) — same reading, from the other side."
    else:
        finding = "\n\n**Finding:** optimal blend weights are intermediate — both the model and the grid-only baseline contribute independently useful information."

    report.append_entry(
        title=f"Entry {report.next_entry_number()} — Stage 8: calibration (temperature + grid-only blend)",
        commit_sha=git_sha(),
        what_changed="Fit one temperature parameter per model (plackett_luce, conditional_logit) on development folds by log loss, via an exact power-transform on already-logged softmax probabilities (no retraining). Then fit a blend weight against grid-only logistic, also on development folds by log loss.",
        hypothesis="Contrast with the pre-rebuild project's three parameters (grid_alpha, temperature, form_boost_weight) tuned by differential evolution against Hit@1 on fifteen races — one honestly-fit parameter on a proper metric with a defensible sample, vs three parameters chasing a noisy metric on too few races. See README.",
        config_diff="No feature/split change. Post-hoc calibration of Stage 6's already-logged development-fold predictions.",
        metrics_table_md=temp_table + "\n" + blend_table + finding,
        headline=f"Temperatures: {', '.join(f'{k}={t:.2f}' for k, t in temperatures.items())}. Blend weights: {', '.join(f'{k}={v[0]:.2f}' for k, v in blend_results.items())}.",
        verdict="n/a (calibration entry)",
        next_steps="Locked holdout (2024-2025), run exactly once with --final, using the calibrated model chosen here.",
    )
    print("REPORT.md updated.")
    print(temp_table)
    print(blend_table)
    print(finding)


if __name__ == "__main__":
    main()
