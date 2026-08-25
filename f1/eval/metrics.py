"""Race-level ranking and probabilistic metrics, computed by querying the
`predictions` table — never by re-deriving numbers off to the side.

Design: every metric here is "one sufficient statistic per race, averaged."
`RaceLevelStats.from_predictions` pays for the one real per-race computation
(rank stats, Spearman, NDCG — each needs a groupby) exactly once. Bootstrap
resampling afterward is pure numpy array indexing: draw a
(n_draws, n_races) matrix of race indices, fancy-index the per-race value
arrays, reduce. No DataFrame is rebuilt and no groupby reruns inside the
bootstrap loop, which is what makes 1000+ resamples cheap instead of the
dominant cost.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

SIMPLE_MEAN_METRICS = ("hit_at_1", "hit_at_3", "mrr", "ndcg_at_5", "spearman")
WEIGHTED_MEAN_METRICS = ("log_loss", "brier_score")
HEADLINE_METRIC_NAMES = SIMPLE_MEAN_METRICS + WEIGHTED_MEAN_METRICS


def _single_race_stats(g: pd.DataFrame) -> tuple:
    winner = g.loc[g["actual_position"] == 1]
    if winner.empty:
        hit1 = hit3 = rr = np.nan
    else:
        winner_rank = winner["predicted_rank"].iloc[0]
        hit1 = float(winner_rank <= 1)
        hit3 = float(winner_rank <= 3)
        rr = 1.0 / winner_rank

    scored = g.dropna(subset=["actual_position"])
    if len(scored) >= 2:
        spearman, _ = spearmanr(scored["predicted_rank"], scored["actual_position"])
    else:
        spearman = np.nan

    if not scored.empty:
        relevance = 1.0 / scored["actual_position"]
        by_pred = scored.assign(rel=relevance).sort_values("predicted_rank").head(5)
        dcg = (by_pred["rel"] / np.log2(np.arange(2, len(by_pred) + 2))).sum()
        by_ideal = scored.assign(rel=relevance).sort_values("rel", ascending=False).head(5)
        idcg = (by_ideal["rel"] / np.log2(np.arange(2, len(by_ideal) + 2))).sum()
        ndcg5 = dcg / idcg if idcg > 0 else np.nan
    else:
        ndcg5 = np.nan

    p = g["predicted_probability"]
    total = p.sum()
    p_norm = (p / total) if total > 0 else pd.Series(1.0 / len(g), index=g.index)
    p_norm = p_norm.clip(1e-15, 1 - 1e-15)
    y = (g["actual_position"] == 1).astype(float)
    logloss_sum = float(-(y * np.log(p_norm) + (1 - y) * np.log(1 - p_norm)).sum())
    brier_sum = float(((p_norm - y) ** 2).sum())

    return hit1, hit3, rr, ndcg5, spearman, logloss_sum, brier_sum, len(g)


@dataclass
class RaceLevelStats:
    race_ids: np.ndarray
    hit_at_1: np.ndarray
    hit_at_3: np.ndarray
    mrr: np.ndarray
    ndcg_at_5: np.ndarray
    spearman: np.ndarray
    log_loss_sum: np.ndarray
    brier_score_sum: np.ndarray
    n_drivers: np.ndarray

    @classmethod
    def from_predictions(cls, preds: pd.DataFrame) -> "RaceLevelStats":
        race_ids, rows = [], []
        for race_id, g in preds.groupby("race_id"):
            race_ids.append(race_id)
            rows.append(_single_race_stats(g))
        cols = list(zip(*rows)) if rows else [[]] * 8
        hit1, hit3, rr, ndcg5, spearman, ll_sum, br_sum, n = (np.array(c, dtype=float) for c in cols)
        return cls(np.array(race_ids), hit1, hit3, rr, ndcg5, spearman, ll_sum, br_sum, n)

    def subset(self, race_ids: np.ndarray) -> "RaceLevelStats":
        """Reorder/filter to exactly `race_ids`, in that order — used to align
        two stats objects (e.g. model vs. baseline) on a common race set."""
        order = {rid: i for i, rid in enumerate(self.race_ids)}
        idx = np.array([order[rid] for rid in race_ids])
        return RaceLevelStats(
            self.race_ids[idx], self.hit_at_1[idx], self.hit_at_3[idx], self.mrr[idx],
            self.ndcg_at_5[idx], self.spearman[idx], self.log_loss_sum[idx],
            self.brier_score_sum[idx], self.n_drivers[idx],
        )

    def _draws(self, metric: str, idx: np.ndarray) -> np.ndarray:
        """idx: (n_draws, n_races) race-index matrix. Returns (n_draws,) metric values."""
        if metric in WEIGHTED_MEAN_METRICS:
            sums = self.log_loss_sum if metric == "log_loss" else self.brier_score_sum
            return sums[idx].sum(axis=1) / self.n_drivers[idx].sum(axis=1)
        return np.nanmean(getattr(self, metric)[idx], axis=1)

    def value(self, metric: str) -> float:
        return float(self._draws(metric, np.arange(len(self.race_ids))[None, :])[0])


def bootstrap_ci(stats: RaceLevelStats, metric: str, n_draws: int = 1000, seed: int = 0):
    """Resample races with replacement. Returns (point_estimate, ci_low, ci_high)."""
    point = stats.value(metric)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(stats.race_ids), size=(n_draws, len(stats.race_ids)))
    draws = stats._draws(metric, idx)
    return point, float(np.nanpercentile(draws, 2.5)), float(np.nanpercentile(draws, 97.5))


def paired_bootstrap_ci(stats_a: RaceLevelStats, stats_b: RaceLevelStats, metric: str,
                          n_draws: int = 1000, seed: int = 0):
    """CI on (metric(a) - metric(b)), resampling the same races each draw for both."""
    common = np.intersect1d(stats_a.race_ids, stats_b.race_ids)
    a, b = stats_a.subset(common), stats_b.subset(common)
    point = a.value(metric) - b.value(metric)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(common), size=(n_draws, len(common)))
    diffs = a._draws(metric, idx) - b._draws(metric, idx)
    return point, float(np.nanpercentile(diffs, 2.5)), float(np.nanpercentile(diffs, 97.5))


LOWER_IS_BETTER = {"log_loss", "brier_score"}


def verdict(ci_low: float, ci_high: float, metric: str | None = None) -> str:
    """ci_low/ci_high bound (model - baseline). For log_loss/brier_score, lower
    is better, so the sign is flipped before interpreting the interval."""
    if metric in LOWER_IS_BETTER:
        ci_low, ci_high = -ci_high, -ci_low
    if ci_low > 0:
        return "improved"
    if ci_high < 0:
        return "worsened"
    return "inconclusive"


# --- convenience wrappers for ad-hoc slices (breakdowns, one-off checks) ---
# Not for use in bootstrap loops: each call pays for its own groupby.

def hit_at_k(preds: pd.DataFrame, k: int) -> float:
    return RaceLevelStats.from_predictions(preds).value("hit_at_1" if k == 1 else "hit_at_3")


def mean_reciprocal_rank(preds: pd.DataFrame) -> float:
    return RaceLevelStats.from_predictions(preds).value("mrr")


def ndcg_at_k(preds: pd.DataFrame, k: int = 5) -> float:
    return RaceLevelStats.from_predictions(preds).value("ndcg_at_5")


def spearman_rank_correlation(preds: pd.DataFrame) -> float:
    return RaceLevelStats.from_predictions(preds).value("spearman")


def log_loss_race_normalized(preds: pd.DataFrame) -> float:
    return RaceLevelStats.from_predictions(preds).value("log_loss")


def brier_score_race_normalized(preds: pd.DataFrame) -> float:
    return RaceLevelStats.from_predictions(preds).value("brier_score")


def expected_calibration_error(preds: pd.DataFrame, n_bins: int = 10) -> tuple[float, pd.DataFrame]:
    total = preds.groupby("race_id")["predicted_probability"].transform("sum")
    p = preds["predicted_probability"] / total.replace(0, np.nan)
    y = (preds["actual_position"] == 1).astype(float)
    bins = pd.cut(p, bins=np.linspace(0, 1, n_bins + 1), include_lowest=True)
    grouped = pd.DataFrame({"p": p, "y": y, "bin": bins}).groupby("bin", observed=True)
    reliability = grouped.agg(mean_pred=("p", "mean"), mean_actual=("y", "mean"), n=("y", "size")).reset_index()
    ece = float((reliability["n"] / reliability["n"].sum() *
                 (reliability["mean_pred"] - reliability["mean_actual"]).abs()).sum())
    return ece, reliability


def brier_skill_score(preds: pd.DataFrame, baseline_preds: pd.DataFrame) -> float:
    model_brier = brier_score_race_normalized(preds)
    baseline_brier = brier_score_race_normalized(baseline_preds)
    if baseline_brier == 0:
        return np.nan
    return 1 - model_brier / baseline_brier
