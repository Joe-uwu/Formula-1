"""Pure in-memory checks: point estimates match a naive per-metric reference,
and the vectorized bootstrap machinery produces sane, reproducible bounds.
No DB needed.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from f1.eval import metrics as M


def _toy_preds():
    # 3 races, 4 drivers each. Race 1: model nails the winner (rank 1 = actual 1).
    # Race 2: model puts the winner 2nd. Race 3: model puts the winner last.
    rows = []
    for race_id, winner_rank in [(1, 1), (2, 2), (3, 4)]:
        for driver_id in range(1, 5):
            actual_position = driver_id
            predicted_rank = 1 + (driver_id - winner_rank) % 4
            rows.append(dict(
                race_id=race_id, driver_id=driver_id,
                predicted_probability=1.0 / predicted_rank,
                predicted_rank=predicted_rank,
                actual_position=actual_position,
            ))
    return pd.DataFrame(rows)


def test_hit_at_1_matches_naive_count():
    preds = _toy_preds()
    stats = M.RaceLevelStats.from_predictions(preds)
    # race 1: winner predicted rank 1 -> hit. race 2: rank 2 -> miss. race 3: rank 4 -> miss.
    assert stats.value("hit_at_1") == 1 / 3


def test_mrr_matches_manual_reciprocal_ranks():
    preds = _toy_preds()
    stats = M.RaceLevelStats.from_predictions(preds)
    expected = np.mean([1 / 1, 1 / 2, 1 / 4])
    assert stats.value("mrr") == expected


def test_bootstrap_point_estimate_matches_direct_value():
    preds = _toy_preds()
    stats = M.RaceLevelStats.from_predictions(preds)
    point, lo, hi = M.bootstrap_ci(stats, "hit_at_1", n_draws=500, seed=1)
    assert point == stats.value("hit_at_1")
    assert lo <= point <= hi


def test_paired_bootstrap_zero_against_itself():
    """A model compared against an identical copy of itself must show a
    paired difference whose CI is exactly zero everywhere."""
    preds = _toy_preds()
    stats = M.RaceLevelStats.from_predictions(preds)
    point, lo, hi = M.paired_bootstrap_ci(stats, stats, "hit_at_1", n_draws=200, seed=2)
    assert point == 0.0
    assert lo == 0.0 and hi == 0.0
    assert M.verdict(lo, hi) == "inconclusive"


def test_bootstrap_is_reproducible_given_seed():
    preds = _toy_preds()
    stats = M.RaceLevelStats.from_predictions(preds)
    a = M.bootstrap_ci(stats, "spearman", n_draws=300, seed=42)
    b = M.bootstrap_ci(stats, "spearman", n_draws=300, seed=42)
    assert a == b


if __name__ == "__main__":
    test_hit_at_1_matches_naive_count()
    test_mrr_matches_manual_reciprocal_ranks()
    test_bootstrap_point_estimate_matches_direct_value()
    test_paired_bootstrap_zero_against_itself()
    test_bootstrap_is_reproducible_given_seed()
    print("ok")
