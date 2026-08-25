"""Trivial baselines. The pole-sitter baseline is the yardstick every real
model has to beat: predict the pole-sitter (quali P1, falling back to grid P1)
wins with probability 1, everyone else 0; rank the field by grid position.
"""
import pandas as pd


def pole_sitter_predictions(race_df: pd.DataFrame) -> pd.DataFrame:
    """race_df: one row per driver for one race, with grid_position, quali_position,
    actual_position. Returns predicted_probability, predicted_rank per driver."""
    df = race_df.copy()
    rank_basis = df["quali_position"].fillna(df["grid_position"])
    df["predicted_rank"] = rank_basis.rank(method="first").astype(int)
    df["predicted_probability"] = (df["predicted_rank"] == 1).astype(float)
    return df


def uniform_predictions(race_df: pd.DataFrame) -> pd.DataFrame:
    """1/N for every driver — the floor a probabilistic model has to clear."""
    df = race_df.copy()
    df["predicted_probability"] = 1.0 / len(df)
    # Ranking metrics aren't meaningful for a uniform model; break ties on
    # driver_id only so predicted_rank is a valid, deterministic column.
    df["predicted_rank"] = df["driver_id"].rank(method="first").astype(int)
    return df
