"""Row counts vs source CSVs, and referential integrity. Requires a live DB
(DATABASE_URL, defaults to the docker-compose Postgres) that has already been
migrated and ingested via `python -m f1.ingest.kaggle_ingest`.
"""
import sys
from pathlib import Path

import pytest
from sqlalchemy import text

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from f1.db.session import engine
from f1.ingest.kaggle_ingest import download


def _db_available():
    try:
        with engine.connect() as c:
            c.execute(text("SELECT 1"))
        return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _db_available(), reason="no live Postgres to test against")


@pytest.fixture(scope="module")
def data_dir():
    return download()


# The DB is Kaggle ∪ FastF1(2025+) — Kaggle's `source` starts with "kaggle:",
# so row counts are checked against the Kaggle-sourced subset specifically,
# not the whole table.
@pytest.mark.parametrize("table,csv", [
    ("seasons", "seasons.csv"), ("circuits", "circuits.csv"), ("drivers", "drivers.csv"),
    ("constructors", "constructors.csv"), ("races", "races.csv"),
    ("qualifying", "qualifying.csv"),
])
def test_row_counts_match_source(data_dir, table, csv):
    import pandas as pd
    expected = len(pd.read_csv(data_dir / csv))
    with engine.connect() as conn:
        actual = conn.execute(text(f"SELECT COUNT(*) FROM {table} WHERE source LIKE 'kaggle%%'")).scalar()
    assert actual == expected, f"{table}: kaggle-sourced rows in db = {actual}, source csv has {expected}"


def test_results_row_count_matches_source_minus_known_duplicates(data_dir):
    """~176 source rows are legitimate duplicate (raceId, driverId) pairs from
    historical shared-drive entries; kaggle_ingest.py dedupes them deterministically."""
    import pandas as pd
    src = pd.read_csv(data_dir / "results.csv")
    expected = len(src.drop_duplicates(subset=["raceId", "driverId"]))
    with engine.connect() as conn:
        actual = conn.execute(text("SELECT COUNT(*) FROM results WHERE source LIKE 'kaggle%%'")).scalar()
    assert actual == expected, f"results: kaggle-sourced rows in db = {actual}, expected {expected} after dedup"


def test_fastf1_2025_season_present():
    """The 2025 season isn't in the Kaggle CSVs — confirms the FastF1 backfill
    (f1/ingest/fastf1_results_ingest.py) actually populated it."""
    with engine.connect() as conn:
        races = conn.execute(text("SELECT COUNT(*) FROM races WHERE year = 2025")).scalar()
        results = conn.execute(text(
            "SELECT COUNT(*) FROM results r JOIN races ra ON ra.race_id = r.race_id WHERE ra.year = 2025"
        )).scalar()
    assert races > 0, "2025 season not ingested from FastF1"
    assert results > 0, "2025 results not ingested from FastF1"


@pytest.mark.parametrize("child,parent,fk", [
    ("races", "seasons", "year"),
    ("races", "circuits", "circuit_id"),
    ("results", "races", "race_id"),
    ("results", "drivers", "driver_id"),
    ("results", "constructors", "constructor_id"),
    ("qualifying", "races", "race_id"),
    ("qualifying", "drivers", "driver_id"),
])
def test_no_orphan_foreign_keys(child, parent, fk):
    with engine.connect() as conn:
        orphans = conn.execute(text(
            f"SELECT COUNT(*) FROM {child} c "
            f"LEFT JOIN {parent} p ON p.{fk} = c.{fk} "
            f"WHERE p.{fk} IS NULL"
        )).scalar()
    assert orphans == 0


def test_no_constructor_fields_more_than_two_cars_per_race_modern_era():
    """Catches driver-identity mismatches, not just literal duplicate rows:
    a constructor's `code`s in Kaggle history aren't globally unique (two
    real drivers, e.g. Verstappen and Vergne, are both coded 'VER'), so a
    key-resolution bug can silently attribute one driver's race to another
    without ever creating a literal duplicate (race_id, driver_id) row.
    Scoped to 2015+ (this project's training range): the two-cars-per-team
    rule wasn't universal in early F1 — 1950s privateer entries routinely
    fielded dozens of cars under one constructor banner, which is real
    history, not a bug."""
    with engine.connect() as conn:
        over = conn.execute(text(
            "SELECT res.race_id, res.constructor_id, COUNT(*) c FROM results res "
            "JOIN races ra ON ra.race_id = res.race_id WHERE ra.year >= 2015 "
            "GROUP BY res.race_id, res.constructor_id HAVING COUNT(*) > 2"
        )).fetchall()
    assert over == [], f"constructor fielding >2 cars in one 2015+ race (likely a driver-identity bug): {over}"


def test_results_unique_race_driver():
    with engine.connect() as conn:
        dupes = conn.execute(text(
            "SELECT race_id, driver_id, COUNT(*) c FROM results "
            "GROUP BY race_id, driver_id HAVING COUNT(*) > 1"
        )).fetchall()
    assert dupes == []


def test_reingestion_is_idempotent(data_dir):
    from f1.ingest.kaggle_ingest import load_csvs
    with engine.connect() as conn:
        before = conn.execute(text("SELECT COUNT(*) FROM results")).scalar()
    load_csvs(data_dir)
    with engine.connect() as conn:
        after = conn.execute(text("SELECT COUNT(*) FROM results")).scalar()
    assert before == after
