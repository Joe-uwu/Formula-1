"""Proves the point-in-time feature queries cannot see the row they're scoring
or anything after it. Requires a live, ingested Postgres (see test_ingestion.py).
"""
import sys
from pathlib import Path

import pandas as pd
import pytest
from sqlalchemy import text

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from f1.db.session import engine
from f1.features.queries import driver_form, circuit_history, constructor_form, circuit_weather_history


def _db_available():
    try:
        with engine.connect() as c:
            c.execute(text("SELECT 1"))
        return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _db_available(), reason="no live Postgres to test against")


def test_first_career_race_has_no_trailing_history():
    """A driver's chronologically first race must show NULL trailing stats.
    If the window frame included the current row, avg_finish_last5 etc. would
    equal that row's own value instead of NULL — this is the tell for leakage."""
    with engine.connect() as conn:
        driver_id, first_date = conn.execute(text("""
            SELECT res.driver_id, MIN(ra.date)
            FROM results res JOIN races ra ON ra.race_id = res.race_id
            GROUP BY res.driver_id
            ORDER BY MIN(ra.date) DESC
            LIMIT 1
        """)).one()

        df = driver_form(conn, first_date)
    row = df[df["driver_id"] == driver_id]
    assert not row.empty
    assert row["avg_finish_last5"].isna().all()
    assert row["driver_races_before"].iloc[0] == 0


def test_trailing_stats_match_manual_pandas_computation():
    """Cross-check the SQL window-function result against an independent pandas
    computation restricted to strictly-earlier rows, for a driver with history."""
    with engine.connect() as conn:
        driver_id, as_of, expected_races_before = conn.execute(text("""
            SELECT driver_id, race_date, n FROM (
                SELECT res.driver_id, ra.date AS race_date,
                       ROW_NUMBER() OVER (PARTITION BY res.driver_id ORDER BY ra.date) - 1 AS n
                FROM results res JOIN races ra ON ra.race_id = res.race_id
            ) t WHERE n = 6
            LIMIT 1
        """)).one()

        history = pd.read_sql(text("""
            SELECT ra.date AS race_date, res.position AS finish_position
            FROM results res JOIN races ra ON ra.race_id = res.race_id
            WHERE res.driver_id = :d AND ra.date < :as_of
            ORDER BY ra.date
        """), conn, params={"d": driver_id, "as_of": as_of})

        sql_df = driver_form(conn, as_of)

    expected_last5 = history.tail(5)["finish_position"].mean()
    actual = sql_df.loc[sql_df["driver_id"] == driver_id, "avg_finish_last5"].iloc[0]
    assert actual == pytest.approx(expected_last5)
    assert len(history) == expected_races_before


def test_circuit_history_excludes_current_race_result():
    """A driver's very first visit to a circuit must show NULL circuit history,
    even though they have a result *at* this race — proving the current row
    is excluded from its own historical aggregate."""
    with engine.connect() as conn:
        driver_id, circuit_first_date = conn.execute(text("""
            SELECT res.driver_id, MIN(ra.date)
            FROM results res JOIN races ra ON ra.race_id = res.race_id
            GROUP BY res.driver_id, ra.circuit_id
            ORDER BY MIN(ra.date) DESC
            LIMIT 1
        """)).one()
        df = circuit_history(conn, circuit_first_date)
    row = df[df["driver_id"] == driver_id]
    assert not row.empty
    assert pd.isna(row["circuit_win_rate"].iloc[0])


def test_constructor_form_one_row_per_constructor():
    """A constructor fields two cars — constructor_form must collapse them to
    one row per (constructor, race) before windowing, not leave two rows with
    an unspecified tie-break order between teammates on the same race date."""
    with engine.connect() as conn:
        as_of = conn.execute(text("SELECT MAX(date) FROM races WHERE date < CURRENT_DATE")).scalar()
        df = constructor_form(conn, as_of)
    assert not df.empty
    assert df["constructor_id"].is_unique


def test_circuit_weather_history_uses_only_past_races_own_weather():
    """Weather features must be historical circuit averages from PAST races,
    never the target race's own weather (which isn't knowable before it's
    run). Proof: change what a future race at this circuit reports for
    weather and confirm today's as_of value is completely unaffected —
    the query must not even be looking at that row."""
    with engine.connect() as conn:
        row = conn.execute(text("""
            SELECT ra.circuit_id, MIN(ra.date)
            FROM races ra JOIN weather w ON w.race_id = ra.race_id
            GROUP BY ra.circuit_id HAVING COUNT(*) > 1
            ORDER BY ra.circuit_id LIMIT 1
        """)).one_or_none()
        if row is None:
            pytest.skip("no circuit with 2+ weather-covered races to plant a sentinel on")
        circuit_id, as_of = row
        before = circuit_weather_history(conn, as_of)

        # A later race at the same circuit with an extreme, unmistakable weather
        # reading. If circuit_weather_history(as_of) changes because of this,
        # it read a future (or the target) row it should never touch.
        future_race_id = conn.execute(text(
            "SELECT race_id FROM races WHERE circuit_id = :c AND date > :d ORDER BY date LIMIT 1"
        ), {"c": circuit_id, "d": as_of}).scalar()

    if future_race_id is None:
        pytest.skip("no later race at this circuit to plant a sentinel on")

    with engine.begin() as conn:
        conn.execute(text("""
            INSERT INTO weather (race_id, air_temp_avg, track_temp_avg, humidity_avg,
                                   wind_speed_avg, rainfall, source, ingested_at)
            VALUES (:r, 999.0, 999.0, 999.0, 999.0, true, 'test_sentinel', now())
            ON CONFLICT (race_id) DO UPDATE SET track_temp_avg = 999.0, wind_speed_avg = 999.0, rainfall = true
        """), {"r": future_race_id})

    try:
        with engine.connect() as conn:
            after = circuit_weather_history(conn, as_of)
        pd.testing.assert_frame_equal(before.reset_index(drop=True), after.reset_index(drop=True))
    finally:
        with engine.begin() as conn:
            conn.execute(text("DELETE FROM weather WHERE source = 'test_sentinel'"))
