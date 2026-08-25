"""Point-in-time feature queries.

Every function takes `as_of` (a race date) as a REQUIRED argument with no
default, and computes trailing/cumulative stats with window functions whose
frame stops at "1 PRECEDING" relative to the row being scored. That frame
boundary is what makes leakage structurally impossible: a race dated
`as_of` can never see its own row or any later row, regardless of what
`as_of` is passed in.
"""
from datetime import date

import pandas as pd
from sqlalchemy import text
from sqlalchemy.engine import Connection


def driver_form(conn: Connection, as_of: date) -> pd.DataFrame:
    """Per-driver trailing form: qualifying/finish rolling averages, win counts, cumulative points."""
    sql = text("""
        WITH base AS (
            SELECT
                res.driver_id,
                ra.date AS race_date,
                res.position AS finish_position,
                res.points AS points,
                CASE WHEN res.position = 1 THEN 1 ELSE 0 END AS win,
                q.position AS quali_position
            FROM results res
            JOIN races ra ON ra.race_id = res.race_id
            LEFT JOIN qualifying q ON q.race_id = res.race_id AND q.driver_id = res.driver_id
        ),
        windowed AS (
            SELECT
                driver_id,
                race_date,
                AVG(quali_position) OVER w5 AS avg_quali_last5,
                AVG(finish_position) OVER w5 AS avg_finish_last5,
                SUM(win) OVER w5 AS wins_last5,
                AVG(finish_position) OVER w3 AS avg_finish_last3,
                SUM(win) OVER w3 AS wins_last3,
                SUM(points) OVER wall AS driver_points_cum,
                COUNT(*) OVER wall AS driver_races_before
            FROM base
            WINDOW
                w5 AS (PARTITION BY driver_id ORDER BY race_date ROWS BETWEEN 5 PRECEDING AND 1 PRECEDING),
                w3 AS (PARTITION BY driver_id ORDER BY race_date ROWS BETWEEN 3 PRECEDING AND 1 PRECEDING),
                wall AS (PARTITION BY driver_id ORDER BY race_date ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING)
        )
        SELECT * FROM windowed WHERE race_date = :as_of
    """)
    return pd.read_sql(sql, conn, params={"as_of": as_of})


def constructor_form(conn: Connection, as_of: date) -> pd.DataFrame:
    """Per-constructor trailing form: cumulative points/wins, rolling avg finish.

    A constructor fields two cars, so results has two rows per (constructor,
    race) — one per driver. Those are aggregated to a single per-race totals
    row *before* windowing (`race_totals` below); windowing directly over
    per-driver rows would give a constructor two rows per race with tied
    `race_date` values, which both duplicates output (breaking the one-row-
    per-driver merge downstream) and makes the "1 PRECEDING" frame boundary
    depend on an unspecified tie-break order between the two teammates'
    same-day rows.
    """
    sql = text("""
        WITH race_totals AS (
            SELECT
                res.constructor_id,
                ra.date AS race_date,
                SUM(res.points) AS race_points,
                MAX(CASE WHEN res.position = 1 THEN 1 ELSE 0 END) AS race_win,
                AVG(res.position) AS race_avg_finish
            FROM results res
            JOIN races ra ON ra.race_id = res.race_id
            GROUP BY res.constructor_id, ra.date
        ),
        windowed AS (
            SELECT
                constructor_id,
                race_date,
                SUM(race_points) OVER wall AS constructor_points_cum,
                SUM(race_win) OVER wall AS constructor_wins_cum,
                AVG(race_avg_finish) OVER w3 AS constructor_avg_finish_last3
            FROM race_totals
            WINDOW
                wall AS (PARTITION BY constructor_id ORDER BY race_date ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING),
                w3 AS (PARTITION BY constructor_id ORDER BY race_date ROWS BETWEEN 3 PRECEDING AND 1 PRECEDING)
        )
        SELECT * FROM windowed WHERE race_date = :as_of
    """)
    return pd.read_sql(sql, conn, params={"as_of": as_of})


def circuit_history(conn: Connection, as_of: date) -> pd.DataFrame:
    """Per-driver history at this specific race's circuit: win rate, avg finish, pass rate."""
    sql = text("""
        WITH target_race AS (
            SELECT circuit_id FROM races WHERE date = :as_of
        ),
        base AS (
            SELECT
                res.driver_id,
                ra.date AS race_date,
                res.position AS finish_position,
                res.grid AS grid_position,
                CASE WHEN res.position = 1 THEN 1.0 ELSE 0.0 END AS win
            FROM results res
            JOIN races ra ON ra.race_id = res.race_id
            WHERE ra.circuit_id = (SELECT circuit_id FROM target_race)
        ),
        windowed AS (
            SELECT
                driver_id,
                race_date,
                AVG(win) OVER wall AS circuit_win_rate,
                AVG(finish_position) OVER wall AS circuit_avg_finish,
                AVG(grid_position - finish_position) OVER w5 AS circuit_pass_rate_last5
            FROM base
            WINDOW
                wall AS (PARTITION BY driver_id ORDER BY race_date ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING),
                w5 AS (PARTITION BY driver_id ORDER BY race_date ROWS BETWEEN 5 PRECEDING AND 1 PRECEDING)
        )
        SELECT * FROM windowed WHERE race_date = :as_of
    """)
    return pd.read_sql(sql, conn, params={"as_of": as_of})


def grid_and_qualifying(conn: Connection, as_of: date) -> pd.DataFrame:
    """The target race's own grid/qualifying positions. Not a leak: known before lights out."""
    sql = text("""
        SELECT
            res.driver_id,
            res.constructor_id,
            res.grid AS grid_position,
            q.position AS quali_position,
            res.position AS actual_position
        FROM results res
        JOIN races ra ON ra.race_id = res.race_id
        LEFT JOIN qualifying q ON q.race_id = res.race_id AND q.driver_id = res.driver_id
        WHERE ra.date = :as_of
    """)
    return pd.read_sql(sql, conn, params={"as_of": as_of})


def qualifying_pace(conn: Connection, as_of: date) -> pd.DataFrame:
    """This race's own qualifying pace, in seconds — not a leak, qualifying
    happens before the race. Rank (grid/quali position) throws away
    magnitude: a 0.03s gap to pole and a 0.9s gap are very different races.
    Gaps are z-scored within the session (divided by that session's stddev)
    so eras and circuits with different lap times are comparable. Also
    includes this race's own teammate qualifying gap — a same-session,
    not-a-leak comparison of the two cars fielded by one constructor."""
    sql = text("""
        WITH quali AS (
            SELECT q.driver_id, q.constructor_id,
                   LEAST(q.q1_seconds, q.q2_seconds, q.q3_seconds) AS best_seconds
            FROM qualifying q
            JOIN races ra ON ra.race_id = q.race_id
            WHERE ra.date = :as_of
        ),
        session_stats AS (
            SELECT MIN(best_seconds) AS pole_seconds,
                   PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY best_seconds) AS median_seconds,
                   STDDEV(best_seconds) AS std_seconds
            FROM quali
        ),
        teammates AS (
            SELECT a.driver_id, b.best_seconds AS teammate_best_seconds
            FROM quali a
            JOIN quali b ON b.constructor_id = a.constructor_id AND b.driver_id != a.driver_id
        )
        SELECT
            quali.driver_id,
            (quali.best_seconds - session_stats.pole_seconds) / NULLIF(session_stats.std_seconds, 0) AS quali_gap_to_pole_norm,
            (quali.best_seconds - session_stats.median_seconds) / NULLIF(session_stats.std_seconds, 0) AS quali_gap_to_median_norm,
            quali.best_seconds - teammates.teammate_best_seconds AS teammate_quali_delta
        FROM quali
        CROSS JOIN session_stats
        LEFT JOIN teammates ON teammates.driver_id = quali.driver_id
    """)
    return pd.read_sql(sql, conn, params={"as_of": as_of})


def teammate_quali_delta_history(conn: Connection, as_of: date) -> pd.DataFrame:
    """Trailing 5-race average of a driver's qualifying gap to their own
    teammate. Separates driver form from car pace: a driver consistently
    beating a teammate in the same car is a signal nothing else in the
    feature set captures directly."""
    sql = text("""
        WITH quali AS (
            SELECT q.driver_id, q.constructor_id, ra.date AS race_date,
                   LEAST(q.q1_seconds, q.q2_seconds, q.q3_seconds) AS best_seconds
            FROM qualifying q
            JOIN races ra ON ra.race_id = q.race_id
        ),
        deltas AS (
            SELECT a.driver_id, a.race_date,
                   a.best_seconds - b.best_seconds AS teammate_delta
            FROM quali a
            JOIN quali b ON b.constructor_id = a.constructor_id AND b.driver_id != a.driver_id
                          AND b.race_date = a.race_date
        ),
        windowed AS (
            SELECT driver_id, race_date,
                AVG(teammate_delta) OVER (PARTITION BY driver_id ORDER BY race_date
                    ROWS BETWEEN 5 PRECEDING AND 1 PRECEDING) AS teammate_quali_delta_avg5
            FROM deltas
        )
        SELECT * FROM windowed WHERE race_date = :as_of
    """)
    return pd.read_sql(sql, conn, params={"as_of": as_of})


def constructor_reliability(conn: Connection, as_of: date) -> pd.DataFrame:
    """Trailing DNF rate per constructor over its last 10 races. The winner
    has to finish — nothing in the rest of the feature set represents that.
    Aggregated to one row per (constructor, race) first, same discipline as
    constructor_form: a 2-car team has two result rows per race."""
    sql = text("""
        WITH race_totals AS (
            SELECT res.constructor_id, ra.date AS race_date,
                   AVG(CASE WHEN res.position IS NULL THEN 1.0 ELSE 0.0 END) AS dnf_rate_this_race
            FROM results res
            JOIN races ra ON ra.race_id = res.race_id
            GROUP BY res.constructor_id, ra.date
        ),
        windowed AS (
            SELECT constructor_id, race_date,
                AVG(dnf_rate_this_race) OVER (PARTITION BY constructor_id ORDER BY race_date
                    ROWS BETWEEN 10 PRECEDING AND 1 PRECEDING) AS constructor_dnf_rate_last10
            FROM race_totals
        )
        SELECT * FROM windowed WHERE race_date = :as_of
    """)
    return pd.read_sql(sql, conn, params={"as_of": as_of})


def circuit_overtaking_difficulty(conn: Connection, as_of: date) -> pd.DataFrame:
    """Mean absolute grid-to-finish position change at this circuit, over
    prior races only (expanding window, shifted — same UNBOUNDED PRECEDING
    ... 1 PRECEDING discipline as every other feature here). Circuit-level,
    not per-driver: broadcast the same value to every driver in the race,
    same pattern as circuit_weather_history."""
    sql = text("""
        WITH target_race AS (SELECT circuit_id FROM races WHERE date = :as_of),
        race_totals AS (
            SELECT ra.circuit_id, ra.date AS race_date,
                   AVG(ABS(res.grid - res.position)) AS mean_abs_change
            FROM results res
            JOIN races ra ON ra.race_id = res.race_id
            WHERE ra.circuit_id = (SELECT circuit_id FROM target_race)
              AND res.grid IS NOT NULL AND res.position IS NOT NULL
            GROUP BY ra.circuit_id, ra.date
        ),
        windowed AS (
            SELECT circuit_id, race_date,
                AVG(mean_abs_change) OVER (ORDER BY race_date
                    ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) AS circuit_overtaking_difficulty
            FROM race_totals
        )
        SELECT * FROM windowed WHERE race_date = :as_of
    """)
    return pd.read_sql(sql, conn, params={"as_of": as_of})


def constructor_season_pace(conn: Connection, as_of: date) -> pd.DataFrame:
    """Rolling constructor qualifying-gap-to-pole within the CURRENT season
    only (expanding window restricted to races in the same year as as_of,
    shifted). Historical cumulative points are a poor proxy for how fast a
    car is right now — this is a direct, current-form pace measurement."""
    sql = text("""
        WITH target_race AS (SELECT year FROM races WHERE date = :as_of),
        pole AS (
            SELECT race_id, MIN(LEAST(q1_seconds, q2_seconds, q3_seconds)) AS pole_seconds
            FROM qualifying GROUP BY race_id
        ),
        base AS (
            SELECT q.constructor_id, ra.date AS race_date,
                   AVG(LEAST(q.q1_seconds, q.q2_seconds, q.q3_seconds) - pole.pole_seconds) AS gap_to_pole_this_race
            FROM qualifying q
            JOIN races ra ON ra.race_id = q.race_id
            JOIN pole ON pole.race_id = q.race_id
            WHERE ra.year = (SELECT year FROM target_race)
            GROUP BY q.constructor_id, ra.date
        ),
        windowed AS (
            SELECT constructor_id, race_date,
                AVG(gap_to_pole_this_race) OVER (PARTITION BY constructor_id ORDER BY race_date
                    ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) AS constructor_season_pace_gap
            FROM base
        )
        SELECT * FROM windowed WHERE race_date = :as_of
    """)
    return pd.read_sql(sql, conn, params={"as_of": as_of})


def circuit_weather_history(conn: Connection, as_of: date) -> pd.DataFrame:
    """Circuit-level historical weather pattern (not this race's actual weather, which isn't
    known pre-race) — a proxy signal, strictly from past races at the same circuit."""
    sql = text("""
        WITH target_race AS (
            SELECT circuit_id FROM races WHERE date = :as_of
        ),
        base AS (
            SELECT
                ra.date AS race_date,
                w.track_temp_avg,
                w.wind_speed_avg,
                CASE WHEN w.rainfall THEN 1.0 ELSE 0.0 END AS rain
            FROM races ra
            JOIN weather w ON w.race_id = ra.race_id
            WHERE ra.circuit_id = (SELECT circuit_id FROM target_race)
        ),
        windowed AS (
            SELECT
                race_date,
                AVG(track_temp_avg) OVER wall AS hist_track_temp_avg,
                AVG(wind_speed_avg) OVER wall AS hist_wind_speed_avg,
                AVG(rain) OVER wall AS hist_rain_rate
            FROM base
            WINDOW wall AS (ORDER BY race_date ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING)
        )
        SELECT * FROM windowed WHERE race_date = :as_of
    """)
    return pd.read_sql(sql, conn, params={"as_of": as_of})
