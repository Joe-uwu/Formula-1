"""Resolve who's racing in a not-yet-ingested race, live from FastF1: the
entry list and the grid order from qualifying. Grid position is what
actually determines a race winner's odds, so prediction requires qualifying
to have happened — raises NoGridOrderError otherwise rather than guessing
from a stale lineup.

Ensures the race/circuit/season and any new drivers/constructors exist in
the DB first, using the same synthetic-id get-or-create scheme as the
season backfill ingestion (f1/ingest/fastf1_results_ingest.py), so the
point-in-time feature queries can run against this race's `as_of` date.
"""
import sys
from datetime import datetime
from pathlib import Path

import fastf1
import pandas as pd
from sqlalchemy import func, select
from sqlalchemy.dialects.postgresql import insert as pg_insert

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from f1.db.session import engine
from f1.db.models import Circuit, Driver, Constructor, Race, Result, Season
from f1.ingest.id_resolver import IdResolver, slug

SOURCE = "fastf1_live"
CACHE_DIR = Path(__file__).resolve().parents[2] / ".fastf1_cache"


def _enable_cache():
    CACHE_DIR.mkdir(exist_ok=True)
    fastf1.Cache.enable_cache(str(CACHE_DIR))


class NoGridOrderError(Exception):
    """Qualifying hasn't happened (or produced no classification) for this race yet."""


def _driver_constructor_rows(results: pd.DataFrame) -> list[dict]:
    """results: a FastF1 qualifying session.results DataFrame — its 'Position'
    column becomes both grid_position and quali_position."""
    rows = []
    for _, r in results.iterrows():
        pos = r["Position"]
        rows.append(dict(
            code=r["Abbreviation"], full_name=r["FullName"], forename=r["FirstName"],
            surname=r["LastName"], driver_number=r["DriverNumber"], team=r["TeamName"],
            grid_position=float(pos) if pd.notna(pos) else None,
            quali_position=float(pos) if pd.notna(pos) else None,
        ))
    return rows


def resolve_lineup(year: int, round_no: int) -> list[dict]:
    """Returns lineup_rows with real grid order. Raises NoGridOrderError if
    qualifying hasn't happened yet for this race."""
    _enable_cache()
    try:
        quali = fastf1.get_session(year, round_no, "Q")
        quali.load(laps=False, telemetry=False, weather=False, messages=False)
    except Exception as e:
        raise NoGridOrderError(f"Qualifying not available yet for {year} round {round_no}: {e}") from e
    if quali.results is None or quali.results.empty or not quali.results["Position"].notna().any():
        raise NoGridOrderError(f"Qualifying hasn't happened yet for {year} round {round_no}.")
    return _driver_constructor_rows(quali.results)


def ensure_race_and_lineup(year: int, round_no: int):
    """Returns (race_id, as_of_date, lineup_df) where lineup_df has one row
    per driver: driver_id, constructor_id, grid_position, quali_position,
    code, full_name, team. Inserts any new race/circuit/driver/constructor
    rows needed, tagged source='fastf1_live'."""
    _enable_cache()
    lineup_rows = resolve_lineup(year, round_no)
    event = fastf1.get_event(year, round_no)
    event_date = pd.Timestamp(event["EventDate"]).date()
    now = datetime.utcnow()

    with engine.begin() as conn:
        conn.execute(pg_insert(Season.__table__).values(
            year=year, url=None, source=SOURCE, ingested_at=now
        ).on_conflict_do_nothing(index_elements=["year"]))

        circuit_ids = IdResolver(conn, Circuit, "circuit_id", lambda r: (r.name, r.location))
        circuit_id = circuit_ids.get_or_create(
            (event["Location"], event["Location"]),
            lambda new_id: dict(circuit_id=new_id, ref=slug(event["Location"]),
                                 name=event["Location"], location=event["Location"],
                                 country=event["Country"], source=SOURCE, ingested_at=now),
        )

        existing_race_id = conn.execute(
            select(Race.race_id).where(Race.year == year, Race.round == round_no)
        ).scalar()
        if existing_race_id is None:
            max_race_id = conn.execute(select(func.coalesce(func.max(Race.race_id), 0))).scalar()
            race_id = max_race_id + 1
            conn.execute(pg_insert(Race.__table__).values(
                race_id=race_id, year=year, round=round_no, circuit_id=circuit_id,
                name=str(event["EventName"]), date=event_date, time=None,
                source=SOURCE, ingested_at=now,
            ))
        else:
            race_id = existing_race_id

        driver_ids = IdResolver(conn, Driver, "driver_id", lambda r: (r.forename.lower(), r.surname.lower()))
        constructor_ids = IdResolver(conn, Constructor, "constructor_id", lambda r: r.name)
        next_result_id = conn.execute(select(func.coalesce(func.max(Result.result_id), 0))).scalar()

        resolved = []
        for row in lineup_rows:
            driver_id = driver_ids.get_or_create(
                (row["forename"].lower(), row["surname"].lower()),
                lambda new_id, row=row: dict(driver_id=new_id, ref=slug(row["full_name"]),
                                               number=int(row["driver_number"]) if pd.notna(row["driver_number"]) else None,
                                               code=row["code"], forename=row["forename"], surname=row["surname"],
                                               dob=None, nationality=None, source=SOURCE, ingested_at=now),
            )
            constructor_id = constructor_ids.get_or_create(
                row["team"],
                lambda new_id, row=row: dict(constructor_id=new_id, ref=slug(row["team"]),
                                               name=row["team"], nationality=None, source=SOURCE, ingested_at=now),
            )
            resolved.append(dict(driver_id=driver_id, constructor_id=constructor_id,
                                   grid_position=row["grid_position"], quali_position=row["quali_position"],
                                   code=row["code"], full_name=row["full_name"], team=row["team"]))

            # The point-in-time queries (f1/features/queries.py) window over
            # each driver's own result row as the anchor for "everything
            # before this race" — a race that hasn't happened yet has no such
            # row. Insert a placeholder (never overwriting a real result, via
            # on_conflict_do_nothing) so those queries see the same shape they
            # always do, unmodified.
            next_result_id += 1
            grid = row["grid_position"]
            conn.execute(pg_insert(Result.__table__).values(
                result_id=next_result_id, race_id=race_id, driver_id=driver_id, constructor_id=constructor_id,
                grid=int(grid) if pd.notna(grid) else None, position=None, position_order=99,
                points=0.0, laps=None, status="Not started", milliseconds=None, fastest_lap_rank=None,
                source=SOURCE, ingested_at=now,
            ).on_conflict_do_nothing(constraint="uq_results_race_driver"))

    lineup_df = pd.DataFrame(resolved)
    return race_id, event_date, lineup_df
