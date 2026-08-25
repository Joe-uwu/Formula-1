"""Ingest full seasons (races, circuits, drivers, constructors, results,
qualifying) from FastF1 for years the Kaggle CSVs don't cover (2025+).

Kaggle's `driverId`/`constructorId`/`circuitId`/`raceId` are arbitrary
integers with no natural key, so this module keeps its own get-or-create
mapping keyed on a stable natural key (driver (forename, surname) — NOT
`code`, which collides: Kaggle already has two different real drivers both
coded "VER" (Verstappen, Vergne), plus ALB/HAR/DOO/MAG/MSC/BIA — team name,
circuit name+location, (year, round)) and assigns new synthetic ids past the
current max when a row is new. Idempotent: rerunning resolves to the same
ids and results/qualifying upsert on their existing (race_id, driver_id)
unique constraint.
"""
import sys
from datetime import datetime
from pathlib import Path

import fastf1
import pandas as pd
from sqlalchemy import func, select
from sqlalchemy.dialects.postgresql import insert as pg_insert

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from f1.db.session import engine, Session
from f1.db.models import Circuit, Driver, Constructor, Race, Result, Qualifying, Season
from f1.ingest.id_resolver import IdResolver, slug as _slug

SOURCE = "fastf1"
CACHE_DIR = Path(__file__).resolve().parents[2] / ".fastf1_cache"


def ingest_season(year: int, rounds: list[int] | None = None) -> dict:
    CACHE_DIR.mkdir(exist_ok=True)
    fastf1.Cache.enable_cache(str(CACHE_DIR))
    now = datetime.utcnow()

    schedule = fastf1.get_event_schedule(year, include_testing=False)
    if rounds is not None:
        schedule = schedule[schedule["RoundNumber"].isin(rounds)]

    counts = {"races": 0, "results": 0, "qualifying": 0}

    with engine.begin() as conn:
        conn.execute(pg_insert(Season.__table__).values(
            year=year, url=None, source=SOURCE, ingested_at=now
        ).on_conflict_do_nothing(index_elements=["year"]))

        circuit_ids = IdResolver(conn, Circuit, "circuit_id",
                                   lambda r: (r.name, r.location))
        driver_ids = IdResolver(conn, Driver, "driver_id", lambda r: (r.forename.lower(), r.surname.lower()))
        constructor_ids = IdResolver(conn, Constructor, "constructor_id", lambda r: r.name)
        next_result_id = conn.execute(select(func.coalesce(func.max(Result.result_id), 0))).scalar()
        next_qualify_id = conn.execute(select(func.coalesce(func.max(Qualifying.qualify_id), 0))).scalar()

        for _, event in schedule.iterrows():
            round_no = int(event["RoundNumber"])
            if round_no == 0:  # testing rounds
                continue
            event_date = event["EventDate"].date()
            if event_date > datetime.utcnow().date():
                continue  # hasn't happened yet — nothing to ingest

            circuit_id = circuit_ids.get_or_create(
                (event["Location"], event["Location"]),
                lambda new_id: dict(circuit_id=new_id, ref=_slug(event["Location"]),
                                     name=event["Location"], location=event["Location"],
                                     country=event["Country"], source=SOURCE, ingested_at=now),
            )

            existing_race = conn.execute(select(Race.race_id).where(
                Race.year == year, Race.round == round_no
            )).scalar()
            if existing_race is None:
                max_race_id = conn.execute(select(func.coalesce(func.max(Race.race_id), 0))).scalar()
                race_id = max_race_id + 1
                conn.execute(pg_insert(Race.__table__).values(
                    race_id=race_id, year=year, round=round_no, circuit_id=circuit_id,
                    name=event["EventName"], date=event_date, time=None,
                    source=SOURCE, ingested_at=now,
                ))
            else:
                race_id = existing_race
            counts["races"] += 1

            try:
                quali = fastf1.get_session(year, round_no, "Q")
                quali.load(laps=False, telemetry=False, weather=False, messages=False)
                for _, r in quali.results.iterrows():
                    driver_id = driver_ids.get_or_create(
                        (r["FirstName"].lower(), r["LastName"].lower()),
                        lambda new_id, r=r: dict(driver_id=new_id, ref=_slug(r["FullName"]),
                                                   number=int(r["DriverNumber"]) if pd.notna(r["DriverNumber"]) else None,
                                                   code=r["Abbreviation"], forename=r["FirstName"],
                                                   surname=r["LastName"], dob=None, nationality=None,
                                                   source=SOURCE, ingested_at=now),
                    )
                    constructor_id = constructor_ids.get_or_create(
                        r["TeamName"],
                        lambda new_id, r=r: dict(constructor_id=new_id, ref=_slug(r["TeamName"]),
                                                   name=r["TeamName"], nationality=None,
                                                   source=SOURCE, ingested_at=now),
                    )
                    next_qualify_id += 1
                    quali_values = dict(
                        qualify_id=next_qualify_id,
                        race_id=race_id, driver_id=driver_id, constructor_id=constructor_id,
                        position=int(r["Position"]) if pd.notna(r["Position"]) else None,
                        q1=str(r.get("Q1")) if pd.notna(r.get("Q1")) else None,
                        q2=str(r.get("Q2")) if pd.notna(r.get("Q2")) else None,
                        q3=str(r.get("Q3")) if pd.notna(r.get("Q3")) else None,
                        q1_seconds=r["Q1"].total_seconds() if pd.notna(r.get("Q1")) else None,
                        q2_seconds=r["Q2"].total_seconds() if pd.notna(r.get("Q2")) else None,
                        q3_seconds=r["Q3"].total_seconds() if pd.notna(r.get("Q3")) else None,
                        source=SOURCE, ingested_at=now,
                    )
                    stmt = pg_insert(Qualifying.__table__).values(**quali_values)
                    stmt = stmt.on_conflict_do_update(
                        constraint="uq_qualifying_race_driver",
                        set_={k: stmt.excluded[k] for k in quali_values
                              if k not in ("qualify_id", "race_id", "driver_id")},
                    )
                    conn.execute(stmt)
                    counts["qualifying"] += 1
            except Exception as e:
                print(f"skip quali {year} round {round_no}: {e}")

            try:
                race = fastf1.get_session(year, round_no, "R")
                race.load(laps=False, telemetry=False, weather=False, messages=False)
                for _, r in race.results.iterrows():
                    driver_id = driver_ids.get_or_create(
                        (r["FirstName"].lower(), r["LastName"].lower()),
                        lambda new_id, r=r: dict(driver_id=new_id, ref=_slug(r["FullName"]),
                                                   number=int(r["DriverNumber"]) if pd.notna(r["DriverNumber"]) else None,
                                                   code=r["Abbreviation"], forename=r["FirstName"],
                                                   surname=r["LastName"], dob=None, nationality=None,
                                                   source=SOURCE, ingested_at=now),
                    )
                    constructor_id = constructor_ids.get_or_create(
                        r["TeamName"],
                        lambda new_id, r=r: dict(constructor_id=new_id, ref=_slug(r["TeamName"]),
                                                   name=r["TeamName"], nationality=None,
                                                   source=SOURCE, ingested_at=now),
                    )
                    next_result_id += 1
                    result_values = dict(
                        result_id=next_result_id,
                        race_id=race_id, driver_id=driver_id, constructor_id=constructor_id,
                        grid=int(r["GridPosition"]) if pd.notna(r["GridPosition"]) else None,
                        position=int(r["Position"]) if pd.notna(r["Position"]) else None,
                        position_order=int(r["Position"]) if pd.notna(r["Position"]) else 99,
                        points=float(r["Points"]) if pd.notna(r["Points"]) else 0.0,
                        laps=None, status=str(r.get("Status", "")),
                        milliseconds=None, fastest_lap_rank=None,
                        source=SOURCE, ingested_at=now,
                    )
                    stmt = pg_insert(Result.__table__).values(**result_values)
                    stmt = stmt.on_conflict_do_update(
                        constraint="uq_results_race_driver",
                        set_={k: stmt.excluded[k] for k in result_values
                              if k not in ("result_id", "race_id", "driver_id")},
                    )
                    conn.execute(stmt)
                    counts["results"] += 1
            except Exception as e:
                print(f"skip race {year} round {round_no}: {e}")

    return counts


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, required=True)
    args = ap.parse_args()
    print(ingest_season(args.year))
