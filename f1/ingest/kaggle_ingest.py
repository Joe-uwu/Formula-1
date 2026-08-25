"""One-time load of the Kaggle F1 dataset into Postgres. Idempotent: re-running upserts."""
import sys
from datetime import datetime, date
from pathlib import Path

import numpy as np
import pandas as pd
from sqlalchemy.dialects.postgresql import insert as pg_insert

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from f1.db.session import engine
from f1.db.models import (
    Base, Season, Circuit, Driver, Constructor, Race, Result, Qualifying,
)
from f1.ingest.time_parse import parse_time_to_seconds

SOURCE = "kaggle:rohanrao/formula-1-world-championship-1950-2020"


def _na(v):
    if pd.isna(v):
        return None
    return v


def download() -> Path:
    import kagglehub
    return Path(kagglehub.dataset_download("rohanrao/formula-1-world-championship-1950-2020"))


def _upsert(conn, model, rows: list[dict], pk_cols: list[str]):
    if not rows:
        return
    table = model.__table__
    stmt = pg_insert(table).values(rows)
    update_cols = {c.name: stmt.excluded[c.name] for c in table.columns if c.name not in pk_cols}
    stmt = stmt.on_conflict_do_update(index_elements=pk_cols, set_=update_cols)
    conn.execute(stmt)


def load_csvs(data_dir: Path):
    now = datetime.utcnow()

    na = ["\\N"]
    seasons = pd.read_csv(data_dir / "seasons.csv", na_values=na)
    circuits = pd.read_csv(data_dir / "circuits.csv", na_values=na)
    drivers = pd.read_csv(data_dir / "drivers.csv", na_values=na)
    constructors = pd.read_csv(data_dir / "constructors.csv", na_values=na)
    races = pd.read_csv(data_dir / "races.csv", na_values=na)
    results = pd.read_csv(data_dir / "results.csv", na_values=na)
    qualifying = pd.read_csv(data_dir / "qualifying.csv", na_values=na)
    status = pd.read_csv(data_dir / "status.csv").set_index("statusId")["status"].to_dict()

    counts = {
        "seasons": len(seasons), "circuits": len(circuits), "drivers": len(drivers),
        "constructors": len(constructors), "races": len(races), "results": len(results),
        "qualifying": len(qualifying),
    }

    with engine.begin() as conn:
        _upsert(conn, Season, [
            dict(year=int(r.year), url=_na(r.url), source=SOURCE, ingested_at=now)
            for r in seasons.itertuples()
        ], ["year"])

        _upsert(conn, Circuit, [
            dict(circuit_id=int(r.circuitId), ref=r.circuitRef, name=r.name,
                 location=_na(r.location), country=_na(r.country),
                 lat=_na(r.lat), lng=_na(r.lng), alt=_na(r.alt),
                 source=SOURCE, ingested_at=now)
            for r in circuits.itertuples()
        ], ["circuit_id"])

        _upsert(conn, Driver, [
            dict(driver_id=int(r.driverId), ref=r.driverRef,
                 number=_na(r.number) and int(r.number), code=_na(r.code),
                 forename=r.forename, surname=r.surname,
                 dob=(pd.to_datetime(r.dob).date() if _na(r.dob) else None),
                 nationality=_na(r.nationality), source=SOURCE, ingested_at=now)
            for r in drivers.itertuples()
        ], ["driver_id"])

        _upsert(conn, Constructor, [
            dict(constructor_id=int(r.constructorId), ref=r.constructorRef, name=r.name,
                 nationality=_na(r.nationality), source=SOURCE, ingested_at=now)
            for r in constructors.itertuples()
        ], ["constructor_id"])

        _upsert(conn, Race, [
            dict(race_id=int(r.raceId), year=int(r.year), round=int(r.round),
                 circuit_id=int(r.circuitId), name=r.name,
                 date=pd.to_datetime(r.date).date(), time=_na(r.time),
                 source=SOURCE, ingested_at=now)
            for r in races.itertuples()
        ], ["race_id"])

        def to_int(v):
            v = _na(v)
            return int(v) if v is not None else None

        def to_float(v):
            v = _na(v)
            return float(v) if v is not None else None

        # Source has ~176 legitimate duplicate (raceId, driverId) rows from
        # historical shared-drive entries (pre-1960s F1 let two drivers split
        # a car). Keep the better-classified row per pair so the unique
        # constraint holds without dropping real races.
        results = results.sort_values("positionOrder").drop_duplicates(
            subset=["raceId", "driverId"], keep="first"
        )

        _upsert(conn, Result, [
            dict(result_id=int(r.resultId), race_id=int(r.raceId), driver_id=int(r.driverId),
                 constructor_id=int(r.constructorId),
                 grid=to_int(r.grid),
                 position=to_int(r.position),
                 position_order=to_int(r.positionOrder),
                 points=to_float(r.points), laps=to_int(r.laps),
                 status=status.get(r.statusId, str(r.statusId)),
                 milliseconds=to_int(r.milliseconds),
                 fastest_lap_rank=to_int(r.rank),
                 source=SOURCE, ingested_at=now)
            for r in results.itertuples()
        ], ["result_id"])

        _upsert(conn, Qualifying, [
            dict(qualify_id=int(r.qualifyId), race_id=int(r.raceId), driver_id=int(r.driverId),
                 constructor_id=int(r.constructorId), position=to_int(r.position),
                 q1=_na(r.q1) and str(r.q1), q2=_na(r.q2) and str(r.q2), q3=_na(r.q3) and str(r.q3),
                 q1_seconds=parse_time_to_seconds(r.q1), q2_seconds=parse_time_to_seconds(r.q2),
                 q3_seconds=parse_time_to_seconds(r.q3),
                 source=SOURCE, ingested_at=now)
            for r in qualifying.itertuples()
        ], ["qualify_id"])

    return counts


def main():
    Base.metadata.create_all(engine)  # no-op once alembic has run; safety net for ad-hoc use
    data_dir = download()
    counts = load_csvs(data_dir)
    print("Ingested row counts:", counts)


if __name__ == "__main__":
    main()
