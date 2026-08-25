"""One-time load of race-weekend weather from FastF1 (available from ~2018 onward).
Matched to existing `races` rows by (year, round) since FastF1 uses the same numbering.
"""
import sys
from datetime import datetime
from pathlib import Path

import fastf1
from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert as pg_insert

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from f1.db.session import engine, Session
from f1.db.models import Race, Weather

SOURCE = "fastf1"
CACHE_DIR = Path(__file__).resolve().parents[2] / ".fastf1_cache"


def load_weather(start_year: int = 2018, end_year: int = 2024):
    CACHE_DIR.mkdir(exist_ok=True)
    fastf1.Cache.enable_cache(str(CACHE_DIR))
    now = datetime.utcnow()

    with Session() as session:
        races = session.execute(
            select(Race.race_id, Race.year, Race.round)
            .where(Race.year >= start_year, Race.year <= end_year)
        ).all()

    rows = []
    for race_id, year, rnd in races:
        try:
            sess = fastf1.get_session(year, rnd, "R")
            sess.load(laps=False, telemetry=False, weather=True, messages=False)
            w = sess.weather_data
            if w is None or w.empty:
                continue
            rows.append(dict(
                race_id=race_id,
                air_temp_avg=float(w["AirTemp"].mean()),
                track_temp_avg=float(w["TrackTemp"].mean()),
                humidity_avg=float(w["Humidity"].mean()),
                wind_speed_avg=float(w["WindSpeed"].mean()),
                rainfall=bool(w["Rainfall"].any()),
                source=SOURCE, ingested_at=now,
            ))
        except Exception as e:
            print(f"skip {year} round {rnd}: {e}")

    if not rows:
        return 0

    with engine.begin() as conn:
        table = Weather.__table__
        stmt = pg_insert(table).values(rows)
        update_cols = {c.name: stmt.excluded[c.name] for c in table.columns if c.name not in ("id",)}
        stmt = stmt.on_conflict_do_update(index_elements=["race_id"], set_=update_cols)
        conn.execute(stmt)

    return len(rows)


if __name__ == "__main__":
    n = load_weather()
    print(f"Ingested weather for {n} races")
