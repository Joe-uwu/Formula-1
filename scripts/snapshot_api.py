"""Dump the API responses the dashboard reads to static JSON under
f1-dashboard/public/api/, so the built site can be hosted with no backend.
Paths mirror the API routes (+ ".json"); see f1-dashboard/src/api.js.

First pulls in any finished race results and scores the live predictions
made for them — without that, "next race" never advances past the last
race this DB saw finish.

Run (DB up):  python scripts/snapshot_api.py
"""
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fastapi.testclient import TestClient
from sqlalchemy import select

from api.main import app
from f1.db.models import Race, Result
from f1.db.session import engine
from f1.ingest.fastf1_results_ingest import ingest_season
from f1.live.predict_upcoming import backfill_and_score

OUT = ROOT / "f1-dashboard" / "public" / "api"
client = TestClient(app)


def dump(path: str, **params) -> dict:
    res = client.get(path, params=params)
    res.raise_for_status()
    dest = OUT / (path.lstrip("/") + ".json")
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(res.json()))
    print(f"wrote {dest.relative_to(ROOT)}")
    return res.json()


if __name__ == "__main__":
    year = datetime.utcnow().year
    with engine.connect() as conn:
        done = set(conn.scalars(
            select(Race.round).join(Result, Result.race_id == Race.race_id)
            .where(Race.year == year, Result.position.isnot(None)).distinct()
        ))
    # Only rounds still missing a result: keeps a cold-cache run well under
    # FastF1's API rate limit. Future rounds are skipped inside ingest_season.
    print("ingest", year, ingest_season(year, [r for r in range(1, 31) if r not in done]))
    backfill_and_score(year, 0)  # args unused; scores every unscored live prediction
    shutil.rmtree(OUT, ignore_errors=True)
    # n values match the calls in f1-dashboard/src/pages/Home.jsx — the static
    # site ignores query params, so these are the only variants it can serve.
    dump("/races/upcoming")
    dump("/track-record", n=10)
    recent = dump("/races/recent", n=8)
    for race in recent["races"]:
        dump(f"/races/{race['race_id']}/share-card")
    assert recent["races"], "no recent races snapshotted — is the DB populated?"
