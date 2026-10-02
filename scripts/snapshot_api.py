"""Dump the API responses the dashboard reads to static JSON under
f1-dashboard/public/api/, so the built site can be hosted with no backend.
Paths mirror the API routes (+ ".json"); see f1-dashboard/src/api.js.

Run (DB up):  python scripts/snapshot_api.py
"""
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fastapi.testclient import TestClient

from api.main import app

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
    shutil.rmtree(OUT, ignore_errors=True)
    # n values match the calls in f1-dashboard/src/pages/Home.jsx — the static
    # site ignores query params, so these are the only variants it can serve.
    dump("/races/upcoming")
    dump("/track-record", n=10)
    recent = dump("/races/recent", n=8)
    for race in recent["races"]:
        dump(f"/races/{race['race_id']}/share-card")
    assert recent["races"], "no recent races snapshotted — is the DB populated?"
