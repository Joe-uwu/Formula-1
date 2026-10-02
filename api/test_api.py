"""Smoke tests against the real DB (docker compose up -d db). Not a mock of
the pipeline — these hit the same Postgres instance the app runs against, so
a passing run means the endpoints actually produce real, DB-backed JSON.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)


def test_track_record_shape():
    r = client.get("/track-record")
    assert r.status_code == 200
    body = r.json()
    assert body["n_races"] > 0
    for system in ("model", "pole_sitter"):
        assert 0.0 <= body[system]["hit_at_1"] <= 1.0
        assert 0.0 <= body[system]["hit_at_3"] <= 1.0


def test_history_known_race():
    r = client.get("/races/history", params={"year": 2024, "round": 1})
    assert r.status_code == 200
    body = r.json()
    assert body["race"]["name"] == "Bahrain Grand Prix"
    preds = body["predictions"]
    assert len(preds) > 0
    assert all(p["predicted_rank"] >= 1 for p in preds)
    winner = [p for p in preds if p["actual_position"] == 1]
    assert len(winner) == 1


def test_history_unknown_race_404():
    r = client.get("/races/history", params={"year": 1900, "round": 1})
    assert r.status_code == 404


def test_recent_races_shape():
    r = client.get("/races/recent", params={"n": 5})
    assert r.status_code == 200
    races = r.json()["races"]
    assert 0 < len(races) <= 5
    for race in races:
        assert race["top_pick"]["code"]
        assert isinstance(race["hit"], bool)


def test_share_card_matches_history():
    r = client.get("/races/1121/share-card")
    assert r.status_code == 200
    body = r.json()
    assert body["top_pick"]["code"]
    assert len(body["field"]) > 0


def test_share_card_unknown_race_404():
    r = client.get("/races/999999999/share-card")
    assert r.status_code == 404


def test_upcoming_race_has_valid_status():
    r = client.get("/races/upcoming")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] in ("ok", "qualifying_not_done")
    if body["status"] == "ok":
        assert len(body["predictions"]) > 0
    else:
        assert "next_race" in body and "message" in body


def test_upcoming_features_matches_upcoming_status():
    upcoming = client.get("/races/upcoming").json()
    r = client.get("/races/upcoming/830/features")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == upcoming["status"]
    if body["status"] == "ok":
        assert set(body["features"]) == {
            "grid_position", "quali_position", "avg_quali_last5", "avg_finish_last5",
            "wins_last5", "avg_finish_last3", "wins_last3", "driver_points_cum",
            "constructor_points_cum", "constructor_wins_cum", "constructor_avg_finish_last3",
            "circuit_win_rate",
        }


if __name__ == "__main__":
    import traceback
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS {t.__name__}")
        except Exception:
            failed += 1
            print(f"FAIL {t.__name__}")
            traceback.print_exc()
    sys.exit(1 if failed else 0)
