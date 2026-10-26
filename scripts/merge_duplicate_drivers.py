"""One-off repair: merge drivers that were split into two driver_ids because
FastF1 spelled their name differently between runs (see
f1.ingest.id_resolver.driver_key). Every row moves to the lowest id, which
keeps the driver's full career history attached. The name shown is the
newest row's spelling. Safe to rerun; does nothing once merged.

Run (DB up):  python scripts/merge_duplicate_drivers.py
"""
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sqlalchemy import text

from f1.db.session import engine
from f1.ingest.id_resolver import driver_key

with engine.begin() as conn:
    groups = defaultdict(list)
    for r in conn.execute(text("SELECT driver_id, forename, surname FROM drivers ORDER BY driver_id")):
        groups[driver_key(r.forename, r.surname)].append(r)

    for rows in groups.values():
        keep, newest = rows[0], rows[-1]
        for dup in rows[1:]:
            ids = dict(keep=keep.driver_id, dup=dup.driver_id)
            # Same race under both ids (a pre-race placeholder vs the real
            # result): drop whichever side has no classified position.
            for table in ("results", "qualifying"):
                conn.execute(text(f"""
                    DELETE FROM {table} t USING {table} o
                    WHERE t.race_id = o.race_id AND t.position IS NULL AND (
                        (t.driver_id = :dup AND o.driver_id = :keep) OR
                        (t.driver_id = :keep AND o.driver_id = :dup))"""), ids)
            for table in ("results", "qualifying", "feature_rows", "predictions"):
                conn.execute(text(f"UPDATE {table} SET driver_id = :keep WHERE driver_id = :dup"), ids)
            conn.execute(text("DELETE FROM drivers WHERE driver_id = :dup"), ids)
            print(f"merged {dup.driver_id} ({dup.forename} {dup.surname}) into {keep.driver_id}")
        if len(rows) > 1:
            conn.execute(text("UPDATE drivers SET forename = :f, surname = :s WHERE driver_id = :id"),
                         dict(f=newest.forename, s=newest.surname, id=keep.driver_id))

    # A driver can't appear twice in one race's predictions; if a merge made
    # that happen, abort the whole transaction rather than keep bad data.
    clash = conn.execute(text("""
        SELECT model_version, config_hash, race_id, driver_id FROM predictions
        GROUP BY 1, 2, 3, 4 HAVING COUNT(*) > 1 LIMIT 5""")).all()
    assert not clash, f"duplicate predictions after merge: {clash}"
