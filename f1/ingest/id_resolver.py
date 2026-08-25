"""Get-or-create id mapping for tables with no natural key (Kaggle's
driverId/constructorId/circuitId are arbitrary integers). Shared by season
backfill ingestion (f1/ingest/fastf1_results_ingest.py) and live single-race
lineup resolution (f1/live/lineup.py) so both assign the same id to the same
driver/constructor/circuit instead of drifting apart.
"""
import re

from sqlalchemy import func, select
from sqlalchemy.dialects.postgresql import insert as pg_insert


def slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


class IdResolver:
    """In-memory get-or-create cache over a table's natural key, backed by
    the DB so ids stay stable across runs and across modules."""

    def __init__(self, conn, model, id_col: str, key_fn):
        self.conn, self.model, self.id_col, self.key_fn = conn, model, id_col, key_fn
        table = model.__table__
        self.max_id = conn.execute(select(func.coalesce(func.max(getattr(table.c, id_col)), 0))).scalar()
        self.cache: dict = {}
        for row in conn.execute(select(model)).all():
            self.cache[key_fn(row)] = getattr(row, id_col)

    def get_or_create(self, key, row_factory) -> int:
        if key in self.cache:
            return self.cache[key]
        self.max_id += 1
        new_id = self.max_id
        values = row_factory(new_id)
        self.conn.execute(pg_insert(self.model.__table__).values(**values))
        self.cache[key] = new_id
        return new_id
