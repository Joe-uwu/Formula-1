"""Parse a qualifying lap time string to seconds. Two source formats show up
in the qualifying table: Kaggle's "M:SS.mmm" (e.g. "1:26.572") and FastF1's
pandas Timedelta repr ("0 days 00:01:12.695000")."""
import re

import pandas as pd

_KAGGLE_RE = re.compile(r"^(\d+):(\d+(?:\.\d+)?)$")


def parse_time_to_seconds(value) -> float | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    s = str(value).strip()
    if not s or s == "\\N":
        return None

    m = _KAGGLE_RE.match(s)
    if m:
        minutes, seconds = m.groups()
        return int(minutes) * 60 + float(seconds)

    try:
        return pd.Timedelta(s).total_seconds()
    except (ValueError, TypeError):
        return None
