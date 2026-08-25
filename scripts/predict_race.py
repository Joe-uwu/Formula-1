"""Predict win probabilities for a race not yet in the database — resolves
the entry list and grid order live from FastF1 (requires qualifying to have
happened) and logs the prediction so it can be scored once the race is run.

Usage: python scripts/predict_race.py --year 2026 --round 14
       python scripts/predict_race.py --year 2026 --round 14 --score   (after the race has happened)
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from f1.live.lineup import NoGridOrderError
from f1.live.predict_upcoming import predict_race, backfill_and_score


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, required=True)
    ap.add_argument("--round", type=int, required=True)
    ap.add_argument("--score", action="store_true",
                     help="Race has already happened and its result is ingested: backfill actual "
                          "positions on the earlier live prediction instead of making a new one.")
    args = ap.parse_args()

    pd.set_option("display.max_rows", None)
    if args.score:
        scored = backfill_and_score(args.year, args.round)
        if scored is None:
            print("No unscored live predictions to backfill (predict first, or the result isn't ingested yet).")
        else:
            print(scored.to_string(index=False))
        return

    try:
        out = predict_race(args.year, args.round)
    except NoGridOrderError as e:
        print(f"Can't predict yet: {e}")
        sys.exit(1)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
