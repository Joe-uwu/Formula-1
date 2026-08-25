"""Single source of truth for the train/val/test split, so the training run
and the live prediction pipeline hash to (and thus load) the same model."""
import hashlib
import json

SPLIT_CONFIG = {
    "start_date": "2015-01-01",
    "end_date": "2025-12-31",
    "val_start_date": "2024-01-01",   # train: [start, val_start)
    "test_start_date": "2025-01-01",  # val:   [val_start, test_start); test: [test_start, end)
}


def config_hash(split_config: dict = SPLIT_CONFIG) -> str:
    return hashlib.sha256(json.dumps(split_config, sort_keys=True).encode()).hexdigest()[:12]
