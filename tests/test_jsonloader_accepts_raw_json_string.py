import json
import os
import pytest

# Ensure Python fallback for reproducible behavior in CI/dev machines
os.environ.setdefault('DM_DISABLE_NATIVE', '1')

dm = pytest.importorskip("dm_ai_module")


def test_jsonloader_accepts_raw_json_string():
    cards = [
        {"id": 42, "name": "Raw", "type": "CREATURE", "cost": 1, "static_abilities": []}
    ]
    raw = json.dumps(cards)

    db = dm.JsonLoader.load_cards(raw)
    assert isinstance(db, dict)
    assert 42 in db
    cd = db[42]
    assert getattr(cd, 'name', None) == 'Raw'
