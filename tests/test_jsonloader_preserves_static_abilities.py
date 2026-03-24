import json
import os
import pytest

# Ensure Python fallback for reproducible behavior in CI/dev machines
os.environ.setdefault('DM_DISABLE_NATIVE', '1')

dm = pytest.importorskip("dm_ai_module")


def test_jsonloader_preserves_static_abilities(tmp_path):
    cards = [
        {
            "id": 1,
            "name": "AbilityHolder",
            "type": "CREATURE",
            "cost": 2,
            "static_abilities": [
                {"type": "COST_MODIFIER", "value_mode": "FIXED", "value": 1}
            ],
        }
    ]

    p = tmp_path / "cards_preserve.json"
    p.write_text(json.dumps(cards), encoding="utf-8")

    db = dm.JsonLoader.load_cards(str(p))
    assert 1 in db
    cd = db[1]

    sabs = getattr(cd, "static_abilities", None)
    assert sabs is not None, "JsonLoader must preserve static_abilities"
    assert isinstance(sabs, (list, tuple)), "static_abilities should be a sequence"
    assert len(sabs) >= 1

    first = sabs[0]
    # Accept dict or simple namespace-like object
    if isinstance(first, dict):
        assert first.get("type") == "COST_MODIFIER"
    else:
        assert getattr(first, "type", None) == "COST_MODIFIER"
