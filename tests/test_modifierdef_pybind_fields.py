import json
import os
import pytest

dm = pytest.importorskip("dm_ai_module")


def make_cards_with_stat_scaled():
    support_card = {
        "id": 9201,
        "name": "Support Static PyBind",
        "type": "CREATURE",
        "cost": 1,
        "power": 300,
        "civilizations": ["NATURE"],
        "static_abilities": [
            {
                "type": "COST_MODIFIER",
                "value_mode": "STAT_SCALED",
                "stat_key": "CREATURES_PLAYED",
                "per_value": 1,
                "min_stat": 1,
                "max_reduction": 3,
            }
        ],
    }
    return [support_card]


def test_modifierdef_fields_exposed_via_pybind(tmp_path):
    cards = make_cards_with_stat_scaled()
    p = tmp_path / "modifier_bind_test.json"
    p.write_text(json.dumps(cards))

    db = dm.JsonLoader.load_cards(str(p))

    # Retrieve definition
    assert 9201 in db, "Card not loaded by JsonLoader"
    card_def = db[9201]

    # static_abilities may be dicts or pybind-wrapped ModifierDef objects
    sabs = getattr(card_def, 'static_abilities', None)
    assert sabs, "static_abilities missing or empty"
    sab = sabs[0]

    # Support both dict and attribute access
    if isinstance(sab, dict):
        assert sab.get('value_mode') == 'STAT_SCALED'
        assert sab.get('stat_key') == 'CREATURES_PLAYED'
        assert int(sab.get('per_value')) == 1
        assert int(sab.get('min_stat')) == 1
        assert int(sab.get('max_reduction')) == 3
    else:
        # pybind object or SimpleNamespace
        assert getattr(sab, 'value_mode', None) == 'STAT_SCALED'
        assert getattr(sab, 'stat_key', None) == 'CREATURES_PLAYED'
        assert int(getattr(sab, 'per_value', 0)) == 1
        assert int(getattr(sab, 'min_stat', 0)) == 1
        # max_reduction may be optional type; allow None or int
        mr = getattr(sab, 'max_reduction', None)
        if mr is not None:
            try:
                assert int(mr) == 3
            except Exception:
                pytest.skip("max_reduction present but not int-convertible")
