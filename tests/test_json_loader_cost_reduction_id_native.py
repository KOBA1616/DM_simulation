import json

import pytest


dm = pytest.importorskip("dm_ai_module")


def _get_card_def(card_map, card_id: int):
    try:
        return card_map[card_id]
    except Exception:
        return card_map.get(card_id)


def test_json_loader_assigns_missing_cost_reduction_id(tmp_path):
    cards = [
        {
            "id": 9901,
            "name": "Native Loader ID Migration",
            "type": "CREATURE",
            "cost": 5,
            "power": 1000,
            "civilizations": ["NATURE"],
            "cost_reductions": [
                {
                    "type": "PASSIVE",
                    "reduction_amount": 2
                }
            ],
        }
    ]

    p = tmp_path / "cards_missing_cr_id.json"
    p.write_text(json.dumps(cards), encoding="utf-8")

    card_map = dm.JsonLoader.load_cards(str(p))
    card_def = _get_card_def(card_map, 9901)
    assert card_def is not None

    # Regression prevention: C++ JsonLoader must assign a stable non-empty id
    # for cost_reductions entries that omit id in source JSON.
    if hasattr(card_def, "get_cost_reductions_size") and hasattr(card_def, "get_cost_reduction"):
        assert card_def.get_cost_reductions_size() == 1
        cr0 = card_def.get_cost_reduction(0)
        assert cr0 is not None
        assert isinstance(cr0.id, str) and len(cr0.id) > 0
    else:
        # Fallback path for environments exposing only the vector field.
        crs = getattr(card_def, "cost_reductions", [])
        assert len(crs) == 1
        assert isinstance(crs[0].id, str) and len(crs[0].id) > 0
