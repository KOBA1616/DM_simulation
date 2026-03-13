import json
from pathlib import Path
import dm_ai_module


def make_null_heavy_card(card_id=9000):
    # Card with many explicit nulls in nested fields
    card = {
        "id": card_id,
        "name": None,
        "civilizations": None,
        "type": None,
        "cost": None,
        "power": None,
        "races": None,
        "effects": [
            {
                "trigger": None,
                "condition": None,
                "actions": None,
                "commands": None,
                "trigger_descriptor": None,
                "filter": None
            }
        ],
        "metamorph_abilities": None,
        "spell_side": None,
        "keywords": None,
        "reaction_abilities": None,
        "cost_reductions": None,
        "is_key_card": None,
        "ai_importance_score": None
    }
    return [card]


def test_json_loader_handles_explicit_nulls(tmp_path: Path):
    data = make_null_heavy_card()
    jf = tmp_path / "cards.json"
    with jf.open("w", encoding="utf-8") as f:
        json.dump(data, f)

    loader = dm_ai_module.JsonLoader
    # should not raise and should return a mapping with an entry
    card_map = loader.load_cards(str(jf))
    assert card_map is not None
    # normalize possible container types
    if hasattr(card_map, 'values'):
        entries = list(card_map.values())
    else:
        try:
            entries = list(card_map)
        except Exception:
            entries = []
    assert len(entries) == 1
    first = entries[0]
    # basic sanity: id present and effects present (possibly empty)
    assert getattr(first, 'id', None) is not None
    assert hasattr(first, 'effects')
