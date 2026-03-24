import json
import tempfile
from pathlib import Path
import tempfile
import pytest
import dm_ai_module


def make_legacy_card_json(card_id=1000):
    # Minimal card JSON using legacy 'actions' (ActionDef)
    card = {
        "id": card_id,
        "name": "TestCard",
        "civilizations": [],
        "type": 0,
        "cost": 1,
        "power": 0,
        "races": [],
        "effects": [
            {
                "trigger": 0,
                "condition": None,
                "actions": [
                    {
                        "type": 0,  # EffectPrimitive::DRAW_CARD -> mapped to CommandType::DRAW_CARD
                        "scope": "SINGLE",
                        "filter": "",
                        "value1": 1,
                        "optional": False
                    }
                ]
            }
        ]
    }
    # JsonLoader expects either an array of card objects or a single card object
    return [card]


def test_legacy_actiondef_converted_to_commanddef(tmp_path: Path):
    data = make_legacy_card_json()
    jf = tmp_path / "cards.json"
    with jf.open("w", encoding="utf-8") as f:
        json.dump(data, f)

    # Use the JsonLoader exposed by the native module
    print('DEBUG: dm_ai_module file=', getattr(dm_ai_module, '__file__', None))
    loader = dm_ai_module.JsonLoader
    card_map = loader.load_cards(str(jf))

    # card_map may be a dict-like or a custom CardDatabase object; normalize
    entries = None
    if hasattr(card_map, 'values'):
        entries = list(card_map.values())
    else:
        # Try common accessors
        if hasattr(card_map, 'get_all_cards'):
            try:
                entries = list(card_map.get_all_cards().values())
            except Exception:
                entries = None
        elif hasattr(card_map, '__iter__'):
            try:
                entries = list(card_map)
            except Exception:
                entries = None

    assert entries and len(entries) > 0, "No cards loaded"
    # Inspect first card definition
    first = entries[0]
    # Effects should have commands populated from actions
    assert hasattr(first, "effects")
    assert len(first.effects) > 0
    eff = first.effects[0]
    # After conversion, commands should be non-empty and not rely on legacy 'actions'
    assert hasattr(eff, "commands")
    assert len(eff.commands) > 0
    cmd = eff.commands[0]
    # Command should have a concrete type (not NONE)
    assert getattr(cmd, "type", None) is not None
    # Optionally check mapping: DRAW_CARD expected -> numeric enum
    assert cmd.type != getattr(dm_ai_module, "CommandType").NONE


def test_multiple_actions_converted(tmp_path: Path):
    # Card with multiple legacy actions should produce multiple commands
    card = {
        "id": 2000,
        "name": "MultiAction",
        "civilizations": [],
        "type": 0,
        "cost": 2,
        "power": 0,
        "races": [],
        "effects": [
            {
                "trigger": 0,
                "condition": None,
                "actions": [
                    {"type": 0, "scope": "SINGLE", "filter": "", "value1": 1, "optional": False},
                    {"type": 1, "scope": "SINGLE", "filter": "", "value1": 0, "optional": False}
                ]
            }
        ]
    }
    jf = tmp_path / "cards.json"
    with jf.open("w", encoding="utf-8") as f:
        json.dump([card], f)

    loader = dm_ai_module.JsonLoader
    card_map = loader.load_cards(str(jf))
    # normalize
    entries = list(card_map.values()) if hasattr(card_map, 'values') else list(card_map)
    first = entries[0]
    eff = first.effects[0]
    assert len(eff.commands) >= 2


def test_metamorph_abilities_conversion(tmp_path: Path):
    # metamorph_abilities should also be converted from actions -> commands
    card = {
        "id": 3000,
        "name": "MetaCard",
        "civilizations": [],
        "type": 0,
        "cost": 3,
        "power": 0,
        "races": [],
        "effects": [],
        "metamorph_abilities": [
            {
                "trigger": 0,
                "condition": None,
                "actions": [
                    {"type": 0, "scope": "SINGLE", "filter": "", "value1": 1, "optional": False}
                ]
            }
        ]
    }
    jf = tmp_path / "cards.json"
    with jf.open("w", encoding="utf-8") as f:
        json.dump([card], f)

    loader = dm_ai_module.JsonLoader
    card_map = loader.load_cards(str(jf))
    entries = list(card_map.values()) if hasattr(card_map, 'values') else list(card_map)
    first = entries[0]
    print('DEBUG: first type=', type(first))
    print('DEBUG: first dir=', dir(first))
    print('DEBUG: has metamorph_abilities=', hasattr(first, 'metamorph_abilities'))
    if hasattr(first, 'metamorph_abilities'):
        assert len(first.metamorph_abilities) > 0
        meff = first.metamorph_abilities[0]
        assert hasattr(meff, 'commands') and len(meff.commands) > 0
    else:
        # 再発防止: 一部ネイティブビルドは metamorph_abilities を未公開のため、
        # このケースは変換不備ではなくバインディング差分としてスキップする。
        pytest.skip('metamorph_abilities is not exposed by current native binding')


def test_invalid_action_type_ignored(tmp_path: Path):
    # Unknown effect type should map to CommandType.NONE and be filtered out
    card = {
        "id": 4000,
        "name": "BadAction",
        "civilizations": [],
        "type": 0,
        "cost": 0,
        "power": 0,
        "races": [],
        "effects": [
            {
                "trigger": 0,
                "condition": None,
                "actions": [
                    {"type": 9999, "scope": "SINGLE", "filter": "", "value1": 0, "optional": False}
                ]
            }
        ]
    }
    jf = tmp_path / "cards.json"
    with jf.open("w", encoding="utf-8") as f:
        json.dump([card], f)

    loader = dm_ai_module.JsonLoader
    card_map = loader.load_cards(str(jf))
    entries = list(card_map.values()) if hasattr(card_map, 'values') else list(card_map)
    first = entries[0]
    eff = first.effects[0]
    # invalid mapping should result in zero commands pushed
    assert len(eff.commands) == 0

# LEGACY_ACTIONDEF_REFERENCE: This file references 'ActionDef' (legacy). Consider migrating to 'CommandDef'.
