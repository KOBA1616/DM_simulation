"""Trigger scope + filter adjective generation tests.

Verifies CardTextGenerator applies scope and trigger_filter details to ON_PLAY/ON_CAST_SPELL/ON_OTHER_ENTER.
"""
from dm_toolkit.gui.editor.text_generator import CardTextGenerator


def test_on_play_with_civ_and_cost():
    # Effect: opponent's LIGHT creature with exact cost 3 enters
    effect = {
        "trigger": "ON_PLAY",
        "trigger_scope": "OPPONENT",
        "trigger_filter": {
            "civilizations": ["LIGHT"],
            "types": ["CREATURE"],
            "exact_cost": 3,
        },
        "commands": [{"type": "SHUFFLE_DECK"}],
    }
    text = CardTextGenerator._format_effect(effect, is_spell=False)
    assert "相手の光のコスト3のクリーチャーがバトルゾーンに出た時" in text


def test_on_cast_spell_with_input_link_max_cost():
    # Effect: self casts a spell with linked max_cost
    effect = {
        "trigger": "ON_CAST_SPELL",
        "trigger_scope": "SELF",
        "trigger_filter": {
            "types": ["SPELL"],
            "max_cost": {"input_value_usage": "MAX_COST"},
        },
        "commands": [{"type": "SHUFFLE_DECK"}],
    }
    text = CardTextGenerator._format_effect(effect, is_spell=True)
    assert "自分のコストその数以下の呪文を唱えた時" in text


def test_on_other_enter_with_race():
    # Effect: opponent's other Dragon creature enters
    effect = {
        "trigger": "ON_OTHER_ENTER",
        "trigger_scope": "OPPONENT",
        "trigger_filter": {
            "types": ["CREATURE"],
            "races": ["ドラゴン"],
        },
        "commands": [{"type": "SHUFFLE_DECK"}],
    }
    text = CardTextGenerator._format_effect(effect, is_spell=False)
    assert "相手の他のドラゴンのクリーチャーがバトルゾーンに出た時" in text
