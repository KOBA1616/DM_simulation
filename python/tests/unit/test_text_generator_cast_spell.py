# -*- coding: utf-8 -*-

from dm_toolkit.gui.editor.text_generator import CardTextGenerator


def test_cast_spell_handles_none_filter_without_crashing():
    text = CardTextGenerator._format_command({"type": "CAST_SPELL", "target_filter": None})
    assert "唱える" in text


def test_cast_spell_describes_zone_cost_and_implicit_selection():
    cmd = {
        "type": "CAST_SPELL",
        "target_group": "PLAYER_SELF",
        "target_filter": {
            "zones": ["HAND", "GRAVEYARD"],
            "types": ["SPELL"],
            "max_cost": 5,
            "count": 1,
        },
    }

    text = CardTextGenerator._format_command(cmd)

    assert "手札または墓地から" in text
    assert "コスト5以下" in text
    assert "1枚選び" in text
    assert "コストを支払わずに唱える" in text


def test_cast_spell_uses_explicit_cost_when_present():
    cmd = {
        "type": "CAST_SPELL",
        "target_group": "PLAYER_SELF",
        "target_filter": {
            "zones": ["HAND"],
            "types": ["SPELL"],
        },
        "cost": 3,
        "play_flags": False,
    }

    text = CardTextGenerator._format_command(cmd)

    assert "手札から" in text
    assert "1枚選び" in text
    assert "コスト3を支払って唱える" in text


def test_cast_spell_omits_card_no_when_target_already_filtered_to_cards():
    cmd = {
        "type": "CAST_SPELL",
        "target_group": "PLAYER_SELF",
        "target_filter": {
            "zones": ["HAND"],
            "max_cost": 3,
            "count": 1,
        },
    }

    text = CardTextGenerator._format_command(cmd)

    assert "コスト3以下の呪文" in text
    assert "カードの呪文" not in text


def test_cast_spell_linked_max_cost_uses_source_count_phrase_without_metadata_parentheses():
    cmd = {
        "type": "CAST_SPELL",
        "target_group": "PLAYER_SELF",
        "target_filter": {
            "zones": ["HAND"],
            "types": ["SPELL"],
            "max_cost": {"input_value_usage": "MAX_COST"},
            "count": 1,
        },
        "input_value_key": "var_QUERY_0",
        "_input_value_label": "マナゾーンのカード枚 (derived)",
        "optional": True,
    }

    text = CardTextGenerator._format_command(cmd)

    assert "手札から" in text
    assert "マナゾーンの枚数以下のコストの呪文" in text
    assert "1枚選び" in text
    assert "唱えてもよい" in text
    assert "(" not in text
    assert "（" not in text