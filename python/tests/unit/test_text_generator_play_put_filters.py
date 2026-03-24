# -*- coding: utf-8 -*-

from dm_toolkit.gui.editor.text_generator import CardTextGenerator


def test_put_creature_uses_cost_and_type_without_duplicate_source_zone():
    text = CardTextGenerator._format_command(
        {
            "type": "PUT_CREATURE",
            "from_zone": "HAND",
            "amount": 1,
            "target_filter": {
                "zones": ["HAND"],
                "types": ["CREATURE"],
                "max_cost": 4,
            },
        }
    )

    assert "手札から" in text
    assert "コスト4以下のクリーチャー" in text
    assert "手札から手札の" not in text


def test_put_creature_supports_element_cost_filter_from_target_filter():
    text = CardTextGenerator._format_command(
        {
            "type": "PUT_CREATURE",
            "from_zone": "HAND",
            "amount": 1,
            "target_filter": {
                "zones": ["HAND"],
                "types": ["ELEMENT"],
                "max_cost": 2,
            },
        }
    )

    assert "手札から" in text
    assert "コスト2以下のエレメント" in text
    assert "手札から手札の" not in text


def test_play_from_zone_uses_cost_type_and_selection_count():
    text = CardTextGenerator._format_command(
        {
            "type": "PLAY_FROM_ZONE",
            "from_zone": "GRAVEYARD",
            "target_group": "PLAYER_SELF",
            "amount": 5,
            "target_filter": {
                "types": ["CREATURE"],
                "count": 1,
            },
            "play_flags": True,
        }
    )

    assert "墓地から" in text
    assert "コスト5以下のクリーチャー" in text
    assert "1体選び" in text
    assert "コストを支払わずに召喚する" in text