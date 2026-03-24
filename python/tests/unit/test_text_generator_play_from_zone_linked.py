# -*- coding: utf-8 -*-

from dm_toolkit.gui.editor.text_generator import CardTextGenerator


def test_play_from_zone_linked_max_cost_uses_source_count_phrase():
    cmd = {
        "type": "PLAY_FROM_ZONE",
        "from_zone": "HAND",
        "target_group": "PLAYER_SELF",
        "target_filter": {
            "types": ["SPELL"],
            "max_cost": {"input_value_usage": "MAX_COST"},
            "count": 1,
        },
        "input_value_key": "var_QUERY_0",
        "_input_value_label": "マナゾーンのカード枚 (derived)",
        "play_flags": True,
        "optional": True,
    }

    text = CardTextGenerator._format_command(cmd)

    assert "手札から" in text
    assert "マナゾーンの枚数以下のコストの呪文" in text
    assert "1枚選び" in text
    assert "コストを支払わずに唱えてもよい" in text
    assert "(" not in text
    assert "（" not in text
