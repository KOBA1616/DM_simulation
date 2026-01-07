# -*- coding: utf-8 -*-

from dm_toolkit.gui.editor.text_generator import CardTextGenerator


def test_transition_zone_short_names_render_naturally() -> None:
    text = CardTextGenerator._format_command(
        {
            "type": "TRANSITION",
            "from_zone": "BATTLE",
            "to_zone": "GRAVEYARD",
            "amount": 1,
            "target_group": "NONE",
        }
    )
    # Should become a natural "destroy" sentence, not a raw zone dump.
    assert "破壊" in text
    assert "BATTLE" not in text


def test_choice_options_accept_command_dicts() -> None:
    text = CardTextGenerator._format_command(
        {
            "type": "CHOICE",
            "amount": 1,
            "flags": ["ALLOW_DUPLICATES"],
            "options": [
                [
                    {
                        "type": "TRANSITION",
                        "from_zone": "BATTLE",
                        "to_zone": "GRAVEYARD",
                        "amount": 1,
                        "target_group": "NONE",
                    }
                ],
                [
                    {
                        "type": "DRAW_CARD",
                        "amount": 1,
                    }
                ],
            ],
        }
    )
    # Should render both options without leaking enum-like tokens.
    assert "次の中から" in text
    assert "BATTLE" not in text
    assert "CHOICE" not in text
