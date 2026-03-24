from dm_toolkit.gui.editor.text_generator import CardTextGenerator


def test_multiple_reaction_mappings():
    cases = [
        ({"type": "COUNTER_ATTACK", "cost": 2}, "カウンター・アタック 2"),
        ({"type": "SHIELD_TRIGGER"}, "シールド・トリガー"),
        ({"type": "RETURN_ATTACK", "cost": 1}, "リターン・アタック 1"),
        ({"type": "ON_DEFEND"}, "守りのトリガー"),
    ]

    for inp, expected in cases:
        out = CardTextGenerator._format_reaction(inp)
        assert out == expected
