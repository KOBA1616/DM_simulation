from dm_toolkit.gui.editor.text_generator import CardTextGenerator


def test_format_reaction_counter_attack_missing():
    # RED: expect formatting for COUNTER_ATTACK to be a specific Japanese phrase
    r = {"type": "COUNTER_ATTACK", "cost": 2}
    out = CardTextGenerator._format_reaction(r)
    # Expectation: counter attack should be rendered as 'カウンター・アタック 2'
    assert out == "カウンター・アタック 2"
