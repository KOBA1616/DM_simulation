from dm_toolkit.gui.editor.text_generator import CardTextGenerator


def test_format_reaction_ninja_strike():
    r = {'type': 'NINJA_STRIKE', 'cost': 3}
    out = CardTextGenerator._format_reaction(r)
    assert 'ニンジャ・ストライク' in out
    assert '3' in out


def test_format_reaction_strike_back():
    r = {'type': 'STRIKE_BACK'}
    out = CardTextGenerator._format_reaction(r)
    assert out == 'ストライク・バック'


def test_format_reaction_unknown_falls_back():
    r = {'type': 'SOME_UNKNOWN_REACTION'}
    out = CardTextGenerator._format_reaction(r)
    # For unknown types, CardTextGenerator falls back to tr(type) which returns the key
    assert isinstance(out, str)
    assert out
