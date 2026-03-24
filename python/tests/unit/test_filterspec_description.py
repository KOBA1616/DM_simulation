from dm_toolkit.gui.editor.models import describe_filterspec


def test_describe_filterspec_minimal():
    d = {"zones": ["HAND", "DECK"], "types": ["CREATURE"], "owner": "SELF"}
    desc = describe_filterspec(d)
    assert "Zones: HAND, DECK" in desc
    assert "Types: CREATURE" in desc
    assert "Owner: SELF" in desc


def test_describe_filterspec_with_extras_and_flags():
    d = {"zones": ["BATTLE"], "extras": {"note": "x"}, "is_tapped": 1}
    desc = describe_filterspec(d)
    assert "Zones: BATTLE" in desc
    assert "Flags: is_tapped" in desc
    assert "Extras: 1 items" in desc
