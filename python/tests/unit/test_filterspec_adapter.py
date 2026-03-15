from dm_toolkit.gui.editor.models.filterspec import FilterSpec, filterspec_from_legacy, filterspec_to_legacy


def test_filterspec_from_legacy_minimal():
    d = {"zones": ["HAND", "DECK"], "types": ["CREATURE"], "owner": "SELF"}
    f = filterspec_from_legacy(d)
    # Accept either FilterSpec instance or compatible object; validate fields
    assert getattr(f, 'zones') == ["HAND", "DECK"]
    assert getattr(f, 'types') == ["CREATURE"]
    assert getattr(f, 'owner') == "SELF"


def test_filterspec_roundtrip_extras_and_none():
    d = {"zones": ["BATTLE"], "extras": {"note": "keep"}, "count": 2}
    f = filterspec_from_legacy(d)
    out = filterspec_to_legacy(f)
    assert out["zones"] == ["BATTLE"]
    assert out["extras"]["note"] == "keep"
    assert out["count"] == 2
from dm_toolkit.gui.editor.models import dict_to_filterspec, filterspec_to_dict, FilterSpec


def test_dict_to_filterspec_basic():
    data = {
        'zones': ['HAND', 'BATTLE_ZONE'],
        'min_cost': 2,
        'is_tapped': 1,
        'unknown_key': 'foo'
    }
    fs = dict_to_filterspec(data)
    assert isinstance(fs, FilterSpec)
    assert fs.zones == ['HAND', 'BATTLE_ZONE']
    assert fs.min_cost == 2
    assert fs.is_tapped is True
    assert fs.extras.get('unknown_key') == 'foo'


def test_filterspec_to_dict_roundtrip():
    fs = FilterSpec(zones=['DECK'], civilizations=['WATER'], is_blocker=False, extras={'x': 1})
    d = filterspec_to_dict(fs)
    assert d['zones'] == ['DECK']
    assert d['civilizations'] == ['WATER']
    assert d['is_blocker'] is False
    assert d['x'] == 1
