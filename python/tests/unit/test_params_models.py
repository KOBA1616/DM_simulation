from dm_toolkit.gui.editor.models.params import QueryParams, TransitionParams, ModifierParams


def test_query_params_roundtrip():
    data = {"query_type": "SELECT_TARGET", "min": 1, "max": 3, "filter": {"cost": 3}}
    qp = QueryParams.model_validate(data)
    assert qp.query_type == "SELECT_TARGET"
    assert qp.min == 1
    assert qp.max == 3
    assert qp.filter["cost"] == 3


def test_transition_params_defaults_and_values():
    tp = TransitionParams.model_validate({})
    assert tp.amount == 1
    assert tp.preserve_order is False
    tp2 = TransitionParams.model_validate({"from_zone": "DECK", "to_zone": "HAND", "amount": 2, "preserve_order": True})
    assert tp2.from_zone == "DECK"
    assert tp2.to_zone == "HAND"
    assert tp2.amount == 2
    assert tp2.preserve_order is True


def test_modifier_params():
    mp = ModifierParams.model_validate({"target": "power", "delta": 2000, "temporary": True})
    assert mp.target == "power"
    assert mp.delta == 2000
    assert mp.temporary is True
