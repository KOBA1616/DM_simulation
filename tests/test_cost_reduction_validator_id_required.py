from dm_toolkit.gui.editor import validators_shared as v


def test_missing_id_is_error():
    data = [
        {"type": "PASSIVE"},
        {"type": "ACTIVE_PAYMENT", "id": "a1", "max_units": 1, "reduction_per_unit": 1},
    ]
    errs = v.validate_cost_reductions(data)
    assert any("id is required" in e for e in errs)
