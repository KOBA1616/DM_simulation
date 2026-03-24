from dm_toolkit.gui.editor import validators_shared as v


def test_valid_passive_and_active():
    data = [
        {"type": "PASSIVE", "id": "p1", "min_mana_cost": 0},
        {"type": "ACTIVE_PAYMENT", "id": "a1", "max_units": 3, "reduction_per_unit": 2},
    ]
    errs = v.validate_cost_reductions(data)
    assert errs == []


def test_invalid_active_missing_fields():
    data = [{"type": "ACTIVE_PAYMENT", "id": "a2"}]
    errs = v.validate_cost_reductions(data)
    assert any("ACTIVE_PAYMENT requires" in e for e in errs)


def test_duplicate_id_and_bad_values():
    data = [
        {"type": "PASSIVE", "id": "dup"},
        {"type": "PASSIVE", "id": "dup"},
        {"type": "PASSIVE", "id": "ok", "min_mana_cost": -1},
        {"type": "ACTIVE_PAYMENT", "id": "bad", "max_units": 0, "reduction_per_unit": -2},
    ]
    errs = v.validate_cost_reductions(data)
    # Expect duplicate id and invalid numeric errors
    assert any("duplicate id" in e for e in errs)
    assert any("min_mana_cost must be" in e for e in errs)
    assert any("max_units must be" in e or "reduction_per_unit must be" in e for e in errs)
