from dm_toolkit.gui.editor import validators_shared as v


def test_generate_missing_ids_assigns_and_preserves():
    data = [
        {"type": "PASSIVE", "name": "p1"},
        {"type": "ACTIVE_PAYMENT", "id": "existing", "max_units": 2, "reduction_per_unit": 1},
        {"type": "PASSIVE"},
    ]
    v.generate_missing_ids(data)
    ids = [d.get("id") for d in data]
    assert all(isinstance(i, str) and i for i in ids)
    assert ids[1] == "existing"
    # uniqueness
    assert len(set(ids)) == len(ids)


def test_generate_missing_ids_handles_non_list():
    # Should not raise
    v.generate_missing_ids(None)
    v.generate_missing_ids("notalist")
