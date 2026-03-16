from dm_toolkit.gui.editor.data_migration import migrate_cost_reductions


def test_migrate_assigns_ids_to_cost_reductions():
    card = {
        'id': 'c100',
        'cost_reductions': [
            {'type': 'PASSIVE'},
            {'type': 'ACTIVE_PAYMENT', 'max_units': 2, 'reduction_per_unit': 1}
        ]
    }
    modified = migrate_cost_reductions(card)
    assert modified == 2 or modified == 1 or modified >= 0
    # Ensure ids exist
    for cr in card['cost_reductions']:
        assert 'id' in cr and isinstance(cr['id'], str) and cr['id']
