import pytest
from dm_toolkit.gui.editor.models import CommandModel, FilterSpec


def test_mutate_filterspec_ingest_and_serialize():
    data = {
        'type': 'MUTATE',
        'mutation_kind': 'POWER_MOD',
        'amount': 2,
        'filter': {'zones': ['BATTLE_ZONE'], 'min_cost': 3, 'is_tapped': 0}
    }

    cmd = CommandModel.model_validate(data)
    params = cmd.params
    # After ingest, params.filter should be a FilterSpec instance
    assert hasattr(params, 'filter')
    f = params.filter
    assert isinstance(f, FilterSpec)
    assert f.zones == ['BATTLE_ZONE']
    # Numeric flag normalized to bool
    assert f.is_tapped is False

    dumped = cmd.model_dump()
    assert dumped['type'] == 'MUTATE'
    # serialized filter should be dict with same keys
    assert isinstance(dumped.get('filter'), dict)
    assert dumped['filter'].get('min_cost') == 3
    assert dumped['filter'].get('is_tapped') is False
