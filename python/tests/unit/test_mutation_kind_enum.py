from dm_toolkit.gui.editor.models import CommandModel
from dm_toolkit.gui.editor import models


def test_mutate_mutation_kind_enum_ingest_and_serialize():
    cmd = {
        'type': 'MUTATE',
        'mutation_kind': 'ADD_KEYWORD',
        'amount': 1,
        'filter': {'zones': ['BATTLE_ZONE'], 'is_tapped': True}
    }

    cm = CommandModel.model_validate(cmd)
    # After ingest, params should be a MutateParams model
    assert isinstance(cm.params, models.MutateParams)
    # mutation_kind should be coerced to the Enum when possible
    mk = cm.params.mutation_kind
    assert mk is not None
    # Accept either Enum or str, but prefer enum type
    assert (hasattr(mk, 'value') and str(mk) == 'ADD_KEYWORD') or mk == 'ADD_KEYWORD'

    # Serialize back to dict and ensure legacy string is present
    out = cm.model_dump()
    assert out.get('mutation_kind') == 'ADD_KEYWORD'
    # filter should be normalized to legacy dict form under 'filter'
    assert isinstance(out.get('filter'), dict)
    assert out['filter'].get('zones') == ['BATTLE_ZONE']
 