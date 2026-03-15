from dm_toolkit.gui.editor.models import CommandModel


def test_mutate_params_ingest_and_serialize():
    data = {
        'type': 'MUTATE',
        'mutation_kind': 'POWER_MOD',
        'amount': 500,
        'target': 'TARGET_CREATURE'
    }

    # ingest via model validator
    cm = CommandModel.model_validate(data)
    assert cm.type == 'MUTATE'
    # params should be typed MutateParams or dict containing keys
    params = cm.params
    # model may be Pydantic model; check for attributes
    if hasattr(params, 'mutation_kind'):
        assert params.mutation_kind == 'POWER_MOD'
        assert params.amount == 500
        assert params.target == 'TARGET_CREATURE'
    else:
        # fallback: plain dict
        assert params.get('mutation_kind') == 'POWER_MOD'

    # serialization should include flattened fields
    out = cm.model_dump()
    assert out.get('type') == 'MUTATE'
    # amount and mutation_kind should be present at top-level due to serialize_model behavior
    assert out.get('mutation_kind') == 'POWER_MOD' or out.get('params', {}).get('mutation_kind') == 'POWER_MOD'
