from dm_toolkit.gui.editor.models import CommandModel


def test_apply_modifier_params_ingest_and_serialize():
    data = {
        'type': 'APPLY_MODIFIER',
        'modifier_type': 'COST_MOD',
        'value': 2,
        'scope': 'TARGET_CREATURE'
    }

    cm = CommandModel.model_validate(data)
    assert cm.type == 'APPLY_MODIFIER'
    params = cm.params
    if hasattr(params, 'modifier_type'):
        assert params.modifier_type == 'COST_MOD'
        assert params.value == 2
        assert params.scope == 'TARGET_CREATURE'
    else:
        assert params.get('modifier_type') == 'COST_MOD'

    out = cm.model_dump()
    assert out.get('type') == 'APPLY_MODIFIER'
    assert out.get('modifier_type') == 'COST_MOD' or out.get('params', {}).get('modifier_type') == 'COST_MOD'
