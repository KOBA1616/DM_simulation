from dm_toolkit.gui.editor.models import CommandModel


def test_cast_spell_params_ingest_and_serialize():
    data = {
        'type': 'CAST_SPELL',
        'spell_id': 42,
        'target': 'ENEMY_CREATURE',
        'cost': 3,
        'use_mana_from': 'PLAYER_MANA',
        'target_group': 'PLAYER_SELF',
        'target_filter': {'zones': ['HAND'], 'types': ['SPELL'], 'max_cost': 3},
        'optional': True,
    }

    cm = CommandModel.model_validate(data)
    assert cm.type == 'CAST_SPELL'
    params = cm.params
    if hasattr(params, 'spell_id'):
        assert params.spell_id == 42
        assert params.target == 'ENEMY_CREATURE'
        assert params.cost == 3
        assert params.use_mana_from == 'PLAYER_MANA'
        assert params.target_group == 'PLAYER_SELF'
        assert params.target_filter == {'zones': ['HAND'], 'types': ['SPELL'], 'max_cost': 3}
        assert params.optional is True
    else:
        assert params.get('spell_id') == 42

    out = cm.model_dump()
    assert out.get('type') == 'CAST_SPELL'
    assert out.get('spell_id') == 42 or out.get('params', {}).get('spell_id') == 42
    assert out.get('target_group') == 'PLAYER_SELF'
    assert out.get('target_filter') == {'zones': ['HAND'], 'types': ['SPELL'], 'max_cost': 3}
    assert out.get('optional') is True
