from dm_toolkit.gui.editor.models import CommandModel, LockSpellParams


def test_commandmodel_ingests_lock_spell_params():
    data = {
        'type': 'LOCK_SPELL',
        'params': {
            'target_group': 'PLAYER_OPPONENT',
            'duration': 'TURN',
            'filter': {'cost': {'max': 3}}
        }
    }

    cmd = CommandModel.model_validate(data)

    assert isinstance(cmd.params, LockSpellParams)
    assert cmd.params.target_group == 'PLAYER_OPPONENT'
    assert cmd.params.duration == 'TURN'
    assert cmd.params.filter == {'cost': {'max': 3}}


def test_serialize_flattens_lock_spell_params():
    data = {'type': 'LOCK_SPELL', 'params': {'duration': 'ROUND', 'target_group': 'PLAYER_SELF'}}
    cmd = CommandModel.model_validate(data)
    out = cmd.model_dump()
    assert out['duration'] == 'ROUND'
    assert out['target_group'] == 'PLAYER_SELF'
