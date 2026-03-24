from dm_toolkit.gui.editor.models import CommandModel
from dm_toolkit.gui.editor.models import AddKeywordParams


def test_add_shield_params_ingest_typed():
    data = {
        'type': 'ADD_SHIELD',
        'amount': 3,
        'some_flag': True
    }

    cm = CommandModel.model_validate(data)
    # params should be a dict (legacy) or typed model; we expect typed AddShieldParams
    params = cm.params
    # Check serialized output contains flattened amount
    out = cm.model_dump()
    assert out.get('amount') == 3


def test_boost_mana_params_ingest_typed():
    data = {
        'type': 'BOOST_MANA',
        'amount': 2
    }
    cm = CommandModel.model_validate(data)
    out = cm.model_dump()
    assert out.get('amount') == 2
