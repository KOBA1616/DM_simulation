import pytest
from dm_toolkit.gui.editor.models.serializer import ModelSerializer
from dm_toolkit.gui.editor.models import CardModel


def test_serialize_card_model_includes_cir():
    ser = ModelSerializer()
    card = CardModel.construct(id=42, name='CIR Card', effects=[], static_abilities=[], reaction_abilities=[])
    # attach fake CIR
    card._cir = [{'kind': 'COMMAND', 'type': 'FOO'}]

    out = ser._serialize_card_model(card)

    assert '_canonical_ir' in out
    assert isinstance(out['_canonical_ir'], list)
    assert out['_canonical_ir'][0]['type'] == 'FOO'
