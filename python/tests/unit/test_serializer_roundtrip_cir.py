import pytest
from dm_toolkit.gui.editor.models.serializer import ModelSerializer
from dm_toolkit.gui.editor.models import CardModel


def test_cir_round_trip_preserved():
    ser = ModelSerializer()
    cir = [{'kind': 'COMMAND', 'type': 'ROUND'}]
    ser._cir_cache['99'] = cir

    card = CardModel.construct(id=99, name='RoundTrip', effects=[], static_abilities=[], reaction_abilities=[])

    ser._apply_cached_cir_to_card(card)
    out = ser._serialize_card_model(card)

    assert '_canonical_ir' in out
    assert out['_canonical_ir'] == cir
