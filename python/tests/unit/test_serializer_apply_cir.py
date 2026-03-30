import pytest
from dm_toolkit.gui.editor.models.serializer import ModelSerializer
from dm_toolkit.gui.editor.models import CardModel


def test_apply_cached_cir_to_card_attaches_attribute():
    ser = ModelSerializer()
    # Prepare fake cache entry
    card_id = 555
    ser._cir_cache[str(card_id)] = [{'kind': 'COMMAND', 'type': 'TEST'}]

    # Create a minimal CardModel with id
    card = CardModel.construct(id=card_id, name='Test', effects=[], static_abilities=[], reaction_abilities=[])

    # Apply
    ser._apply_cached_cir_to_card(card)

    assert hasattr(card, '_cir')
    assert isinstance(card._cir, list)
    assert card._cir[0]['type'] == 'TEST'
