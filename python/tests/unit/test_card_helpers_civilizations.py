from dm_toolkit.gui.utils.card_helpers import get_card_civilizations


def test_get_card_civilizations_from_dict_singular_multicolor_string() -> None:
    card = {'civilization': 'WATER/FIRE'}
    civs = get_card_civilizations(card)
    assert civs == ['WATER', 'FIRE']


def test_get_card_civilizations_from_dict_singular_single() -> None:
    card = {'civilization': 'NATURE'}
    civs = get_card_civilizations(card)
    assert civs == ['NATURE']
