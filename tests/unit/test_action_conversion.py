from dm_toolkit.gui.editor.action_converter import ActionConverter
from dm_toolkit.gui.editor.data_manager import CardDataManager
from PyQt6.QtGui import QStandardItemModel


def test_convert_move_card_return_to_hand():
    act = {
        "type": "MOVE_CARD",
        "destination_zone": "HAND",
        "source_zone": "BATTLE_ZONE",
    }

    cmd = ActionConverter.convert(act)
    assert isinstance(cmd, dict)
    assert cmd.get('type') == 'RETURN_TO_HAND'
    assert 'uid' in cmd


def test_normalize_card_for_engine_basic():
    model = QStandardItemModel()
    mgr = CardDataManager(model)

    # Create a minimal card with an effect containing the converted command
    act = {"type": "MOVE_CARD", "destination_zone": "HAND", "source_zone": "BATTLE_ZONE"}
    cmd = ActionConverter.convert(act)

    card = {
        "id": 9999,
        "name": "Test Card",
        "effects": [
            {"trigger": "ON_PLAY", "condition": {"type": "NONE"}, "commands": [cmd]}
        ]
    }

    warnings = mgr._normalize_card_for_engine(card)
    assert isinstance(warnings, list)
    # For this simple converted command, we expect no warnings (conversion should be valid)
    assert warnings == []
    # Check uid preserved on command
    assert 'uid' in card['effects'][0]['commands'][0]
