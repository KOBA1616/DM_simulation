import json
import os
from PyQt6.QtGui import QStandardItemModel
from dm_toolkit.gui.editor.data_manager import CardDataManager


def make_simple_card():
    return {
        "id": "1000",
        "name": "Test Card",
        "effects": [
            {
                "trigger": "ON_PLAY",
                "actions": [
                    {"type": "DRAW_CARD", "value1": 2}
                ]
            }
        ]
    }


def test_load_lift_converts_actions(tmp_path, qtbot):
    # Setup model and manager
    model = QStandardItemModel()
    mgr = CardDataManager(model)

    cards = [make_simple_card()]

    # Run load_data
    mgr.load_data(cards)

    # Reconstruct full data
    out = mgr.get_full_data()
    assert len(out) == 1
    card = out[0]

    # Ensure effects exist and have 'commands' not 'actions'
    effects_key = 'triggers' if 'triggers' in card else 'effects'
    effects = card.get(effects_key, [])
    assert len(effects) == 1
    eff = effects[0]
    assert 'actions' not in eff
    assert 'commands' in eff
    assert isinstance(eff['commands'], list)
    # The command should carry DRAW semantics (transition or draw representation)
    cmd = eff['commands'][0]
    assert isinstance(cmd, dict)
    assert cmd.get('type') is not None
