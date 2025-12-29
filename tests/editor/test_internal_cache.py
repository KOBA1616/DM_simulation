from PyQt6.QtGui import QStandardItemModel
from PyQt6.QtCore import Qt
from dm_toolkit.gui.editor.data_manager import CardDataManager


def test_add_option_slots_updates_internal_cache():
    model = QStandardItemModel()
    dm = CardDataManager(model)

    # create a card
    card_item = dm.add_new_card()

    # create an effect and action under card
    eff = dm._create_effect_item({"trigger": "ON_PLAY", "commands": []})
    card_item.appendRow(eff)

    act = dm._create_action_item({"type": "DRAW_CARD", "value1": 1})
    eff.appendRow(act)

    # ensure action was internalized
    payload = act.data(Qt.ItemDataRole.UserRole + 2)
    assert isinstance(payload, dict)
    assert payload.get('uid') is not None
    assert dm.get_internal_by_uid(payload.get('uid')) is not None

    # add option slots
    dm.add_option_slots(act, 2)

    # verify option children exist and are internalized
    for i in range(act.rowCount()):
        child = act.child(i)
        if child.data(Qt.ItemDataRole.UserRole + 1) == 'OPTION':
            d = child.data(Qt.ItemDataRole.UserRole + 2)
            assert isinstance(d, dict)
            assert d.get('uid') in dm._internal_cache


def test_add_command_branches_updates_internal_cache():
    model = QStandardItemModel()
    dm = CardDataManager(model)

    # create a command item
    cmd = dm.create_command_item({"type": "NONE"})

    # add branches
    dm.add_command_branches(cmd)

    # find branch children and confirm internalized
    for i in range(cmd.rowCount()):
        child = cmd.child(i)
        role = child.data(Qt.ItemDataRole.UserRole + 1)
        if role in ('CMD_BRANCH_TRUE', 'CMD_BRANCH_FALSE'):
            d = child.data(Qt.ItemDataRole.UserRole + 2)
            assert isinstance(d, dict)
            assert d.get('uid') in dm._internal_cache
