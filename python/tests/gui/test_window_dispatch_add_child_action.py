# -*- coding: utf-8 -*-
from __future__ import annotations

from dm_toolkit.gui.editor.window import CardEditor
from dm_toolkit.gui.editor.consts import STRUCT_CMD_ADD_CHILD_ACTION


def test_structure_dispatch_add_child_action():
    win = object.__new__(CardEditor)

    calls = {}

    class FakeCardItem:
        def index(self):
            return 'card_idx'

    class FakeItem:
        def index(self):
            return 'item_idx'

    class FakeTree:
        def add_action_to_effect(self, idx):
            calls['add_action_to_effect'] = idx
        def add_action_to_option(self, idx):
            calls['add_action_to_option'] = idx
        def add_action_sibling(self, idx):
            calls['add_action_sibling'] = idx

    win.tree_widget = FakeTree()

    # EFFECT case
    handlers = win._structure_handlers(FakeCardItem(), FakeItem(), 'EFFECT', {})
    handler = handlers.get(STRUCT_CMD_ADD_CHILD_ACTION)
    assert handler is not None
    res = handler()

    assert res is True
    assert calls.get('add_action_to_effect') == 'item_idx'
