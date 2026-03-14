# -*- coding: utf-8 -*-
from __future__ import annotations

import tempfile
from PyQt6.QtWidgets import QApplication
from dm_toolkit.gui.editor.window import CardEditor
from dm_toolkit.gui.editor.consts import STRUCT_CMD_ADD_SPELL_SIDE


def test_structure_dispatch_add_spell_side():
    # Create a bare CardEditor instance without running __init__
    win = object.__new__(CardEditor)

    calls = {}

    class FakeCardItem:
        def index(self):
            return 'card_idx'

    class FakeItem:
        def index(self):
            return 'item_idx'

    class FakeTree:
        def __init__(self):
            self.data_manager = type('DM', (), {'add_option_slots': lambda *a, **k: None})()
        def add_spell_side(self, idx):
            calls['add_spell_side'] = idx
        def expand(self, idx):
            calls['expand'] = idx

    win.tree_widget = FakeTree()

    # Build handlers and invoke the add_spell_side handler
    handlers = win._structure_handlers(FakeCardItem(), FakeItem(), 'EFFECT', {})
    handler = handlers.get(STRUCT_CMD_ADD_SPELL_SIDE)
    assert handler is not None
    res = handler()

    assert res is True
    assert calls.get('add_spell_side') == 'card_idx'
    assert calls.get('expand') == 'card_idx'
