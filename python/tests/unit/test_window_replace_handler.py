# -*- coding: utf-8 -*-
from dm_toolkit.gui.editor.window import CardEditor
from dm_toolkit.gui.editor.consts import STRUCT_CMD_REPLACE_WITH_COMMAND


def test_replace_with_command_handler():
    win = object.__new__(CardEditor)

    calls = {}

    class FakeIdx:
        def isValid(self):
            return True

    class FakeTargetItem:
        def index(self):
            return FakeIdx()

    class FakeTree:
        def replace_item_with_command(self, idx, data):
            calls['replaced'] = (idx, data)
        def currentIndex(self):
            return FakeIdx()

    class FakeInspector:
        def set_selection(self, idx):
            calls['set_selection'] = idx

    # Attach fakes
    win.tree_widget = FakeTree()
    win.inspector = FakeInspector()
    # Create simple item with index method
    class Item:
        def index(self):
            return FakeIdx()
    item = Item()

    handlers = win._structure_handlers(None, item, 'COMMAND', {'target_item': FakeTargetItem(), 'new_data': {'type': 'FOO'}})
    handler = handlers.get(STRUCT_CMD_REPLACE_WITH_COMMAND)
    assert handler is not None
    res = handler()
    assert res is True
    assert 'replaced' in calls
    assert calls['replaced'][1]['type'] == 'FOO'
