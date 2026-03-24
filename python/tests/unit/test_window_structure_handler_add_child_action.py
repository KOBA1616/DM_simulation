import types

from dm_toolkit.gui.editor.window import CardEditor
from dm_toolkit.gui.editor.consts import STRUCT_CMD_ADD_CHILD_ACTION


class FakeTreeWidget:
    def __init__(self):
        self.called = []

    def add_action_to_effect(self, idx):
        self.called.append(('add_action_to_effect', idx))

    def add_action_to_option(self, idx):
        self.called.append(('add_action_to_option', idx))

    def add_action_sibling(self, idx):
        self.called.append(('add_action_sibling', idx))

    def expand(self, idx):
        self.called.append(('expand', idx))


class DummyItem:
    def __init__(self, ident, data_type):
        self._id = ident
        self._data_type = data_type

    def index(self):
        return self._id

    def data(self, role=None):
        # emulate QStandardItem.data for item_type
        return self._data_type


def test_add_child_action_calls_effect_path():
    # Create a lightweight CardEditor-like object without running __init__
    editor = CardEditor.__new__(CardEditor)
    fake_tree = FakeTreeWidget()
    editor.tree_widget = fake_tree

    # create dummy card_item and item
    card_item = DummyItem('card_idx', 'CARD')
    item = DummyItem('eff_idx', 'EFFECT')

    # Call the structure handlers factory and invoke the add-child-action handler
    handlers = editor._structure_handlers(card_item, item, 'EFFECT', payload={})
    handler = handlers.get(STRUCT_CMD_ADD_CHILD_ACTION)
    assert handler is not None
    result = handler()
    assert result is True
    # ensure the expected tree method was called
    assert ('add_action_to_effect', 'eff_idx') in fake_tree.called
