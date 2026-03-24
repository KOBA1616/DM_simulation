from dm_toolkit.gui.editor.window import CardEditor


class FakeTreeWidget:
    def __init__(self):
        self.called = []

    def add_keywords(self, idx):
        self.called.append(('add_keywords', idx))

    def add_trigger(self, idx):
        self.called.append(('add_trigger', idx))

    def add_static(self, idx):
        self.called.append(('add_static', idx))

    def add_reaction(self, idx):
        self.called.append(('add_reaction', idx))


class DummyItem:
    def __init__(self, ident):
        self._id = ident

    def index(self):
        return self._id


def test_handle_add_child_effect_keywords():
    editor = CardEditor.__new__(CardEditor)
    fake_tree = FakeTreeWidget()
    editor.tree_widget = fake_tree

    item = DummyItem('item_idx')
    payload = {'type': 'KEYWORDS'}

    result = editor._handle_add_child_effect(item, payload)
    assert result is True
    assert ('add_keywords', 'item_idx') in fake_tree.called


def test_handle_add_child_effect_triggered():
    editor = CardEditor.__new__(CardEditor)
    fake_tree = FakeTreeWidget()
    editor.tree_widget = fake_tree

    item = DummyItem('item_idx')
    payload = {'type': 'TRIGGERED'}

    result = editor._handle_add_child_effect(item, payload)
    assert result is True
    assert ('add_trigger', 'item_idx') in fake_tree.called


def test_handle_add_child_effect_static():
    editor = CardEditor.__new__(CardEditor)
    fake_tree = FakeTreeWidget()
    editor.tree_widget = fake_tree

    item = DummyItem('item_idx')
    payload = {'type': 'STATIC'}

    result = editor._handle_add_child_effect(item, payload)
    assert result is True
    assert ('add_static', 'item_idx') in fake_tree.called


def test_handle_add_child_effect_reaction():
    editor = CardEditor.__new__(CardEditor)
    fake_tree = FakeTreeWidget()
    editor.tree_widget = fake_tree

    item = DummyItem('item_idx')
    payload = {'type': 'REACTION'}

    result = editor._handle_add_child_effect(item, payload)
    assert result is True
    assert ('add_reaction', 'item_idx') in fake_tree.called


def test_handle_add_child_effect_unknown():
    editor = CardEditor.__new__(CardEditor)
    fake_tree = FakeTreeWidget()
    editor.tree_widget = fake_tree

    item = DummyItem('item_idx')
    payload = {'type': 'UNKNOWN'}

    result = editor._handle_add_child_effect(item, payload)
    assert result is False
