import pytest

from dm_toolkit.gui.editor.forms.unified_action_form import UnifiedActionForm


class FakeItem:
    def data(self, key):
        if key == 'ROLE_CIR':
            return [{'type': 'TEST_CMD', 'payload': {'a': 1, 'b': {'x': 2}, 'options': [{'label': 'L0'}, {'label': 'L1'}]}}]
        return None


def test_apply_selected_partial_emit(monkeypatch):
    form = UnifiedActionForm()

    emitted = []

    def fake_emit(name, payload):
        emitted.append((name, payload))

    form.structure_update_requested.emit = fake_emit

    # stub combo helpers like other tests
    form.set_combo_by_data = lambda combo, value: None
    form.populate_combo = lambda combo, items, data_func=None, display_func=None: None

    # load with CIR
    form._load_ui_from_data({'type': 'TEST_CMD', 'params': {}}, FakeItem())

    # simulate diff lines and select one
    # expected flattened lines: 'a', 'b.x', 'options[1].label'
    form.diff_tree_widget._lines = ['a', 'b.x', 'options[1].label']
    form.diff_tree_widget.select_lines(['options[1].label', 'b.x'])

    # call apply selected
    form.on_apply_selected()

    assert emitted, "No events emitted on partial apply"
    name, payload = emitted[-1]
    assert name == 'APPLY_CIR_PARTIAL'
    assert 'selected' in payload and isinstance(payload['selected'], list)
    assert set(payload['selected']) == set(['options[1].label', 'b.x'])
    assert 'applied' in payload
