import pytest

from dm_toolkit.gui.editor.forms.unified_action_form import UnifiedActionForm


class FakeItem:
    def data(self, key):
        if key == 'ROLE_CIR':
            return [{'type': 'TEST_CMD', 'payload': {'k': 'v'}}]
        return None


def test_apply_and_reject_emit(monkeypatch):
    form = UnifiedActionForm()

    emitted = []

    def fake_emit(name, payload):
        emitted.append((name, payload))

    # replace the signal emitter with our fake
    form.structure_update_requested.emit = fake_emit

    # stub combo helpers to avoid GUI stub limitations in headless tests
    form.set_combo_by_data = lambda combo, value: None
    form.populate_combo = lambda combo, items, data_func=None, display_func=None: None

    # load item with CIR into form (provide required 'type')
    form._load_ui_from_data({'type': 'TEST_CMD', 'params': {}}, FakeItem())

    # buttons should be enabled when CIR present
    assert form.apply_cir_btn.isEnabled()
    # apply should emit APPLY_CIR with applied flag
    form.on_apply_cir()
    assert emitted, "No events emitted on apply"
    assert emitted[-1][0] == 'APPLY_CIR'
    assert isinstance(emitted[-1][1], dict)
    assert 'applied' in emitted[-1][1]

    emitted.clear()

    # reload and reject (provide required 'type')
    form._load_ui_from_data({'type': 'TEST_CMD', 'params': {}}, FakeItem())
    form.on_reject_cir()
    assert emitted, "No events emitted on reject"
    assert emitted[-1][0] == 'REJECT_CIR'
    assert isinstance(emitted[-1][1], dict)
    assert 'cir' in emitted[-1][1]
