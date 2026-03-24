from dm_toolkit.gui.editor.forms.unified_action_form import UnifiedActionForm


class SimpleItem:
    def __init__(self, cir=None):
        self._data = {'ROLE_CIR': cir} if cir is not None else {}

    def data(self, role):
        return self._data.get(role)


def test_unified_form_shows_cir_label_and_enable_button():
    form = UnifiedActionForm()
    # create a simple command dict with minimal fields
    data = {'type': 'DRAW'}
    item = SimpleItem(cir=[{'kind': 'COMMAND', 'type': 'TEST'}])

    # call loader
    # Prevent combo selection helpers from depending on full QComboBox API in test stubs
    form.set_combo_by_data = lambda combo, v: None
    form.populate_combo = lambda *a, **k: None

    form._load_ui_from_data(data, item)

    assert form.cir_label.isVisible()
    assert form.apply_cir_btn.isEnabled()
 