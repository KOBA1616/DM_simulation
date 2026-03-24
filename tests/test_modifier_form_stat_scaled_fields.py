from PyQt6.QtWidgets import QApplication
import pytest

from dm_toolkit.gui.editor.forms.modifier_form import ModifierEditForm


def test_modifier_form_persists_stat_scaled_fields():
    # Ensure QApplication exists (create if necessary)
    app = QApplication.instance()
    created_app = False
    if app is None:
        app = QApplication([])
        created_app = True

    form = ModifierEditForm()

    data = {
        'type': 'COST_MODIFIER',
        'value_mode': 'STAT_SCALED',
        'stat_key': 'MY_MANA_COUNT',
        'per_value': 2,
        'min_stat': 1,
        'max_reduction': 3,
    }

    # Load into form and then save back out
    form._load_ui_from_data(data, None)

    out = {'type': 'COST_MODIFIER'}
    form._save_ui_to_data(out)

    assert out.get('value_mode') == 'STAT_SCALED'
    assert out.get('stat_key') == 'MY_MANA_COUNT'
    assert int(out.get('per_value')) == 2
    assert int(out.get('min_stat')) == 1
    assert out.get('max_reduction') == 3

    # Clean up created QApplication to avoid side-effects
    if created_app:
        app.quit()
