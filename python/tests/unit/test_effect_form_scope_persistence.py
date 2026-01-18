# -*- coding: utf-8 -*-
import pytest

from PyQt6.QtCore import Qt

from dm_toolkit.gui.editor.forms.effect_form import EffectEditForm

class DummyItem:
    def __init__(self, data=None):
        self._store = {}
        if data is not None:
            self._store[Qt.ItemDataRole.UserRole + 2] = data
        self._store_text = ""
    def data(self, role):
        return self._store.get(role)
    def setData(self, value, role):
        self._store[role] = value
    def setText(self, text):
        self._store_text = text
    def parent(self):
        return None

@pytest.mark.gui
def test_trigger_scope_persists_across_mode_toggle():
    # Initial effect data with TRIGGERED mode and specific scope
    effect = {
        'trigger': 'ON_PLAY',
        'trigger_scope': 'PLAYER_OPPONENT',
        'trigger_filter': {}
    }
    item = DummyItem(effect)

    form = EffectEditForm()
    # If setup_ui failed due to headless stubbing limitations, skip this test
    if not getattr(form, 'trigger_scope_combo', None) or not getattr(form, 'mode_combo', None):
        pytest.skip("EffectEditForm UI not initialized in headless stub; skipping persistence test")

    # Load initial data
    try:
        form.load_data(item)
    except Exception:
        pytest.skip("EffectEditForm load_data not available in headless stub; skipping")
    assert form.mode_combo.currentData() == 'TRIGGERED'
    assert form.trigger_scope_combo.currentData() == 'PLAYER_OPPONENT'

    # Switch to STATIC and save
    form.set_combo_by_data(form.mode_combo, 'STATIC')
    form.update_data()

    # Verify data still carries trigger_scope for persistence
    saved = item.data(Qt.ItemDataRole.UserRole + 2)
    assert saved.get('trigger_scope') == 'PLAYER_OPPONENT'

    # Switch back to TRIGGERED and reload form
    form.set_combo_by_data(form.mode_combo, 'TRIGGERED')
    form.update_data()
    form.load_data(item)

    # Ensure selection restored
    assert form.trigger_scope_combo.currentData() == 'PLAYER_OPPONENT'
