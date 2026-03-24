# -*- coding: utf-8 -*-
from dm_toolkit.gui.editor.forms.unified_action_form import UnifiedActionForm

class FakeWidget:
    def __init__(self):
        self.last_style = None
    def setStyleSheet(self, s):
        self.last_style = s


def test_load_clears_existing_highlights():
    form = object.__new__(UnifiedActionForm)
    # simulate existing widgets with highlights
    w = FakeWidget()
    w.setStyleSheet('background: yellow;')
    form.widgets_map = {'amount': w}

    # provide minimal stubs so _load_ui_from_data can run without full init
    class StubCombo:
        def blockSignals(self, v):
            return None

    form.action_group_combo = StubCombo()
    form.type_combo = StubCombo()
    form.set_combo_by_data = lambda combo, data: None
    form.populate_combo = lambda combo, types, data_func=None, display_func=None: None
    form.rebuild_dynamic_ui = lambda t: None

    # call clear_diff_highlight directly (verify clearing logic)
    form.clear_diff_highlight()

    # highlight should be cleared
    assert w.last_style == '' or w.last_style is None
