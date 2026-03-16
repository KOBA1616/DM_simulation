import os
import pytest

# Skip if PyQt6 is not available
try:
    from PyQt6.QtWidgets import QApplication
    from dm_toolkit.gui.editor.forms.parts.cost_reduction_editor import CostReductionEditor
except Exception:
    pytest.skip("PyQt6 not available, skipping GUI preview test", allow_module_level=True)


def test_gui_preview_label_updates():
    # Ensure headless fallback is not forced
    os.environ.pop('DM_EDITOR_HEADLESS', None)
    app = QApplication.instance() or QApplication([])
    ed = CostReductionEditor()
    ed.set_value([
        {"type": "ACTIVE_PAYMENT", "amount": 4, "unit_cost": 3, "min_mana_cost": 2, "max_units": 10}
    ])
    # select first entry
    ed.set_selected_index(0)
    # ensure preview label shows expected value (4 * 3 = 12)
    txt = ed.preview_label.text()
    assert "12" in txt

    # change via spinbox by calling update_selected_fields
    ed.update_selected_fields(amount=2)
    assert "6" in ed.preview_label.text()
