import os
import importlib
import sys
import pytest

# Skip if PyQt6 is not available
try:
    from PyQt6.QtWidgets import QApplication
except Exception:
    pytest.skip("PyQt6 not available, skipping GUI preview test", allow_module_level=True)


def test_gui_preview_label_updates():
    # Ensure headless fallback is not forced
    os.environ.pop('DM_EDITOR_HEADLESS', None)

    # 再発防止: 先行テストで headless 実装が import 済みだと、環境変数を戻しても
    # fallback クラスが再利用されるため、GUI 実装を明示的に再ロードする。
    mod_name = 'dm_toolkit.gui.editor.forms.parts.cost_reduction_editor'
    if mod_name in sys.modules:
        del sys.modules[mod_name]
    CostReductionEditor = importlib.import_module(mod_name).CostReductionEditor

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
