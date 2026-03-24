import os
import pytest


def test_gui_bindings_update_selected_fields():
    # Skip if PyQt is not available in the environment
    try:
        import PyQt6  # type: ignore
    except Exception:
        pytest.skip("PyQt6 not available")

    # Ensure we use real Qt path (do not force headless)
    if 'DM_EDITOR_HEADLESS' in os.environ:
        del os.environ['DM_EDITOR_HEADLESS']

    # Reload module to ensure Qt-backed implementation is used (tests may have
    # previously imported the module under headless mode)
    import sys, importlib
    from PyQt6.QtWidgets import QApplication

    # 再発防止: QWidget を生成する GUI テストで QApplication が未初期化だと
    # プロセスクラッシュする環境があるため、必ず先にアプリを確保する。
    _app = QApplication.instance() or QApplication([])

    mod_name = 'dm_toolkit.gui.editor.forms.parts.cost_reduction_editor'
    if mod_name in sys.modules:
        del sys.modules[mod_name]
    CostReductionEditor = importlib.import_module(mod_name).CostReductionEditor

    w = CostReductionEditor()
    # Add an entry and ensure selection
    w._on_add()
    w.set_selected_index(0)
    # Simulate user interaction via spinboxes
    w.amount_spin.setValue(7)
    w.min_mana_spin.setValue(2)
    w.unit_cost_spin.setValue(3)
    w.max_units_spin.setValue(4)

    out = w.get_value()
    assert out[0]['amount'] == 7
    assert out[0]['min_mana_cost'] == 2
    assert out[0]['unit_cost'] == 3
    assert out[0]['max_units'] == 4
