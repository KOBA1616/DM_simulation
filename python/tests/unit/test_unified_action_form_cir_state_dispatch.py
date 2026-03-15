# -*- coding: utf-8 -*-
import inspect

from dm_toolkit.gui.editor.forms.unified_action_form import UnifiedActionForm


class _FakeButton:
    def __init__(self) -> None:
        self.enabled = True
        self.visible = True
        self.tooltip = ""
        self.text = ""

    def setEnabled(self, enabled: bool) -> None:
        self.enabled = enabled

    def isEnabled(self) -> bool:
        return self.enabled

    def setVisible(self, visible: bool) -> None:
        self.visible = visible

    def isVisible(self) -> bool:
        return self.visible

    def setToolTip(self, text: str) -> None:
        self.tooltip = text

    def setText(self, text: str) -> None:
        self.text = text


class _FakeDiffWidget:
    def __init__(self) -> None:
        self.visible = True

    def setVisible(self, visible: bool) -> None:
        self.visible = visible

    def isVisible(self) -> bool:
        return self.visible


def test_load_ui_delegates_cir_state_handling_to_helpers():
    src = inspect.getsource(UnifiedActionForm._load_ui_from_data)
    assert "self._extract_cir_entries(" in src
    assert "self._update_cir_ui_state(" in src
    assert "self._update_cir_diff_view(" in src


def test_update_cir_ui_state_disables_controls_without_cir():
    dummy = type("_Dummy", (), {})()
    dummy.cir_label = _FakeButton()
    dummy.apply_cir_btn = _FakeButton()
    dummy.reject_cir_btn = _FakeButton()
    dummy.apply_selected_btn = _FakeButton()
    dummy.diff_tree_widget = _FakeDiffWidget()

    UnifiedActionForm._update_cir_ui_state(dummy, None)

    assert not dummy.cir_label.isVisible()
    assert not dummy.apply_cir_btn.isEnabled()
    assert not dummy.reject_cir_btn.isEnabled()
    assert not dummy.apply_selected_btn.isEnabled()
    assert not dummy.diff_tree_widget.isVisible()
