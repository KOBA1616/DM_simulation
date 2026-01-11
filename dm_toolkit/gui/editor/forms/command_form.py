# -*- coding: utf-8 -*-
from PyQt6.QtWidgets import QWidget, QFormLayout, QLabel, QPushButton
from PyQt6.QtCore import Qt, pyqtSignal
from dm_toolkit.gui.localization import tr
from dm_toolkit.gui.editor.forms.unified_action_form import UnifiedActionForm
from dm_toolkit.consts import COMMAND_TYPES
from dm_toolkit.gui.editor.consts import STRUCT_CMD_GENERATE_BRANCHES

class CommandEditForm(UnifiedActionForm):
    """
    Refactored CommandEditForm that relies on UnifiedActionForm schema logic.
    Retains specific FLOW logic (Generate Branches) and legacy warning.
    """
    structure_update_requested = pyqtSignal(str, dict)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()

    def setup_ui(self):
        # Base UI is already set up by UnifiedActionForm.__init__ -> setup_base_ui
        # We just need to add the special Flow button and warning label.

        # Access the main layout created by base
        layout = self.main_layout

        # Warning Label
        self.warning_label = QLabel(tr("Warning: Imperfect Conversion"))
        self.warning_label.setStyleSheet("color: red; font-weight: bold;")
        self.warning_label.setVisible(False)
        layout.insertRow(0, self.warning_label)

        # Branch generation button (visible only for FLOW commands)
        self.gen_branch_btn = QPushButton(tr("Generate Branches"))
        self.gen_branch_btn.setVisible(False)
        layout.addRow(self.gen_branch_btn)
        self.gen_branch_btn.clicked.connect(self.request_generate_branches)

    def on_type_changed(self):
        super().on_type_changed()

        cmd_type = self.type_combo.currentData()

        # Logic specific to CommandEditForm

        # Legacy Warning
        if cmd_type and cmd_type not in COMMAND_TYPES:
             self.warning_label.setText(
                  tr("This type '{cmd_type}' is only supported by the Legacy Action format.")
                  .format(cmd_type=cmd_type)
             )
             self.warning_label.setVisible(True)
        else:
             self.warning_label.setVisible(False)

        # Flow Button
        self.gen_branch_btn.setVisible(cmd_type == "FLOW")

    def _populate_ui(self, item):
        # We override _populate_ui to add the legacy warning check *after* standard load
        # But standard load is _load_ui_from_data called by BaseEditForm.load_data
        # BaseEditForm calls _load_ui_from_data(data, item)
        # So we should override _load_ui_from_data or hook into it.
        # UnifiedActionForm implements _load_ui_from_data.
        pass

    def _load_ui_from_data(self, data, item):
        super()._load_ui_from_data(data, item)

        # Add legacy warning check
        legacy_warning = data.get('legacy_warning', False)
        if legacy_warning:
            orig = data.get('legacy_original_type', 'Unknown')
            self.warning_label.setText(
                tr("Warning: Imperfect Conversion from {orig}").format(orig=orig)
            )
            self.warning_label.setVisible(True)
        else:
            # Re-check based on current type if not explicit legacy flag
            cmd_type = self.type_combo.currentData()
            if cmd_type and cmd_type not in COMMAND_TYPES:
                self.warning_label.setVisible(True)
            else:
                self.warning_label.setVisible(False)

        # Flow button visibility
        cmd_type = self.type_combo.currentData()
        self.gen_branch_btn.setVisible(cmd_type == "FLOW")

    def request_generate_branches(self):
        self.structure_update_requested.emit(STRUCT_CMD_GENERATE_BRANCHES, {})
