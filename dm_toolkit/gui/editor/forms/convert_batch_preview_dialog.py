# -*- coding: utf-8 -*-
from PyQt6.QtWidgets import QDialog, QVBoxLayout, QLabel, QListWidget, QListWidgetItem, QDialogButtonBox
from PyQt6.QtGui import QIcon
from PyQt6.QtCore import Qt
from dm_toolkit.gui.localization import tr

class ConvertBatchPreviewDialog(QDialog):
    def __init__(self, parent, preview_items):
        super().__init__(parent)
        self.setWindowTitle(tr("Preview Conversions"))
        self.resize(640, 420)

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel(tr("The following actions will be converted to Commands. Review warnings before applying.")))

        self.list_widget = QListWidget(self)
        for p in preview_items:
            text = f"{p['label']} — {p['path']}"
            item = QListWidgetItem(text)
            if p.get('warning'):
                # Use a simple warning prefix; icon may not be available cross-platform
                item.setText("⚠️ " + item.text())
                item.setData(Qt.ItemDataRole.UserRole, True)
            self.list_widget.addItem(item)

        layout.addWidget(self.list_widget)

        btns = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        layout.addWidget(btns)

        # Expand height to show enough items
        self.setLayout(layout)
*** End Patch