# -*- coding: utf-8 -*-
from PyQt6.QtWidgets import QLabel, QVBoxLayout, QFormLayout, QLineEdit
from PyQt6.QtCore import Qt
from dm_toolkit.gui.editor.forms.base_form import BaseEditForm
from dm_toolkit.gui.i18n import tr

class OptionForm(BaseEditForm):
    """
    Editable form for OPTION container nodes.

    OPTIONs are structural elements (containers for conditional branches or choices).
    Supports editing the label text shown to the player when a choice is presented.

    再発防止: OPTION の label フィールドは SELECT_OPTION コマンド実行時に
    プレイヤーに表示するテキストとして使用される。
    label が未設定の場合は item.text() (例: "Option 1") がフォールバックとして使われる。
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.label = None
        self.label_edit = None
        try:
            self.setup_ui()
        except Exception:
            pass

    def setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        self.label = QLabel(tr("Option"))
        self.label.setAlignment(Qt.AlignmentFlag.AlignTop)
        layout.addWidget(self.label)

        form_layout = QFormLayout()
        form_layout.setContentsMargins(0, 0, 0, 0)
        layout.addLayout(form_layout)

        # 選択肢テキスト (プレイヤーに表示するラベル)
        self.label_edit = QLineEdit()
        self.label_edit.setPlaceholderText(tr("例: はい / いいえ / 効果A"))
        self.label_edit.setToolTip(tr(
            "SELECT_OPTION コマンドでプレイヤーに表示される選択肢テキスト。"
            "未設定の場合は「Option N」がデフォルトとして使われます。"
        ))
        self.label_edit.textChanged.connect(self.update_data)
        form_layout.addRow(tr("選択肢テキスト:"), self.label_edit)

        info_label = QLabel(tr(
            "この選択肢が選ばれた場合に実行するコマンドを\n"
            "子ノードとして追加してください。"
        ))
        info_label.setWordWrap(True)
        info_label.setStyleSheet("color: #888; font-size: 10px;")
        layout.addWidget(info_label)

        layout.addStretch()

        self.register_widget(self.label_edit, 'label')

    def set_data(self, item):
        super().set_data(item)
        if self.label is not None:
            self.label.setText(item.text())

    def _load_ui_from_data(self, data, item):
        """Load label from data or fall back to item.text()."""
        if self.label_edit is None:
            return
        label_text = data.get('label', '') or item.text()
        self.label_edit.setText(label_text)

    def _save_ui_to_data(self, data):
        """Save label to data dict and update item display text."""
        if self.label_edit is None:
            return
        text = self.label_edit.text().strip()
        data['label'] = text

    def _get_display_text(self, data):
        """Node display text: use label if set, otherwise fall back to existing text."""
        label = data.get('label', '').strip()
        if label:
            return label
        # Keep existing text (e.g. "Option 1") if label is empty
        if self.current_item is not None:
            existing = self.current_item.text()
            if existing:
                return existing
        return tr("Option")

