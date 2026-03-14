# -*- coding: utf-8 -*-
# 再発防止: PropertyInspector は「単一ホスト体験」設計。
# パンくずラベル (breadcrumb) でツリー位置を常時表示し、
# ブランチページには必ず「コマンド追加」ボタンを含めること。
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QStackedWidget, QLabel, QPushButton, QFrame
)
from PyQt6.QtCore import Qt, pyqtSignal
from dm_toolkit.gui.editor.forms.card_form import CardEditForm
from dm_toolkit.gui.editor.forms.effect_form import EffectEditForm
from dm_toolkit.gui.editor.forms.spell_side_form import SpellSideForm
from dm_toolkit.gui.editor.forms.reaction_form import ReactionEditForm
from dm_toolkit.gui.editor.forms.unified_action_form import UnifiedActionForm
from dm_toolkit.gui.editor.forms.keyword_form import KeywordEditForm
from dm_toolkit.gui.editor.forms.modifier_form import ModifierEditForm
from dm_toolkit.gui.editor.forms.option_form import OptionForm
from dm_toolkit.gui.i18n import tr
from dm_toolkit.gui.editor.forms.signal_utils import safe_connect


class CmdBranchPage(QWidget):
    """ブランチ選択時に表示されるページ。コマンド追加ボタン付き。"""
    structure_update_requested = pyqtSignal(str, dict)

    def __init__(self, label_key: str = "Branch selected. Add Commands to this branch.", parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        layout.setContentsMargins(8, 12, 8, 8)
        layout.setSpacing(8)

        self._info_label = QLabel(tr(label_key))
        self._info_label.setWordWrap(True)
        self._info_label.setStyleSheet("color: #888; font-size: 11px;")
        layout.addWidget(self._info_label)

        btn = QPushButton(tr("Add Command"))
        btn.setFixedHeight(28)
        safe_connect(btn, 'clicked', lambda: self.structure_update_requested.emit("ADD_CHILD_ACTION", {}))
        layout.addWidget(btn)
        layout.addStretch()


class OptionEditPage(QWidget):
    structure_update_requested = pyqtSignal(str, dict)
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel(tr("Option selected. Add Commands to define behavior.")))
        btn = QPushButton(tr("Add Command"))
        safe_connect(btn, 'clicked', lambda: self.structure_update_requested.emit("ADD_CHILD_ACTION", {}))
        layout.addWidget(btn)
        layout.addStretch()


class PropertyInspector(QWidget):
    # Forward signal from forms
    structure_update_requested = pyqtSignal(str, dict)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()

    # ---- breadcrumb helper ----
    def _build_breadcrumb(self, index) -> str:
        """ツリーの祖先パスから パンくず文字列を生成する。"""
        parts: list[str] = []
        cur = index
        while cur.isValid():
            text = cur.data(Qt.ItemDataRole.DisplayRole) or ""
            if text:
                parts.append(text)
            cur = cur.parent()
        parts.reverse()
        return " > ".join(parts)

    def _update_breadcrumb(self, index) -> None:
        if index is None or not index.isValid():
            self.breadcrumb_label.setText("")
            self.breadcrumb_label.setVisible(False)
            return
        crumb = self._build_breadcrumb(index)
        self.breadcrumb_label.setText(crumb)
        self.breadcrumb_label.setVisible(True)

    # ---- signal handlers ----
    def _on_structure_update(self, command: str, data: dict):
        """Handle structure update requests from child forms."""
        # Intermediate processing (logging/validation) can be added here
        if command == "INTEGRITY_WARNINGS":
            warns = data.get("warnings", []) if isinstance(data, dict) else []
            if warns:
                # Show a compact summary in the label area (non-blocking)
                summary = "\n".join([f"⚠️ {w}" for w in warns])
                self.header_label.setText(tr("Property Inspector") + "\n" + summary)
        self.structure_update_requested.emit(command, data)

    def _on_data_changed(self):
        """Handle simple data change notifications from forms without specific commands."""
        # For forms that just emit dataChanged without command/data parameters,
        # we emit a generic structure update signal
        self.structure_update_requested.emit("update", {})

    def setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(2)

        # ---- ヘッダー (タイトル + パンくず) ----
        header_widget = QWidget()
        header_layout = QVBoxLayout(header_widget)
        header_layout.setContentsMargins(0, 0, 0, 0)
        header_layout.setSpacing(1)

        self.header_label = QLabel(tr("Property Inspector"))
        self.header_label.setStyleSheet("font-weight: bold; font-size: 12px;")
        header_layout.addWidget(self.header_label)

        # パンくずラベル: ツリー上の位置を "カード > エフェクト > コマンド" 形式で表示
        self.breadcrumb_label = QLabel("")
        self.breadcrumb_label.setStyleSheet(
            "color: #888; font-size: 10px; padding: 1px 0px;"
        )
        self.breadcrumb_label.setVisible(False)
        header_layout.addWidget(self.breadcrumb_label)

        # CIR 状態表示ラベル: 選択中アイテムに正規化IRがあれば表示
        self.cir_label = QLabel("")
        self.cir_label.setStyleSheet("color: #2a7; font-size: 10px; padding: 1px 0px;")
        self.cir_label.setVisible(False)
        header_layout.addWidget(self.cir_label)

        # 区切り線
        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setStyleSheet("color: #555;")
        header_layout.addWidget(sep)

        layout.addWidget(header_widget)

        self.stack = QStackedWidget()

        self.empty_page = QLabel(tr("Select an item to edit"))
        self.empty_page.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.stack.addWidget(self.empty_page)

        # Replaced Label page with OptionForm
        self.option_form = OptionForm()
        self.stack.addWidget(self.option_form)

        # 再発防止: ブランチページは QLabel ではなく CmdBranchPage を使用すること。
        # QLabel に置き換えると「コマンド追加」ボタンが失われる。
        self.cmd_branch_page = CmdBranchPage("Branch selected. Add Commands to this branch.")
        safe_connect(self.cmd_branch_page, 'structure_update_requested', self._on_structure_update)
        self.stack.addWidget(self.cmd_branch_page)

        self.card_form = CardEditForm()
        self.stack.addWidget(self.card_form)
        safe_connect(self.card_form, 'structure_update_requested', self._on_structure_update)

        self.effect_form = EffectEditForm()
        self.stack.addWidget(self.effect_form)
        safe_connect(self.effect_form, 'structure_update_requested', self._on_structure_update)

        # Unified Action UI replaces separate Action/Command editors
        self.unified_form = UnifiedActionForm()
        self.stack.addWidget(self.unified_form)
        safe_connect(self.unified_form, 'structure_update_requested', self._on_structure_update)

        self.spell_side_form = SpellSideForm()
        self.stack.addWidget(self.spell_side_form)

        self.reaction_form = ReactionEditForm()
        self.stack.addWidget(self.reaction_form)

        self.keyword_form = KeywordEditForm()
        self.stack.addWidget(self.keyword_form)
        safe_connect(self.keyword_form, 'structure_update_requested', self._on_structure_update)

        self.modifier_form = ModifierEditForm()
        self.stack.addWidget(self.modifier_form)
        safe_connect(self.modifier_form, 'dataChanged', lambda: self._on_data_changed())

        layout.addWidget(self.stack)

        # Initialize dispatch table
        self.form_map = {
            "CARD": self.card_form,
            "EFFECT": self.effect_form,
            # 再発防止: 旧形式のアクションキーは後方互換として扱うが、
            # 新規コードは必ず "COMMAND" を使用すること。
            "LEGACY_CMD": self.unified_form,  # レガシー後方互換用
            "COMMAND": self.unified_form,
            "SPELL_SIDE": self.spell_side_form,
            "REACTION_ABILITY": self.reaction_form,
            "KEYWORDS": self.keyword_form,
            "MODIFIER": self.modifier_form,
            "OPTION": self.option_form, # Updated to use Form
            "CMD_BRANCH_TRUE": self.cmd_branch_page,
            "CMD_BRANCH_FALSE": self.cmd_branch_page,
        }

    def set_selection(self, index):
        # パンくず更新: 選択変更のたびに呼ぶ
        self._update_breadcrumb(index)

        if index is None or not index.isValid():
            self.stack.setCurrentWidget(self.empty_page)
            return

        item_type = index.data(Qt.ItemDataRole.UserRole + 1)
        item = index.model().itemFromIndex(index)

        widget = self.form_map.get(item_type, self.empty_page)

        if hasattr(widget, 'set_data'):
            widget.set_data(item)

        # Show CIR summary if available on the selected item
        try:
            cir = item.data('ROLE_CIR')
            if cir:
                self.cir_label.setText(tr("CIR entries: {n}").format(n=len(cir)))
                # Tooltip holds serialized CIR for debugging
                self.cir_label.setToolTip(str(cir))
                self.cir_label.setVisible(True)
            else:
                self.cir_label.setVisible(False)
        except Exception:
            self.cir_label.setVisible(False)

        self.stack.setCurrentWidget(widget)
