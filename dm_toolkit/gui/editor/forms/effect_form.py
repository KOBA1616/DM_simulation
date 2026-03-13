# -*- coding: utf-8 -*-
from PyQt6.QtWidgets import QWidget, QFormLayout, QComboBox, QGroupBox, QGridLayout, QCheckBox, QSpinBox, QLabel, QLineEdit, QPushButton, QScrollArea
from PyQt6.QtCore import Qt, pyqtSignal
from dm_toolkit.gui.i18n import tr
from dm_toolkit.gui.editor.forms.base_form import BaseEditForm, get_attr, to_dict
from dm_toolkit.gui.editor.forms.parts.filter_widget import FilterEditorWidget
from dm_toolkit.gui.editor.forms.parts.condition_widget import ConditionEditorWidget
from dm_toolkit.consts import TRIGGER_TYPES, SPELL_TRIGGER_TYPES, LAYER_TYPES
from dm_toolkit.gui.editor.forms.parts.keyword_selector import KeywordSelectorWidget
from dm_toolkit.gui.editor.unified_filter_handler import UnifiedFilterHandler
from dm_toolkit.gui.editor.forms.signal_utils import safe_connect
from dm_toolkit.gui.editor.consistency import validate_trigger_scope_filter

class EffectEditForm(BaseEditForm):
    structure_update_requested = pyqtSignal(str, dict)

    def __init__(self, parent=None):
        super().__init__(parent)
        # Safe defaults for headless/static import
        self.form_layout = getattr(self, 'form_layout', None)
        self.mode_combo = getattr(self, 'mode_combo', None)
        self.trigger_combo = getattr(self, 'trigger_combo', None)
        self.trigger_scope_combo = getattr(self, 'trigger_scope_combo', None)
        self.trigger_filter = getattr(self, 'trigger_filter', None)
        self.layer_group = getattr(self, 'layer_group', None)
        self.layer_type_combo = getattr(self, 'layer_type_combo', None)
        self.target_filter = getattr(self, 'target_filter', None)
        self.condition_widget = getattr(self, 'condition_widget', None)
        try:
            self.setup_ui()
        except Exception:
            pass

    def setup_ui(self):
        self.form_layout = QFormLayout(self)

        # Ability Mode
        self.mode_combo = QComboBox()
        self.mode_combo.addItem(tr("TRIGGERED"), "TRIGGERED")
        self.mode_combo.addItem(tr("STATIC"), "STATIC")
        self.mode_combo.addItem(tr("REPLACEMENT"), "REPLACEMENT")
        self.register_widget(self.mode_combo)
        self.add_field(tr("Ability Mode"), self.mode_combo)

        # Trigger Definition
        self.trigger_combo = QComboBox()
        # Initial population, will be updated by Logic Mask
        self.populate_combo(self.trigger_combo, TRIGGER_TYPES, display_func=tr, data_func=lambda x: x)
        self.lbl_trigger = self.add_field(tr("Trigger"), self.trigger_combo, 'trigger')

        # Trigger Scope
        self.trigger_scope_combo = QComboBox()

        # Explicitly define scope options as requested
        # Order: Self Player, Opponent Player, This Creature, Both Players
        self.trigger_scope_combo.addItem(tr("自プレイヤー"), "PLAYER_SELF")
        self.trigger_scope_combo.addItem(tr("相手プレイヤー"), "PLAYER_OPPONENT")
        self.trigger_scope_combo.addItem(tr("このクリーチャー"), "NONE")
        self.trigger_scope_combo.addItem(tr("両プレイヤー"), "ALL_PLAYERS")

        self.register_widget(self.trigger_scope_combo, 'trigger_scope')
        self.lbl_scope = self.add_field(tr("Trigger Scope"), self.trigger_scope_combo)

        # Trigger Filter
        self.trigger_filter_group = QGroupBox(tr("Trigger Filter"))
        tf_layout = QGridLayout(self.trigger_filter_group)
        # Create filter widget and wrap in scroll area to avoid layout overlap
        self.trigger_filter = UnifiedFilterHandler.create_filter_widget("TRIGGER", self)
        safe_connect(self.trigger_filter, "filterChanged", self.update_data)
        safe_connect(self.trigger_filter, "filterChanged", self.on_trigger_filter_changed)
        self.register_widget(self.trigger_filter, 'trigger_filter')
        self.trigger_filter_area = QScrollArea()
        self.trigger_filter_area.setWidgetResizable(True)
        self.trigger_filter_area.setWidget(self.trigger_filter)
        tf_layout.addWidget(self.trigger_filter_area, 0, 0)
        
        # Trigger Filter Description Label
        self.trigger_filter_desc_label = QLabel("")
        self.trigger_filter_desc_label.setWordWrap(True)
        self.trigger_filter_desc_label.setStyleSheet("color: #666; font-size: 10px; font-style: italic;")
        tf_layout.addWidget(self.trigger_filter_desc_label, 1, 0)
        
        self.add_field(None, self.trigger_filter_group)

        # Layer Definition (Static)
        self.layer_group = QGroupBox(tr("Layer Definition"))
        l_layout = QGridLayout(self.layer_group)

        self.layer_type_combo = QComboBox()
        self.populate_combo(self.layer_type_combo, LAYER_TYPES, display_func=tr, data_func=lambda x: x)
        self.register_widget(self.layer_type_combo, 'type')

        self.layer_val_spin = QSpinBox()
        self.layer_val_spin.setRange(-9999, 9999)
        self.register_widget(self.layer_val_spin, 'value')

        self.layer_str_edit = QLineEdit()
        self.register_widget(self.layer_str_edit, 'str_val')
        
        # Keyword Helper - Unified Widget
        self.layer_keyword_combo = KeywordSelectorWidget(allow_settable=True)
        safe_connect(self.layer_keyword_combo, "keywordSelected", self.on_layer_keyword_changed)

        l_layout.addWidget(QLabel(tr("Layer Type")), 0, 0)
        l_layout.addWidget(self.layer_type_combo, 0, 1)
        l_layout.addWidget(QLabel(tr("Value")), 1, 0)
        l_layout.addWidget(self.layer_val_spin, 1, 1)
        l_layout.addWidget(QLabel(tr("String/Keyword")), 2, 0)
        l_layout.addWidget(self.layer_str_edit, 2, 1)
        l_layout.addWidget(QLabel(tr("Select Keyword")), 3, 0)
        l_layout.addWidget(self.layer_keyword_combo, 3, 1)

        # Target Filter - Unified Handler
        self.target_filter = UnifiedFilterHandler.create_filter_widget("STATIC", self)
        safe_connect(self.target_filter, "filterChanged", self.update_data)
        self.register_widget(self.target_filter, 'filter')
        self.target_filter_area = QScrollArea()
        self.target_filter_area.setWidgetResizable(True)
        self.target_filter_area.setWidget(self.target_filter)
        l_layout.addWidget(QLabel(tr("Target Filter")), 4, 0)
        l_layout.addWidget(self.target_filter_area, 4, 1)

        self.add_field(None, self.layer_group)

        # Condition (Shared)
        self.condition_widget = ConditionEditorWidget()
        safe_connect(self.condition_widget, "dataChanged", self.update_data)
        self.add_field(None, self.condition_widget, 'condition')

        # Actions Section
        self.add_action_btn = QPushButton(tr("Add Command"))
        safe_connect(self.add_action_btn, "clicked", self.on_add_action_clicked)
        self.add_field(None, self.add_action_btn)

        # Connect signals
        safe_connect(self.mode_combo, "currentIndexChanged", self.on_mode_changed)
        safe_connect(self.mode_combo, "currentIndexChanged", self.update_data)

        safe_connect(self.trigger_combo, "currentIndexChanged", self.update_data)
        safe_connect(self.trigger_scope_combo, "currentIndexChanged", self.update_data)

        safe_connect(self.layer_type_combo, "currentIndexChanged", self.update_data)
        safe_connect(self.layer_type_combo, "currentIndexChanged", self.update_layer_keyword_visibility)
        safe_connect(self.layer_val_spin, "valueChanged", self.update_data)
        safe_connect(self.layer_str_edit, "textChanged", self.update_data)

        # Initial UI State
        self.on_mode_changed()

    def on_mode_changed(self):
        mode = self.mode_combo.currentData()
        if mode is None:
            mode = "TRIGGERED"  # Default fallback
        # TRIGGERED と REPLACEMENT はどちらもトリガー系UIを共有する
        is_trigger_based = (mode in ("TRIGGERED", "REPLACEMENT"))

        self.trigger_combo.setVisible(is_trigger_based)
        self.lbl_trigger.setVisible(is_trigger_based)

        self.trigger_scope_combo.setVisible(is_trigger_based)
        self.lbl_scope.setVisible(is_trigger_based)

        self.trigger_filter_group.setVisible(is_trigger_based)

        self.layer_group.setVisible(mode == "STATIC")

        if mode == "TRIGGERED":
            self.condition_widget.setTitle(tr("Trigger Condition"))
        elif mode == "REPLACEMENT":
            self.condition_widget.setTitle(tr("置換条件"))
        else:
            self.condition_widget.setTitle(tr("Apply Condition"))

        # 再発防止: モード切替時にトリガー候補を再生成しないと、REPLACEMENTで誘発用候補が残る。
        card_type = self._get_current_card_type()
        self.update_trigger_options(card_type, mode)
        
        # Update keyword combo visibility
        self.update_layer_keyword_visibility()

    def _get_current_card_type(self) -> str:
        """Return current card type from tree context (CREATURE/SPELL)."""
        if not self.current_item:
            return "CREATURE"

        card_type = "CREATURE"
        parent = self.current_item.parent()
        if parent:
            grandparent = parent.parent()
            if grandparent:
                role = grandparent.data(Qt.ItemDataRole.UserRole + 1)
                if role == "SPELL_SIDE":
                    card_type = "SPELL"
                elif role == "CARD":
                    cdata = grandparent.data(Qt.ItemDataRole.UserRole + 2)
                    card_type = get_attr(cdata, 'type', 'CREATURE')
        return card_type

    @staticmethod
    def _to_replacement_trigger_label(trigger_text: str) -> str:
        """Convert trigger text from post-event tone (〜た時) to replacement tone (〜る時)."""
        text = trigger_text
        replacements = [
            ("された時", "される時"),
            ("置かれた時", "置かれる時"),
            ("唱えた時", "唱える時"),
            ("引いた時", "引く時"),
            ("出た時", "出る時"),
            ("した時", "する時"),
            ("った時", "る時"),
        ]
        for src, dst in replacements:
            if src in text:
                return text.replace(src, dst)
        return text
    
    def on_layer_keyword_changed(self, keyword: str):
        """Update str_val when keyword is selected from unified widget."""
        if keyword:
            self.layer_str_edit.setText(keyword)
        self.update_data()
    
    def update_layer_keyword_visibility(self):
        """Show keyword combo only for GRANT_KEYWORD or SET_KEYWORD types"""
        layer_type = self.layer_type_combo.currentData() if hasattr(self, 'layer_type_combo') else None
        show_keyword = layer_type in ['GRANT_KEYWORD', 'SET_KEYWORD']
        if hasattr(self, 'layer_keyword_combo'):
            self.layer_keyword_combo.setVisible(show_keyword)

    def on_add_action_clicked(self):
        self.structure_update_requested.emit("ADD_CHILD_ACTION", {})

    def _load_ui_from_data(self, data, item):
        """
        Populate UI from data (Hook).
        """
        # Convert to dict if needed
        data = to_dict(data)
        card_type = "CREATURE"
        
        item_type = "EFFECT"
        if item:
            item_type = item.data(Qt.ItemDataRole.UserRole + 1)

            # Logic Mask: Filter triggers based on Card Type
            parent = item.parent() # Group
            if parent:
                grandparent = parent.parent() # Card or Spell Side
                if grandparent:
                    role = grandparent.data(Qt.ItemDataRole.UserRole + 1)
                    if role == "SPELL_SIDE":
                        card_type = "SPELL"
                    elif role == "CARD":
                         cdata = grandparent.data(Qt.ItemDataRole.UserRole + 2)
                         card_type = get_attr(cdata, 'type', 'CREATURE')

        # Determine Mode
        mode = "TRIGGERED"
        if item_type == "MODIFIER":
            mode = "STATIC"
        elif data.get('mode') == "REPLACEMENT" or data.get('timing_mode') == "PRE":
            mode = "REPLACEMENT"
        elif 'layer_type' in data or 'type' in data and item_type != "EFFECT":
            # Legacy check or inferred from data
            mode = "STATIC"

        self.set_combo_by_data(self.mode_combo, mode)
        # Build trigger options after mode has been decided.
        self.update_trigger_options(card_type, mode)

        # 後方互換: 旧実装の PRE:TRIGGER を読み込んだ場合は通常トリガーへ正規化
        if mode == "REPLACEMENT" and isinstance(data.get('trigger'), str):
            trg = data.get('trigger', '')
            if trg.startswith("PRE:"):
                data['trigger'] = trg.split(":", 1)[1]

        # Trigger visibility update immediately
        self.on_mode_changed()

        if mode in ("TRIGGERED", "REPLACEMENT"):
             # Try to normalize data for binding if legacy keys exist
             if 'trigger_condition' in data and 'condition' not in data:
                 data['condition'] = data['trigger_condition']

             # Load Trigger Filter explicitly
             if 'trigger_filter' in data and self.trigger_filter:
                 self.trigger_filter.set_data(data['trigger_filter'])
             else:
                 self.trigger_filter.set_data({})
        else:
            # STATIC (ModifierDef) - Normalize for bindings
            if 'layer_type' in data: data['type'] = data['layer_type']
            if 'layer_value' in data: data['value'] = data['layer_value']
            if 'layer_str' in data: data['str_val'] = data['layer_str']
            if 'static_condition' in data and 'condition' not in data:
                 data['condition'] = data['static_condition']

        # Use Bindings
        self._apply_bindings(data)
        
        # Set keyword combo based on str_val for STATIC mode
        if mode == "STATIC":
            str_val = data.get('str_val', '')
            if str_val and hasattr(self, 'layer_keyword_combo'):
                self.set_combo_by_data(self.layer_keyword_combo, str_val)

        # Ensure fallback for condition if missing
        if not data.get('condition'):
             self.condition_widget.set_data({})
        
        # Update keyword combo visibility
        self.update_layer_keyword_visibility()

    def update_trigger_options(self, card_type, mode=None):
        if mode is None:
            mode = self.mode_combo.currentData() or "TRIGGERED"
        is_spell = (card_type == "SPELL")

        base_allowed = SPELL_TRIGGER_TYPES if is_spell else TRIGGER_TYPES
        if mode == "REPLACEMENT":
            # 置換モードは表示のみ「〜る時」にし、内部値は通常トリガーを保持する
            allowed = [(self._to_replacement_trigger_label(tr(t)), t) for t in base_allowed]
        else:
            allowed = [(tr(t), t) for t in base_allowed]

        current_data = self.trigger_combo.currentData()
        if isinstance(current_data, str):
            if current_data.startswith("PRE:"):
                current_data = current_data.split(":", 1)[1]

        self.trigger_combo.blockSignals(True)
        self.trigger_combo.clear()

        # Ensure current data is preserved if it was valid before or legacy
        current_values = {item[1] for item in allowed}
        if current_data and current_data not in current_values:
            if mode == "REPLACEMENT" and isinstance(current_data, str):
                base = current_data.split(":", 1)[1] if current_data.startswith("PRE:") else current_data
                allowed.append((self._to_replacement_trigger_label(tr(base)), base))
            elif isinstance(current_data, str):
                allowed.append((tr(current_data), current_data))

        self.populate_combo(self.trigger_combo, allowed, clear=True)

        # Restore selection
        idx = self.trigger_combo.findData(current_data)
        if idx >= 0:
            self.trigger_combo.setCurrentIndex(idx)
        else:
            self.trigger_combo.setCurrentIndex(0)

        self.trigger_combo.blockSignals(False)

    def _save_ui_to_data(self, data):
        """
        Save UI to data (Hook).
        """
        mode = self.mode_combo.currentData()

        # Apply bindings (collects into data)
        self._collect_bindings(data)

        # 後方互換: 旧実装の内部値 PRE:<TRIGGER> を正規化
        raw_trigger = data.get('trigger')
        if isinstance(raw_trigger, str) and raw_trigger.startswith('PRE:'):
            data['trigger'] = raw_trigger.split(':', 1)[1]

        # Post-processing based on Mode
        if self.current_item:
            current_type_code = self.current_item.data(Qt.ItemDataRole.UserRole + 1)
            target_type_code = "EFFECT" if mode in ("TRIGGERED", "REPLACEMENT") else "MODIFIER"

            # Update type if changed
            if current_type_code != target_type_code:
                self.current_item.setData(target_type_code, Qt.ItemDataRole.UserRole + 1)
                # Emit MOVE_EFFECT to trigger UI/Label updates in the tree
                self.structure_update_requested.emit("MOVE_EFFECT", {"item": self.current_item, "target_type": mode})

        if mode == "TRIGGERED":
            # Explicitly save trigger filter from widget (bindings might not catch it if it's complex/custom getter)
            data['trigger_filter'] = self.trigger_filter.get_data()
            data.pop('mode', None)  # TRIGGERED はデフォルトなので mode フィールド不要
            # Consistency validation: warn about duplicate or conflicting settings
            try:
                warns = validate_trigger_scope_filter(data)
                if warns:
                    # Emit non-blocking warning event for inspector/host
                    self.structure_update_requested.emit("INTEGRITY_WARNINGS", {"warnings": warns})
            except Exception:
                # Be robust in headless/unit test environments
                pass

            # Clean Static/Legacy keys
            for k in ['type', 'value', 'str_val', 'filter', 'layer_type', 'layer_value', 'layer_str', 'static_condition', 'trigger_condition']:
                data.pop(k, None)

        elif mode == "REPLACEMENT":
            # 置換効果: トリガー系フィールドを保存し mode='REPLACEMENT' を明示する
            data['mode'] = 'REPLACEMENT'
            data['timing_mode'] = 'PRE'
            data['trigger_filter'] = self.trigger_filter.get_data()
            for k in ['type', 'value', 'str_val', 'filter', 'layer_type', 'layer_value', 'layer_str', 'static_condition', 'trigger_condition']:
                data.pop(k, None)

        else: # STATIC
            # Handle str_val optionality
            if not self.layer_str_edit.text():
                data.pop('str_val', None)

            # Clean Trigger/Legacy keys
            # Preserve trigger_scope to maintain user selection across mode toggles.
            # Also preserve trigger_filter so that returning to TRIGGERED restores previous filter.
            # Other legacy/static normalization keys can be safely cleaned.
            for k in ['trigger', 'trigger_condition', 'layer_type', 'layer_value', 'layer_str', 'static_condition', 'mode', 'timing_mode']:
                data.pop(k, None)

    def _get_display_text(self, data):
        if 'trigger' in data:
             scope = data.get('trigger_scope', 'NONE')
             if scope == "NONE":
                 scope_str = "" # Implicit "This Creature"
             elif scope == "PLAYER_SELF":
                 scope_str = " (自プレイヤー)"
             elif scope == "PLAYER_OPPONENT":
                 scope_str = " (相手プレイヤー)"
             elif scope == "ALL_PLAYERS":
                 scope_str = " (両プレイヤー)"
             else:
                 # 再発防止: 未翻訳/空文字スコープで "()" が表示されるのを防ぐ。
                 scope_label = tr(scope).strip()
                 scope_str = f" ({scope_label})" if scope_label else ""

             if data.get('mode') == 'REPLACEMENT':
                 replacement_trigger = self._to_replacement_trigger_label(tr(data.get('trigger', '')))
                 return f"{tr('REPLACEMENT')}: {replacement_trigger}{scope_str}"
             return f"{tr('Effect')}: {tr(data.get('trigger', ''))}{scope_str}"
        elif 'type' in data or 'layer_type' in data:
             t = data.get('type', data.get('layer_type', ''))
             return f"{tr('Static')}: {tr(t)}"
        else:
             return tr("Unknown Effect")
    def on_trigger_filter_changed(self):
        """Update trigger filter description when filter changes."""
        try:
            trigger_filter = self.trigger_filter.get_data()
            if not trigger_filter:
                self.trigger_filter_desc_label.setText("")
                return
            
            # Import CardTextGenerator for description generation
            from dm_toolkit.gui.editor.text_generator import CardTextGenerator
            
            desc = CardTextGenerator.generate_trigger_filter_description(trigger_filter)
            if desc:
                self.trigger_filter_desc_label.setText(tr("📋 条件: {desc}").format(desc=desc))
            else:
                self.trigger_filter_desc_label.setText("")
        except Exception:
            # Gracefully handle errors in headless environments
            self.trigger_filter_desc_label.setText("")