# -*- coding: utf-8 -*-
from PyQt6.QtWidgets import (
    QWidget, QFormLayout, QComboBox, QSpinBox, QLabel, QGroupBox,
    QVBoxLayout, QScrollArea
)
from PyQt6.QtCore import Qt
from dm_toolkit.gui.i18n import tr
from dm_toolkit.gui.editor.forms.base_form import BaseEditForm, get_attr, to_dict
from dm_toolkit.gui.editor.forms.parts.filter_widget import FilterEditorWidget
from dm_toolkit.gui.editor.forms.parts.condition_widget import ConditionEditorWidget
from dm_toolkit.gui.editor.forms.unified_widgets import make_player_scope_selector
from dm_toolkit.gui.editor.forms.parts.keyword_selector import KeywordSelectorWidget
from dm_toolkit.gui.editor.unified_filter_handler import UnifiedFilterHandler

class ModifierEditForm(BaseEditForm):
    """
    Form to edit a Static Ability (ModifierDef).
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        # Safe defaults for static/import checks
        self.basic_group = getattr(self, 'basic_group', None)
        self.condition_widget = getattr(self, 'condition_widget', None)
        self.filter_widget = getattr(self, 'filter_widget', None)
        self.type_combo = getattr(self, 'type_combo', None)
        self.keyword_combo = getattr(self, 'keyword_combo', None)
        self.scope_widget = getattr(self, 'scope_widget', None)
        self.scope_self_check = getattr(self, 'scope_self_check', None)
        self.scope_opp_check = getattr(self, 'scope_opp_check', None)
        self.bindings = getattr(self, 'bindings', {})
        try:
            self.setup_ui()
        except Exception as e:
            print(f"[ModifierForm.__init__] ERROR in setup_ui(): {e}")
            import traceback
            traceback.print_exc()

    def setup_ui(self):
        layout = QVBoxLayout(self)

        # Top section: Basic Modifier Properties
        self.basic_group = QGroupBox(tr("Modifier Settings"))
        form_layout = QFormLayout(self.basic_group)

        # Type
        self.type_combo = QComboBox()
        self.type_combo.currentTextChanged.connect(self.update_data)
        self.type_combo.currentTextChanged.connect(self.update_visibility)
        self.register_widget(self.type_combo, 'type')
        form_layout.addRow(tr("Type"), self.type_combo)

        # Restriction Type (Combo Box)
        # NOTE: Legacy UI kept for backward compatibility, but restriction selection is now done via type_combo.
        self.restriction_combo = QComboBox()
        restriction_types = ["TARGET_RESTRICTION", "SPELL_RESTRICTION"]
        for t in restriction_types:
            self.restriction_combo.addItem(tr(t), t)
        self.restriction_combo.currentTextChanged.connect(self.update_data)
        # We don't register it directly with register_widget because it maps to mutation_kind conditionally
        # self.register_widget(self.restriction_combo)
        self.label_restriction = QLabel(tr("Restriction Type"))
        form_layout.addRow(self.label_restriction, self.restriction_combo)

        # Value (for Power/Cost)
        self.value_spin = QSpinBox()
        self.value_spin.setRange(-99999, 99999)
        self.value_spin.valueChanged.connect(self.update_data)
        self.register_widget(self.value_spin, 'value')
        self.label_value = QLabel(tr("Value"))
        form_layout.addRow(self.label_value, self.value_spin)

        # Keyword Selection - Unified Widget
        self.keyword_combo = KeywordSelectorWidget(allow_settable=True)
        self.keyword_combo.keywordSelected.connect(self.update_data)
        self.register_widget(self.keyword_combo)
        self.label_keyword = QLabel(tr("Keyword"))
        form_layout.addRow(self.label_keyword, self.keyword_combo)

        layout.addWidget(self.basic_group)

        # Scope Section
        scope_group = QGroupBox(tr("Scope (Owner)"))
        scope_layout = QFormLayout(scope_group)
        self.scope_widget, self.scope_self_check, self.scope_opp_check = make_player_scope_selector()
        self.scope_self_check.toggled.connect(self.update_data)
        self.scope_opp_check.toggled.connect(self.update_data)
        self.register_widget(self.scope_self_check)
        self.register_widget(self.scope_opp_check)
        scope_layout.addRow(tr("Owner"), self.scope_widget)
        layout.addWidget(scope_group)

        # Condition Section
        self.condition_widget = ConditionEditorWidget()
        self.condition_widget.dataChanged.connect(self.update_data)
        layout.addWidget(self.condition_widget)

        # Filter Section - Unified Handler (wrap in scroll area)
        self.filter_widget = UnifiedFilterHandler.create_filter_widget("STATIC", self)
        self.filter_widget.filterChanged.connect(self.update_data)
        self.filter_widget.filterChanged.connect(self._refresh_type_combo_items)
        self.filter_area = QScrollArea()
        self.filter_area.setWidgetResizable(True)
        self.filter_area.setWidget(self.filter_widget)
        layout.addWidget(self.filter_area)

        # Define bindings
        # Note: str_val と scope は手動で処理される（_populate_ui と _save_data）
        self.bindings = {
            'type': self.type_combo,
            'value': self.value_spin,
            'condition': self.condition_widget,
            'filter': self.filter_widget
        }

        # Initial visibility
        self._refresh_type_combo_items()
        self.update_visibility()

    def _is_filter_empty(self) -> bool:
        try:
            filt = self.filter_widget.get_data()
        except Exception:
            return True
        return not bool(filt)

    def _refresh_type_combo_items(self):
        """Rebuild Type combo items.

        - Always show core modifier types.
        - Always show TARGET_RESTRICTION / SPELL_RESTRICTION as top-level type choices.
        - If filter is empty, also show self-only target selection adjustments.
        """
        prev = self.type_combo.currentData()

        # If the previous selection was a legacy ADD_RESTRICTION, prefer its subtype.
        if prev == "ADD_RESTRICTION":
            prev = self.restriction_combo.currentData() or prev

        allow_self_only = self._is_filter_empty()

        # If filter is not empty, self-only options must not be selectable
        self_only = {"TARGET_THIS_CANNOT_SELECT", "TARGET_THIS_FORCE_SELECT"}
        if not allow_self_only and prev in self_only:
            prev = "TARGET_RESTRICTION"

        types: list[str] = [
            "NONE",
            "COST_MODIFIER",
            "POWER_MODIFIER",
            "GRANT_KEYWORD",
            "SET_KEYWORD",
            # Restrictions are now top-level choices
            "TARGET_RESTRICTION",
            "SPELL_RESTRICTION",
        ]

        if allow_self_only:
            types.extend([
                "TARGET_THIS_CANNOT_SELECT",
                "TARGET_THIS_FORCE_SELECT",
            ])

        # Preserve unknown/legacy value if present (exclude disallowed self-only)
        if prev and prev not in types and prev not in self_only:
            types.append(str(prev))

        self.type_combo.blockSignals(True)
        self.type_combo.clear()
        for t in types:
            self.type_combo.addItem(tr(t), t)

        # Restore selection
        if prev:
            idx = self.type_combo.findData(prev)
            if idx >= 0:
                self.type_combo.setCurrentIndex(idx)
            else:
                self.type_combo.setCurrentIndex(0)
        self.type_combo.blockSignals(False)

        # Ensure dependent widgets reflect the final selection
        self.update_visibility()

    def update_visibility(self):
        mtype = self.type_combo.currentData()
        if mtype is None:
            mtype = "COST_MODIFIER"  # Default fallback

        # Defaults
        self.label_value.setVisible(False)
        self.value_spin.setVisible(False)
        self.label_keyword.setVisible(False)
        self.keyword_combo.setVisible(False)
        # Restriction combo is legacy; hide by default
        self.label_restriction.setVisible(False)
        self.restriction_combo.setVisible(False)

        if mtype == "COST_MODIFIER":
            self.label_value.setVisible(True)
            self.value_spin.setVisible(True)
            self.label_value.setText("軽減量")
            self.filter_widget.setTitle("軽減対象カード")

        elif mtype == "POWER_MODIFIER":
            self.label_value.setVisible(True)
            self.value_spin.setVisible(True)
            self.label_value.setText("パワー修正値")
            self.filter_widget.setTitle("強化対象クリーチャー")

        elif mtype == "GRANT_KEYWORD":
            self.label_keyword.setVisible(True)
            self.keyword_combo.setVisible(True)
            self.filter_widget.setTitle("対象クリーチャー")

        elif mtype in ("TARGET_RESTRICTION", "SPELL_RESTRICTION", "TARGET_THIS_CANNOT_SELECT", "TARGET_THIS_FORCE_SELECT", "ADD_RESTRICTION"):
            # Unified restriction types
            self.filter_widget.setTitle("制限対象")

        elif mtype == "SET_KEYWORD":
            self.label_keyword.setVisible(True)
            self.keyword_combo.setVisible(True)
            self.filter_widget.setTitle("対象クリーチャー")

    def _load_ui_from_data(self, data, item):
        """Load data into UI widgets."""
        if not data:
            data = {}
        else:
            data = to_dict(data)

        # Fallbacks for empty structures
        if not data.get('condition'):
             self.condition_widget.set_data({"type": "NONE"})
        if not data.get('filter'):
             self.filter_widget.set_data({})

        # Apply basic bindings (type, value, condition, filter)
        self._apply_bindings(data)
        
        # Manually set scope (checkboxes)
        # Note: Signals are already blocked by load_data(), no need to block here
        scope = data.get('scope', 'ALL')
        
        # Set checkboxes based on scope
        if scope == 'SELF':
            self.scope_self_check.setChecked(True)
            self.scope_opp_check.setChecked(False)
        elif scope == 'OPPONENT':
            self.scope_self_check.setChecked(False)
            self.scope_opp_check.setChecked(True)
        else:  # 'ALL' or default - both checked
            self.scope_self_check.setChecked(True)
            self.scope_opp_check.setChecked(True)
        
        # Manually set keyword combo (mutation_kind or str_val)
        # Note: Signals are already blocked by load_data(), no need to block here
        # Prefer mutation_kind, fallback to str_val for legacy data
        keyword = data.get('mutation_kind', '') or data.get('str_val', '')

        # Refresh type choices before selecting
        self._refresh_type_combo_items()

        # Load into appropriate widget based on type
        mtype = data.get('type', '')
        if mtype == 'ADD_RESTRICTION':
            # New UI: the restriction sub-type is selected directly in type_combo
            chosen = keyword or 'TARGET_RESTRICTION'
            idx = self.type_combo.findData(chosen)
            if idx < 0:
                # Ensure unknown legacy value remains selectable
                self._refresh_type_combo_items()
                idx = self.type_combo.findData(chosen)
            if idx >= 0:
                self.type_combo.setCurrentIndex(idx)
            else:
                self.type_combo.setCurrentIndex(0)
            # Keep legacy widget in sync (hidden)
            legacy_idx = self.restriction_combo.findData(chosen)
            if legacy_idx >= 0:
                self.restriction_combo.setCurrentIndex(legacy_idx)
        else:
            # Default behavior for keywords
            if keyword:
                self.keyword_combo.set_keyword(keyword)
            else:
                self.keyword_combo.setCurrentIndex(0)

        # Update visibility based on current modifier type
        self.update_visibility()

    def _save_ui_to_data(self, data):
        """Save form values to data dictionary."""
        self._collect_bindings(data)

        # Determine scope from checkboxes
        from dm_toolkit.consts import TargetScope
        self_checked = self.scope_self_check.isChecked()
        opp_checked = self.scope_opp_check.isChecked()
        
        if self_checked and opp_checked:
            scope_value = TargetScope.ALL
        elif self_checked and not opp_checked:
            scope_value = TargetScope.SELF
        elif not self_checked and opp_checked:
            scope_value = TargetScope.OPPONENT
        else:  # Neither checked
            scope_value = TargetScope.ALL  # Default
        
        data['scope'] = scope_value
        
        # Save keyword to mutation_kind (NEW) for GRANT_KEYWORD/SET_KEYWORD
        # Also save to str_val for backward compatibility
        mtype = data.get('type', '')
        if mtype in ('GRANT_KEYWORD', 'SET_KEYWORD'):
            keyword = self.keyword_combo.get_keyword()
            data['mutation_kind'] = keyword  # Primary field
            data['str_val'] = keyword  # Legacy support
        elif mtype in ('TARGET_RESTRICTION', 'SPELL_RESTRICTION', 'TARGET_THIS_CANNOT_SELECT', 'TARGET_THIS_FORCE_SELECT', 'ADD_RESTRICTION'):
            # Persist as legacy ADD_RESTRICTION + subtype for compatibility
            ui_choice = self.type_combo.currentData() or 'TARGET_RESTRICTION'
            if ui_choice == 'ADD_RESTRICTION':
                ui_choice = self.restriction_combo.currentData() or 'TARGET_RESTRICTION'
            data['type'] = 'ADD_RESTRICTION'
            data['mutation_kind'] = ui_choice
            data['str_val'] = ui_choice
        else:
            # Clear both fields for non-keyword types
            data['mutation_kind'] = ''
            data['str_val'] = ''

    def _get_display_text(self, data):
        """Generate concise display text for tree item."""
        from dm_toolkit.gui.editor.text_resources import CardTextResources
        from dm_toolkit.consts import TargetScope
        
        mtype = data.get('type', 'NONE')
        # Prefer mutation_kind, fallback to str_val
        keyword = data.get('mutation_kind', '') or data.get('str_val', '')
        scope = data.get('scope', TargetScope.ALL)
        value = data.get('value', 0)
        
        # Build concise label
        parts = []
        
        # Add scope prefix if not ALL
        if scope == TargetScope.SELF or scope == 'SELF':
            parts.append('自分の')
        elif scope == TargetScope.OPPONENT or scope == 'OPPONENT':
            parts.append('相手の')
        
        # Add main type description
        if mtype == 'COST_MODIFIER':
            sign = '+' if value < 0 else '-'
            parts.append(f'コスト{sign}{abs(value)}')
        elif mtype == 'POWER_MODIFIER':
            sign = '+' if value >= 0 else ''
            parts.append(f'パワー{sign}{value}')
        elif mtype == 'GRANT_KEYWORD':
            if keyword:
                keyword_display = CardTextResources.get_keyword_text(keyword)
                parts.append(f'付与:{keyword_display}')
            else:
                parts.append('付与:未設定')
        elif mtype == 'SET_KEYWORD':
            if keyword:
                keyword_display = CardTextResources.get_keyword_text(keyword)
                parts.append(f'獲得:{keyword_display}')
            else:
                parts.append('獲得:未設定')
        elif mtype == 'ADD_RESTRICTION':
            if keyword:
                parts.append(tr(keyword))
            else:
                parts.append(tr('ADD_RESTRICTION'))
        else:
            parts.append(tr(mtype))
        
        result = ''.join(parts)
        
        # Limit to 30 characters for tree display
        if len(result) > 30:
            return result[:27] + "..."
        return result
