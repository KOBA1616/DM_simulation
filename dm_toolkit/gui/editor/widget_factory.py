# -*- coding: utf-8 -*-
from PyQt6.QtWidgets import (
    QWidget, QCheckBox, QHBoxLayout, QVBoxLayout, QPushButton, QLabel, QLineEdit, QComboBox
)
from dm_toolkit.gui.editor.widgets.common import (
    ZoneCombo, ScopeCombo, TextWidget, NumberWidget, AmountWithAllWidget, QuantityModeWidget, MultiZoneSelector, BoolCheckWidget, EditorWidgetMixin
)
from dm_toolkit.gui.editor.forms.unified_widgets import (
    make_player_scope_selector, make_measure_mode_combo, make_ref_mode_combo
)
from dm_toolkit.gui.editor.schema_def import FieldType, FieldSchema
import importlib
from dm_toolkit.gui.editor.forms.signal_utils import safe_connect

# Import correct widgets from actual file structure
from dm_toolkit.gui.editor.forms.parts.filter_widget import FilterEditorWidget
from dm_toolkit.gui.editor.forms.parts.variable_link_widget import VariableLinkWidget
from dm_toolkit.gui.editor.forms.parts.civilization_widget import CivilizationSelector
from dm_toolkit.gui.editor.forms.parts.condition_widget import ConditionEditorWidget
from dm_toolkit.consts import CARD_TYPES
from dm_toolkit.gui.i18n import tr
from dm_toolkit.gui.editor.text_resources import CardTextResources # Import for duration text

class PlayerScopeWidget(QWidget, EditorWidgetMixin):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.w, self.self_chk, self.opp_chk = make_player_scope_selector(self)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0,0,0,0)
        layout.addWidget(self.w)

    def get_value(self):
        s = self.self_chk.isChecked()
        o = self.opp_chk.isChecked()
        if s and o: return 'PLAYER_BOTH'
        if o: return 'PLAYER_OPPONENT'
        return 'PLAYER_SELF'

    def set_value(self, value):
        self.self_chk.blockSignals(True)
        self.opp_chk.blockSignals(True)
        self.self_chk.setChecked(value in ['PLAYER_SELF', 'PLAYER_BOTH', 'SELF'])
        self.opp_chk.setChecked(value in ['PLAYER_OPPONENT', 'PLAYER_BOTH', 'OPPONENT'])
        self.self_chk.blockSignals(False)
        self.opp_chk.blockSignals(False)

class FilterEditorWrapper(FilterEditorWidget, EditorWidgetMixin):
    def get_value(self):
        return self.get_data()

    def set_value(self, value):
        self.set_data(value)

class VariableLinkWrapper(VariableLinkWidget, EditorWidgetMixin):
    def get_value(self):
        d = {}
        self.get_data(d)
        return d

    def set_value(self, value):
        self.set_data(value)

class OptionsControlWidget(QWidget):
    """Container for option generation controls."""
    def __init__(self, parent, callback):
        super().__init__(parent)

        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)

        control_widget = QWidget()
        self.layout = QHBoxLayout(control_widget)
        self.layout.setContentsMargins(0, 0, 0, 0)

        self.spin = NumberWidget(self, 1, 10)
        self.btn = QPushButton(tr("Generate Branches"))
        self.btn.setStyleSheet("background-color: #007bff; color: white; font-weight: bold;")
        safe_connect(self.btn, 'clicked', callback)
        self.label = QLabel(tr("Number of branches to generate:"))

        self.layout.addWidget(self.label)
        self.layout.addWidget(self.spin)
        self.layout.addWidget(self.btn)

        main_layout.addWidget(control_widget)

        # Guide text to clarify where the choices are edited
        guide_label = QLabel(tr("Note: Clicking 'Generate Branches' will create child nodes in the logic tree below. You can then edit each branch independently."))
        guide_label.setStyleSheet("color: #666; font-style: italic; font-size: 11px;")
        guide_label.setWordWrap(True)
        main_layout.addWidget(guide_label)

        # Expose spin as property for factory
        self.option_layout = self.layout

    def get_value(self):
        return self.spin.value()

    def set_value(self, value):
        self.spin.setValue(int(value))

class RefModeComboWrapper(QWidget, EditorWidgetMixin):
    """Wrapper for the unified ref mode combo."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.combo = make_ref_mode_combo(self)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0,0,0,0)
        layout.addWidget(self.combo)

    def get_value(self):
        data = self.combo.currentData()
        # Return None if empty item is selected
        return None if data is None else data

    def set_value(self, value):
        if value is None:
            # Select empty item (index 0 if it exists and has None data)
            if self.combo.count() > 0 and self.combo.itemData(0) is None:
                self.combo.setCurrentIndex(0)
            return
        
        idx = self.combo.findData(value)
        if idx >= 0: 
            self.combo.setCurrentIndex(idx)

    @property
    def currentIndexChanged(self):
        return self.combo.currentIndexChanged

class SelectComboWidget(QComboBox, EditorWidgetMixin):
    def get_value(self):
        data = self.currentData()
        return None if data is None else data

    def set_value(self, value):
        if value is None:
            idx = self.findData(None)
            if idx >= 0:
                self.setCurrentIndex(idx)
            return

        if isinstance(value, str):
            value = value.strip()

        idx = self.findData(value)
        if idx >= 0:
            self.setCurrentIndex(idx)
            return

        # Fallback: normalized data match (legacy whitespace/case drift)
        if isinstance(value, str):
            upper_value = value.upper()
            for i in range(self.count()):
                data = self.itemData(i)
                if isinstance(data, str) and data.strip().upper() == upper_value:
                    self.setCurrentIndex(i)
                    return

        # Fallback: match by text if data not found (legacy cases)
        text = str(value)
        idx = self.findText(text)
        if idx >= 0:
            self.setCurrentIndex(idx)


def _is_query_mode_schema(schema: FieldSchema) -> bool:
    opts = schema.options or []
    if schema.key != 'str_param':
        return False
    if any(str(opt) == 'CARDS_MATCHING_FILTER' for opt in opts):
        return True
    label = str(getattr(schema, 'label', '') or '')
    return ('Query Mode' in label) or ('クエリ' in label)


def _format_select_option_label(schema: FieldSchema, opt) -> str:
    # Use CardTextResources for translation if available (e.g. DURATION_THIS_TURN)
    label = str(opt)
    if opt in CardTextResources.DURATION_TRANSLATION:
        return CardTextResources.get_duration_text(opt)
    if opt in CardTextResources.DELAYED_EFFECT_TRANSLATION:
        return CardTextResources.get_delayed_effect_text(opt)
    if opt in CardTextResources.KEYWORD_TRANSLATION:
        return CardTextResources.get_keyword_text(opt)

    if _is_query_mode_schema(schema):
        query_mode_labels = {
            'CARDS_MATCHING_FILTER': tr('条件一致カード枚数'),
            'SELECT_TARGET': tr('対象を選択'),
            'SELECT_OPTION': tr('選択肢を選択'),
        }
        if opt in query_mode_labels:
            return query_mode_labels[opt]
        # Query Mode の統計キーは日本語ラベルを使う
        if isinstance(opt, str) and opt in CardTextResources.STAT_KEY_MAP:
            return CardTextResources.get_stat_key_label(opt)

    return label

class CivilizationWrapper(CivilizationSelector, EditorWidgetMixin):
    def get_value(self):
        return self.get_selected_civs()

    def set_value(self, value):
        self.set_selected_civs(value)

class RacesEditorWidget(TextWidget):
    """Simple wrapper for comma-separated list editing."""
    def get_value(self):
        text = self.text()
        if not text: return []
        return [r.strip() for r in text.split(',') if r.strip()]

    def set_value(self, value):
        if isinstance(value, list):
            self.setText(tr(", ").join(value))
        else:
            self.setText(str(value))

class ConditionEditorWrapper(ConditionEditorWidget, EditorWidgetMixin):
    def get_value(self):
        return self.get_data()

    def set_value(self, value):
        self.set_data(value)

class WidgetFactory:
    """
    Factory for creating editor widgets.
    Uses a Registry Pattern to allow extension without modifying the class.
    """
    _REGISTRY = {}

    @classmethod
    def register(cls, field_type: FieldType, factory_func):
        """Register a factory function for a specific FieldType."""
        cls._REGISTRY[field_type] = factory_func

    @staticmethod
    def create_widget(parent, field_config, update_callback=None):
        """
        Creates a widget based on a configuration object.
        Supports both FieldSchema (new) and dict (legacy) for backward compatibility.
        """
        if isinstance(field_config, FieldSchema):
            return WidgetFactory._create_from_schema(parent, field_config, update_callback)
        elif isinstance(field_config, dict):
            return WidgetFactory._create_from_dict(parent, field_config, update_callback)
        return None

    @classmethod
    def _create_from_schema(cls, parent, field_schema: FieldSchema, update_callback):
        w_type = field_schema.field_type

        # Check registry first
        if w_type in cls._REGISTRY:
            return cls._REGISTRY[w_type](parent, field_schema, update_callback)

        # Fallback to internal dispatch (kept for robustness during migration)
        # In a full migration, these would all be registered.
        # We will register the core ones at module level.
        return None

    @staticmethod
    def _create_from_dict(parent, field_config: dict, update_callback):
        """Legacy creation method for backward compatibility."""
        # For legacy dicts, we might want to map string keys to FieldTypes and use registry,
        # but for now we keep the legacy logic separate or use a legacy registry.
        # This implementation simply replicates the original logic for safety.
        w_type = field_config.get('widget')
        widget = None

        if w_type == 'text':
            widget = TextWidget(parent)
            safe_connect(widget, 'textChanged', lambda: update_callback())

        elif w_type == 'spinbox':
            widget = NumberWidget(parent)
            safe_connect(widget, 'valueChanged', lambda: update_callback())

        elif w_type == 'checkbox':
            widget = BoolCheckWidget(field_config.get('label', ''), parent)
            safe_connect(widget, 'stateChanged', lambda: update_callback())

        elif w_type == 'player_scope':
            widget = PlayerScopeWidget(parent)
            safe_connect(widget.self_chk, 'stateChanged', lambda: update_callback())
            safe_connect(widget.opp_chk, 'stateChanged', lambda: update_callback())

        elif w_type == 'zone_combo':
            widget = ZoneCombo(parent)
            safe_connect(widget, 'currentIndexChanged', lambda: update_callback())

        elif w_type == 'scope_combo':
            widget = ScopeCombo(parent)
            safe_connect(widget, 'currentIndexChanged', lambda: update_callback())

        elif w_type == 'filter_editor':
            widget = FilterEditorWrapper(parent)
            safe_connect(widget, 'dataChanged', update_callback)

        elif w_type == 'variable_link':
            widget = VariableLinkWrapper(parent)
            safe_connect(widget, 'dataChanged', update_callback)

        elif w_type == 'options_control':
            cb = getattr(parent, 'request_generate_options', lambda: None)
            widget = OptionsControlWidget(parent, cb)

        elif w_type == 'ref_mode_combo':
            widget = RefModeComboWrapper(parent)
            safe_connect(widget, 'currentIndexChanged', lambda: update_callback())

        # Fallback for generic combo
        if widget is None and w_type and 'combo' in w_type:
            widget = SelectComboWidget(parent)
            safe_connect(widget, 'currentIndexChanged', lambda: update_callback())

        return widget

# --- Factory Functions ---

def _create_string_widget(parent, schema, cb):
    widget = TextWidget(parent)
    if schema.tooltip:
        widget.setPlaceholderText(schema.tooltip)
    safe_connect(widget, 'textChanged', lambda: cb())
    return widget

def _create_int_widget(parent, schema, cb):
    # widget_hint="amount_all" または min_value==-1 のとき「すべて」オプション付きウィジェットを使用
    # 再発防止: AmountWithAllWidget では -1 が「すべて」として表示される
    if schema.widget_hint == "amount_all" or schema.min_value == -1:
        widget = AmountWithAllWidget(parent, max_val=schema.max_value or 9999)
    else:
        widget = NumberWidget(parent, min_val=schema.min_value or 0, max_val=schema.max_value or 99999)
    # step が指定されている場合はスピンボックスの増減幅を設定する
    if getattr(schema, 'step', None):
        widget.setSingleStep(schema.step)
    safe_connect(widget, 'valueChanged', lambda: cb())
    return widget

def _create_quantity_widget(parent, schema, cb):
    widget = QuantityModeWidget(parent, max_val=schema.max_value or 9999)
    # The QuantityModeWidget emits currentIndexChanged for the mode, and valueChanged for the spin.
    # We can connect to both or add a unified signal in the widget.
    safe_connect(widget.mode_combo, 'currentIndexChanged', lambda: cb())
    safe_connect(widget.spin, 'valueChanged', lambda: cb())
    return widget

def _create_bool_widget(parent, schema, cb):
    widget = BoolCheckWidget(schema.label, parent)
    if schema.tooltip: widget.setToolTip(schema.tooltip)
    safe_connect(widget, 'stateChanged', lambda: cb())
    return widget

def _create_player_widget(parent, schema, cb):
    widget = PlayerScopeWidget(parent)
    safe_connect(widget.self_chk, 'stateChanged', lambda: cb())
    safe_connect(widget.opp_chk, 'stateChanged', lambda: cb())
    return widget

def _create_zone_widget(parent, schema, cb):
    widget = ZoneCombo(parent)
    safe_connect(widget, 'currentIndexChanged', lambda: cb())
    return widget

def _create_zone_list_widget(parent, schema, cb):
    widget = MultiZoneSelector(parent)
    for cb_widget in widget.checkboxes.values():
        safe_connect(cb_widget, 'stateChanged', lambda: cb())
    return widget

def _create_filter_widget(parent, schema, cb):
    widget = FilterEditorWrapper(parent)
    safe_connect(widget, 'filterChanged', cb)
    return widget

def _create_link_widget(parent, schema, cb):
    widget = VariableLinkWrapper(parent)
    safe_connect(widget, 'linkChanged', cb)
    return widget

def _create_options_control(parent, schema, cb):
    # Special callback needed from parent
    parent_cb = getattr(parent, 'request_generate_options', lambda: None)
    widget = OptionsControlWidget(parent, parent_cb)
    return widget

def _create_civ_widget(parent, schema, cb):
    widget = CivilizationWrapper(parent)
    safe_connect(widget, 'changed', lambda: cb())
    return widget

def _create_races_widget(parent, schema, cb):
    # Use a simple text-based editor for comma-separated races
    widget = RacesEditorWidget(parent)
    safe_connect(widget, 'textChanged', lambda: cb())
    return widget

def _create_type_select_widget(parent, schema, cb):
    widget = SelectComboWidget(parent)
    for t in CARD_TYPES:
        widget.addItem(t, t)
    safe_connect(widget, 'currentIndexChanged', lambda: cb())
    return widget

def _create_select_widget(parent, schema, cb):
    hint = schema.widget_hint
    widget = None
    if hint == 'ref_mode_combo':
        widget = RefModeComboWrapper(parent)
    elif hint == 'scope_combo':
        widget = ScopeCombo(parent)
    else:
        widget = SelectComboWidget(parent)
        # Add empty item if field has no default or is optional
        if schema.default is None:
            widget.addItem("---", None)
        
        if schema.options:
            for opt in schema.options:
                label = _format_select_option_label(schema, opt)
                widget.addItem(label, opt) # Display label, store value (opt)

    safe_connect(widget, 'currentIndexChanged', lambda: cb())
    return widget

def _create_enum_widget(parent, schema, cb):
    """Dynamically loads an Enum and populates a ComboBox."""
    widget = SelectComboWidget(parent)
    source = schema.enum_source
    if source:
        try:
            # Assuming format 'module.class'
            parts = source.split('.')
            cls_name = parts[-1]
            mod_name = '.'.join(parts[:-1])

            module = importlib.import_module(mod_name)
            enum_cls = getattr(module, cls_name)

            for member in enum_cls:
                widget.addItem(member.name, member.value) # or member.name depending on usage
        except Exception as e:
            print(f"Failed to load enum {source}: {e}")
            from dm_toolkit.gui.i18n import tr as _tr
            widget.addItem(_tr("ERROR"), None)

    safe_connect(widget, 'currentIndexChanged', lambda: cb())
    return widget

def _create_condition_widget(parent, schema, cb):
    widget = ConditionEditorWrapper(parent)
    safe_connect(widget, 'dataChanged', cb)
    return widget

# --- Register Core Types ---
WidgetFactory.register(FieldType.STRING, _create_string_widget)
WidgetFactory.register(FieldType.INT, _create_int_widget)
WidgetFactory.register(FieldType.BOOL, _create_bool_widget)
WidgetFactory.register(FieldType.PLAYER, _create_player_widget)
WidgetFactory.register(FieldType.ZONE, _create_zone_widget)
WidgetFactory.register(FieldType.FILTER, _create_filter_widget)
WidgetFactory.register(FieldType.LINK, _create_link_widget)
WidgetFactory.register(FieldType.OPTIONS_CONTROL, _create_options_control)
WidgetFactory.register(FieldType.CIVILIZATION, _create_civ_widget)
WidgetFactory.register(FieldType.RACES, _create_races_widget)
WidgetFactory.register(FieldType.TYPE_SELECT, _create_type_select_widget)
WidgetFactory.register(FieldType.SELECT, _create_select_widget)
WidgetFactory.register(FieldType.ENUM, _create_enum_widget)
WidgetFactory.register(FieldType.CONDITION, _create_condition_widget)
WidgetFactory.register(FieldType.CONDITION_TREE, _create_condition_widget)
WidgetFactory.register(FieldType.QUANTITY, _create_quantity_widget)
WidgetFactory.register(FieldType.ZONE_LIST, _create_zone_list_widget)
