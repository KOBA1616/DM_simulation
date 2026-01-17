# -*- coding: utf-8 -*-
from PyQt6.QtWidgets import (
    QWidget, QCheckBox, QHBoxLayout, QVBoxLayout, QPushButton, QLabel, QLineEdit, QComboBox
)
from dm_toolkit.gui.editor.widgets.common import (
    ZoneCombo, ScopeCombo, TextWidget, NumberWidget, BoolCheckWidget, EditorWidgetMixin
)
from dm_toolkit.gui.editor.forms.unified_widgets import (
    make_player_scope_selector, make_measure_mode_combo, make_ref_mode_combo
)
from dm_toolkit.gui.editor.schema_def import FieldType, FieldSchema
import importlib

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
        self.layout = QHBoxLayout(self)
        self.layout.setContentsMargins(0,0,0,0)

        self.spin = NumberWidget(self, 1, 10)
        self.btn = QPushButton("Generate Options")
        self.btn.clicked.connect(callback)
        self.label = QLabel("Options Count")

        self.layout.addWidget(self.label)
        self.layout.addWidget(self.spin)
        self.layout.addWidget(self.btn)

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

        idx = self.findData(value)
        if idx >= 0:
            self.setCurrentIndex(idx)
            return

        # Fallback: match by text if data not found (legacy cases)
        text = str(value)
        idx = self.findText(text)
        if idx >= 0:
            self.setCurrentIndex(idx)

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
            self.setText(", ".join(value))
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
            widget.textChanged.connect(lambda: update_callback())

        elif w_type == 'spinbox':
            widget = NumberWidget(parent)
            widget.valueChanged.connect(lambda: update_callback())

        elif w_type == 'checkbox':
            widget = BoolCheckWidget(field_config.get('label', ''), parent)
            widget.stateChanged.connect(lambda: update_callback())

        elif w_type == 'player_scope':
            widget = PlayerScopeWidget(parent)
            widget.self_chk.stateChanged.connect(lambda: update_callback())
            widget.opp_chk.stateChanged.connect(lambda: update_callback())

        elif w_type == 'zone_combo':
            widget = ZoneCombo(parent)
            widget.currentIndexChanged.connect(lambda: update_callback())

        elif w_type == 'scope_combo':
            widget = ScopeCombo(parent)
            widget.currentIndexChanged.connect(lambda: update_callback())

        elif w_type == 'filter_editor':
            widget = FilterEditorWrapper(parent)
            widget.dataChanged.connect(update_callback)

        elif w_type == 'variable_link':
            widget = VariableLinkWrapper(parent)
            widget.dataChanged.connect(update_callback)

        elif w_type == 'options_control':
            cb = getattr(parent, 'request_generate_options', lambda: None)
            widget = OptionsControlWidget(parent, cb)

           elif w_type == 'ref_mode_combo':
               widget = RefModeComboWrapper(parent)
               widget.currentIndexChanged.connect(lambda: update_callback())

        # Fallback for generic combo
        if widget is None and w_type and 'combo' in w_type:
              widget = SelectComboWidget(parent)
            widget.currentIndexChanged.connect(lambda: update_callback())

        return widget

# --- Factory Functions ---

def _create_string_widget(parent, schema, cb):
    widget = TextWidget(parent)
    if schema.tooltip:
        widget.setPlaceholderText(schema.tooltip)
    widget.textChanged.connect(lambda: cb())
    return widget

def _create_int_widget(parent, schema, cb):
    widget = NumberWidget(parent, min_val=schema.min_value or 0, max_val=schema.max_value or 99999)
    widget.valueChanged.connect(lambda: cb())
    return widget

def _create_bool_widget(parent, schema, cb):
    widget = BoolCheckWidget(schema.label, parent)
    if schema.tooltip: widget.setToolTip(schema.tooltip)
    widget.stateChanged.connect(lambda: cb())
    return widget

def _create_player_widget(parent, schema, cb):
    widget = PlayerScopeWidget(parent)
    widget.self_chk.stateChanged.connect(lambda: cb())
    widget.opp_chk.stateChanged.connect(lambda: cb())
    return widget

def _create_zone_widget(parent, schema, cb):
    widget = ZoneCombo(parent)
    widget.currentIndexChanged.connect(lambda: cb())
    return widget

def _create_filter_widget(parent, schema, cb):
    widget = FilterEditorWrapper(parent)
    widget.filterChanged.connect(cb)
    return widget

def _create_link_widget(parent, schema, cb):
    widget = VariableLinkWrapper(parent)
    widget.linkChanged.connect(cb)
    return widget

def _create_options_control(parent, schema, cb):
    # Special callback needed from parent
    parent_cb = getattr(parent, 'request_generate_options', lambda: None)
    widget = OptionsControlWidget(parent, parent_cb)
    return widget

def _create_civ_widget(parent, schema, cb):
    widget = CivilizationWrapper(parent)
    widget.changed.connect(lambda: cb())
    return widget

def _create_enum_widget(parent, schema, cb):
    widget = RacesEditorWidget(parent)
    widget = SelectComboWidget(parent)
    return widget

def _create_type_select_widget(parent, schema, cb):
    widget = QComboBox(parent)
    for t in CARD_TYPES:
        widget.addItem(t, t)
    widget.currentIndexChanged.connect(lambda: cb())
    return widget

def _create_select_widget(parent, schema, cb):

            if schema.default is None:
                widget.addItem("---", None)
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
                # Use CardTextResources for translation if available (e.g. DURATION_THIS_TURN)
                label = str(opt)
                if opt in CardTextResources.DURATION_TRANSLATION:
                     label = CardTextResources.get_duration_text(opt)
                elif opt in CardTextResources.KEYWORD_TRANSLATION: # Also handle keywords if passed as raw list
                     label = CardTextResources.get_keyword_text(opt)

                widget.addItem(label, opt) # Display label, store value (opt)

    widget.currentIndexChanged.connect(lambda: cb())
    return widget

def _create_enum_widget(parent, schema, cb):
    """Dynamically loads an Enum and populates a ComboBox."""
    widget = QComboBox(parent)
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
            widget.addItem("ERROR", None)

    widget.currentIndexChanged.connect(lambda: cb())
    return widget

def _create_condition_widget(parent, schema, cb):
    widget = ConditionEditorWrapper(parent)
    widget.dataChanged.connect(cb)
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
