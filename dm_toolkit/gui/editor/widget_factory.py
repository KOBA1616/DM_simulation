# -*- coding: utf-8 -*-
from PyQt6.QtWidgets import (
    QWidget, QCheckBox, QHBoxLayout, QVBoxLayout, QPushButton, QLabel
)
from dm_toolkit.gui.editor.widgets.common import (
    ZoneCombo, ScopeCombo, TextWidget, NumberWidget, BoolCheckWidget, EditorWidgetMixin
)
from dm_toolkit.gui.editor.forms.unified_widgets import (
    make_player_scope_selector, make_measure_mode_combo, make_ref_mode_combo
)
# from dm_toolkit.gui.editor.widgets.interfaces import EditorWidgetInterface

# Import correct widgets from actual file structure
from dm_toolkit.gui.editor.forms.parts.filter_widget import FilterEditorWidget
from dm_toolkit.gui.editor.forms.parts.variable_link_widget import VariableLinkWidget

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
        # VariableLinkWidget usually writes directly to a dict passed in get_data
        # We need to adapt this.
        d = {}
        self.get_data(d)
        return d # This might need special handling in the Form if it expects flattened keys

    def set_value(self, value):
        # Value is the full command data dict usually
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

class WidgetFactory:
    @staticmethod
    def create_widget(parent, field_config, update_callback=None):
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
            # VariableLinkWidget constructor might vary, assuming standard parent
            widget = VariableLinkWrapper(parent)
            widget.dataChanged.connect(update_callback)

        elif w_type == 'options_control':
            # Special case, needs callback
            # We assume the parent (Form) has a method request_generate_options
            cb = getattr(parent, 'request_generate_options', lambda: None)
            widget = OptionsControlWidget(parent, cb)

        # Fallback for others (query_mode_combo, etc. using standard ComboBox)
        if widget is None and 'combo' in w_type:
            # Generic combo handling if specific class not found
            from PyQt6.QtWidgets import QComboBox
            widget = QComboBox(parent)
            # Populate based on type? Needs more context usually provided in Form
            widget.currentIndexChanged.connect(lambda: update_callback())

        return widget
