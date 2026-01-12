# -*- coding: utf-8 -*-
from PyQt6.QtWidgets import (
    QWidget, QComboBox, QSpinBox, QLineEdit, QCheckBox,
    QGroupBox, QDoubleSpinBox, QLabel, QFormLayout
)
from PyQt6.QtCore import Qt, pyqtSignal
from contextlib import contextmanager

class BaseEditForm(QWidget):
    """
    Base class for all edit forms in the Card Editor.
    Implements the Template Method pattern for the load -> update -> save flow.
    """

    # Signal emitted when data is changed by the user
    dataChanged = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.current_item = None
        self._is_populating = False
        self.bindings = {} # key: widget
        self.input_widgets = set() # Set of widgets to block signals for

    @contextmanager
    def suppress_signals(self):
        """Context manager to block signals for all registered input widgets."""
        self.block_signals_all(True)
        try:
            yield
        finally:
            self.block_signals_all(False)

    def register_widget(self, widget, key=None):
        """
        Registers a widget for signal blocking and optional data binding.
        """
        self.input_widgets.add(widget)
        if key:
            self.bindings[key] = widget
        return widget

    def add_field(self, label, widget, key=None, layout=None):
        """
        Helper method to add a labeled field to a form layout.
        Also registers the widget for signal blocking and binding.
        """
        # Register the widget
        self.register_widget(widget, key)

        if layout is None:
            layout = getattr(self, 'form_layout', None)

        if layout is None and isinstance(self.layout(), QFormLayout):
            layout = self.layout()

        if layout is None:
            # If no layout found, we just return the label/widget but can't add to layout
            return None

        if isinstance(label, str):
            lbl = QLabel(label)
            layout.addRow(lbl, widget)
            return lbl
        elif isinstance(label, QWidget):
            layout.addRow(label, widget)
            return label
        else:
            # If label is None, just add the widget spanning
            layout.addRow(widget)
            return None

    # --- Template Methods ---

    def load_data(self, item):
        """
        Template method to load data from an item into the UI.
        Steps:
        1. Pre-load setup (block signals).
        2. Get data from item.
        3. _load_ui_from_data (Hook) - passed data AND item context.
        4. _update_ui_state (Hook).
        5. Post-load cleanup (unblock signals).
        """
        self.current_item = None # Prevent save_data from triggering during population
        self._is_populating = True

        try:
            self.block_signals_all(True)

            data = item.data(Qt.ItemDataRole.UserRole + 2)
            if data is None:
                data = {}

            self._load_ui_from_data(data, item)
            self._update_ui_state(data)

        finally:
            self.block_signals_all(False)
            self._is_populating = False
            self.current_item = item

    def save_data(self):
        """
        Template method to save data from UI back to the item.
        Steps:
        1. Validation check (current_item exists, not populating).
        2. Get existing data (or create new).
        3. _save_ui_to_data (Hook).
        4. Update item data and display text.
        5. Emit change signal.
        """
        if not self.current_item or self._is_populating:
            return

        data = self.current_item.data(Qt.ItemDataRole.UserRole + 2)
        if data is None:
            # Should not happen if item is valid, but handle gracefully
            data = {}

        self._save_ui_to_data(data)

        self.current_item.setData(data, Qt.ItemDataRole.UserRole + 2)
        self.current_item.setText(self._get_display_text(data))

        self.dataChanged.emit()

    # --- Aliases for Backward Compatibility ---

    def set_data(self, item):
        self.load_data(item)

    def update_data(self):
        self.save_data()

    # --- Hooks (Override these in subclasses) ---

    def _load_ui_from_data(self, data, item):
        """
        Hook to populate UI widgets from data.
        Override to implement custom loading logic.
        Uses bindings by default.
        """
        self._apply_bindings(data)

    def _save_ui_to_data(self, data):
        """
        Hook to save UI values back into the data dictionary.
        Override to implement custom saving logic.
        Uses bindings by default.
        """
        self._collect_bindings(data)

    def _update_ui_state(self, data):
        """
        Hook to update UI visibility or enabled state based on loaded data.
        Default implementation does nothing.
        """
        pass

    # --- Common Helpers ---

    def _apply_bindings(self, data):
        """
        Populate widgets from data using self.bindings.
        """
        for key, widget in self.bindings.items():
            val = data.get(key)

            # Handle default values if binding is (widget, default)
            if isinstance(widget, tuple):
                widget_obj, default_val = widget
                if val is None: val = default_val
            else:
                widget_obj = widget

            if val is None:
                 # Try to deduce default from widget type
                 if isinstance(widget_obj, (QSpinBox, QDoubleSpinBox)): val = 0
                 elif isinstance(widget_obj, QCheckBox): val = False
                 elif isinstance(widget_obj, QLineEdit): val = ""
                 elif isinstance(widget_obj, QComboBox): val = None
                 # Add default for custom widgets that implement set_data but aren't standard types
                 elif hasattr(widget_obj, 'set_data'): val = {}

            # Apply to widget
            if hasattr(widget_obj, 'set_data'):
                # Some widgets expect a dict for set_data, ensure it's not None if dict expected
                if val is None: val = {}
                # Guard against invalid data types (e.g. integer 0 being passed to a widget expecting dict)
                # This often happens if data.get(key) returns 0 (int) but the widget is a complex editor
                if not isinstance(val, dict) and not isinstance(widget_obj, (QSpinBox, QDoubleSpinBox, QComboBox, QLineEdit, QCheckBox)):
                     # If it's a primitive value but widget is complex, default to empty dict
                     if isinstance(val, (int, str, bool)) or val == 0:
                          val = {}
                widget_obj.set_data(val)
            elif isinstance(widget_obj, QComboBox):
                self.set_combo_by_data(widget_obj, val)
            elif isinstance(widget_obj, (QSpinBox, QDoubleSpinBox)):
                widget_obj.setValue(val)
            elif isinstance(widget_obj, QLineEdit):
                widget_obj.setText(str(val) if val is not None else "")
            elif isinstance(widget_obj, QCheckBox):
                widget_obj.setChecked(bool(val))

    def _collect_bindings(self, data):
        """
        Collect data from widgets using self.bindings.
        """
        for key, widget in self.bindings.items():
            widget_obj = widget
            if isinstance(widget, tuple):
                widget_obj = widget[0]

            if hasattr(widget_obj, 'get_data'):
                # Handle widgets with get_data()
                # Special check for VariableLinkWidget which acts in-place
                if widget_obj.__class__.__name__ == 'VariableLinkWidget':
                     widget_obj.get_data(data)
                else:
                     data[key] = widget_obj.get_data()

            elif isinstance(widget_obj, QComboBox):
                collected_data = widget_obj.currentData()
                data[key] = collected_data
            elif isinstance(widget_obj, (QSpinBox, QDoubleSpinBox)):
                data[key] = widget_obj.value()
            elif isinstance(widget_obj, QLineEdit):
                data[key] = widget_obj.text()
            elif isinstance(widget_obj, QCheckBox):
                data[key] = widget_obj.isChecked()

    def _get_display_text(self, data):
        """
        Override to return the text to display in the tree.
        """
        return str(data.get('id', ''))

    def block_signals_all(self, block):
        """
        Override to block signals for all input widgets.
        Default implementation blocks signals for widgets in bindings and registered input_widgets.
        """
        for widget in self.input_widgets:
            w = widget[0] if isinstance(widget, tuple) else widget
            if hasattr(w, 'blockSignals'):
                w.blockSignals(block)

    def populate_combo(self, combo: QComboBox, items: list, data_func=None, display_func=None, clear=True):
        """
        Populates a QComboBox with items.
        """
        if clear:
            combo.clear()

        for item in items:
            if isinstance(item, tuple):
                label, user_data = item
                if display_func:
                    label = display_func(label)
                combo.addItem(str(label), user_data)
            else:
                label = str(item)
                if display_func:
                    label = display_func(item)

                user_data = item if data_func is None else data_func(item)
                combo.addItem(label, user_data)

    def set_combo_by_data(self, combo: QComboBox, value):
        """
        Sets QComboBox selection by item data.
        """
        idx = combo.findData(value)
        if idx >= 0:
            combo.setCurrentIndex(idx)
