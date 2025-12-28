# -*- coding: utf-8 -*-
from PyQt6.QtWidgets import (
    QWidget, QComboBox, QSpinBox, QLineEdit, QCheckBox,
    QGroupBox, QDoubleSpinBox, QLabel, QFormLayout
)
from PyQt6.QtCore import Qt, pyqtSignal

class BaseEditForm(QWidget):
    """
    Base class for all edit forms in the Card Editor.
    Handles common logic like data binding, signal blocking, and UI population.
    """

    # Signal emitted when data is changed by the user
    dataChanged = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.current_item = None
        self._is_populating = False
        self.bindings = {} # key: widget

    def add_field(self, label, widget, layout=None):
        """
        Helper method to add a labeled field to a form layout.

        Args:
            label (str or QWidget): The label text or widget.
            widget (QWidget): The input widget.
            layout (QFormLayout, optional): The layout to add to.
                                            Defaults to self.form_layout if available.

        Returns:
            QLabel or QWidget: The label widget created or passed.
        """
        if layout is None:
            layout = getattr(self, 'form_layout', None)

        if layout is None and isinstance(self.layout(), QFormLayout):
            layout = self.layout()

        if layout is None:
            # If no layout found, we can't add.
            # In a strict scenario, we might raise an error, but here we'll just return.
            return None

        if isinstance(label, str):
            lbl = QLabel(label)
            lbl.setBuddy(widget)
            layout.addRow(lbl, widget)
            return lbl
        elif isinstance(label, QWidget):
            layout.addRow(label, widget)
            return label
        else:
            # If label is None, just add the widget spanning
            layout.addRow(widget)
            return None

    def set_data(self, item):
        """
        Sets the current item to edit and populates the UI.
        """
        self.current_item = None # Prevent update_data from triggering during population
        self._is_populating = True

        try:
            self.block_signals_all(True)
            self._populate_ui(item)
        finally:
            self.block_signals_all(False)
            self._is_populating = False
            self.current_item = item

    def update_data(self):
        """
        Updates the underlying data model from UI values.
        Should be connected to widget signals (valueChanged, textChanged, etc).
        """
        if not self.current_item or self._is_populating:
            return

        data = self.current_item.data(Qt.ItemDataRole.UserRole + 2)
        if data is None:
            # Should not happen if item is valid
            return

        self._save_data(data)

        self.current_item.setData(data, Qt.ItemDataRole.UserRole + 2)
        self.current_item.setText(self._get_display_text(data))

        self.dataChanged.emit()

    def _populate_ui(self, item):
        """
        Override this to populate UI widgets from item data.
        """
        # Default implementation uses bindings
        data = item.data(Qt.ItemDataRole.UserRole + 2)
        self._apply_bindings(data)

    def _save_data(self, data):
        """
        Override this to save UI values back into the data dictionary.
        """
        # Default implementation uses bindings
        self._collect_bindings(data)

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
                # If val is None, we might want a default based on widget type,
                # but usually data.get(key) returning None means key missing.

            if val is None:
                 # Try to deduce default from widget type
                 if isinstance(widget_obj, (QSpinBox, QDoubleSpinBox)): val = 0
                 elif isinstance(widget_obj, QCheckBox): val = False
                 elif isinstance(widget_obj, QLineEdit): val = ""
                 elif isinstance(widget_obj, QComboBox): val = None # Combo handling handles None usually

            # Apply to widget
            if hasattr(widget_obj, 'set_data'):
                widget_obj.set_data(val if val is not None else {})
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
                # Special handling for VariableLinkWidget which updates data in-place usually,
                # but assuming get_data returns the value or updates the dict
                # The existing VariableLinkWidget.get_data(data) updates in place.
                # FilterWidget.get_data() returns a dict.
                # We need to distinguish based on signature or convention.
                # In ActionForm: filter_widget.get_data() returns dict. link_widget.get_data(data) modifies data.
                # This is inconsistent. We should standardize or handle exception.

                # Hacky check for now, or just assume assignment for those returning value
                # Check method signature? No.

                # Let's look at FilterEditorWidget. It returns a dict.
                # VariableLinkWidget updates passed data.

                # Ideally, we standardize. But for now:
                if widget_obj.__class__.__name__ == 'VariableLinkWidget':
                     widget_obj.get_data(data) # Updates in place, returns nothing
                else:
                     data[key] = widget_obj.get_data()

            elif isinstance(widget_obj, QComboBox):
                data[key] = widget_obj.currentData()
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
        Default implementation blocks signals for widgets in bindings.
        """
        for widget in self.bindings.values():
            w = widget[0] if isinstance(widget, tuple) else widget
            w.blockSignals(block)

    # Helper methods
    def populate_combo(self, combo: QComboBox, items: list, data_func=None, display_func=None, clear=True):
        """
        Populates a QComboBox with items.
        items: List of strings or tuples (label, data).
        data_func: Optional function to extract data from an item if needed.
        display_func: Optional function to format the display label.
        clear: Whether to clear existing items (default True).
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
                # If just string, use it as both label and data
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
        else:
            # Optional: Select first or nothing?
            pass
