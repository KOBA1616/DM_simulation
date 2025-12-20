# -*- coding: cp932 -*-
from PyQt6.QtWidgets import QWidget, QComboBox
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
        raise NotImplementedError

    def _save_data(self, data):
        """
        Override this to save UI values back into the data dictionary.
        """
        raise NotImplementedError

    def _get_display_text(self, data):
        """
        Override to return the text to display in the tree.
        """
        return str(data.get('id', ''))

    def block_signals_all(self, block):
        """
        Override to block signals for all input widgets.
        Alternatively, we can rely on _is_populating flag in update_data,
        but explicit blocking is often safer for complex widgets.
        """
        pass

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
