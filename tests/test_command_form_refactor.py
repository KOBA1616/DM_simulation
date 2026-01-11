import sys
import os
import unittest
from unittest.mock import MagicMock

# Mock PyQt6 modules before they are imported by the code under test
sys.modules["PyQt6"] = MagicMock()
sys.modules["PyQt6.QtWidgets"] = MagicMock()
sys.modules["PyQt6.QtCore"] = MagicMock()
sys.modules["PyQt6.QtGui"] = MagicMock()

# Define dummy classes to satisfy inheritance
class QWidget:
    def __init__(self, parent=None): pass
    def layout(self): return MagicMock()
    def setLayout(self, layout): pass
    def deleteLater(self): pass

class QFormLayout:
    def __init__(self, parent=None): pass
    def addRow(self, *args): pass
    def insertRow(self, *args): pass
    def setContentsMargins(self, *args): pass
    def count(self): return 0
    def takeAt(self, index): return MagicMock()

class QComboBox(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.currentIndexChanged = MagicMock()
        self.currentData = MagicMock(return_value="DRAW_CARD")
        self.addItem = MagicMock()
        self.setCurrentIndex = MagicMock()

class QSpinBox(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.valueChanged = MagicMock()
        self.setValue = MagicMock()
        self.value = MagicMock(return_value=0)
        self.setRange = MagicMock()  # Added setRange

class QCheckBox(QWidget):
    def __init__(self, text="", parent=None):
        super().__init__(parent)
        self.stateChanged = MagicMock()
        self.setChecked = MagicMock()
        self.isChecked = MagicMock(return_value=False)

class QLabel(QWidget):
    def __init__(self, text="", parent=None):
        super().__init__(parent)
        self.setText = MagicMock()
        self.setVisible = MagicMock()
        self.setStyleSheet = MagicMock()

class QPushButton(QWidget):
    def __init__(self, text="", parent=None):
        super().__init__(parent)
        self.clicked = MagicMock()
        self.setVisible = MagicMock()

class QLineEdit(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.textChanged = MagicMock()
        self.setText = MagicMock()
        self.text = MagicMock(return_value="")

# Mock implementations
sys.modules["PyQt6.QtWidgets"].QWidget = QWidget
sys.modules["PyQt6.QtWidgets"].QFormLayout = QFormLayout
sys.modules["PyQt6.QtWidgets"].QComboBox = QComboBox
sys.modules["PyQt6.QtWidgets"].QSpinBox = QSpinBox
sys.modules["PyQt6.QtWidgets"].QCheckBox = QCheckBox
sys.modules["PyQt6.QtWidgets"].QLabel = QLabel
sys.modules["PyQt6.QtWidgets"].QPushButton = QPushButton
sys.modules["PyQt6.QtWidgets"].QLineEdit = QLineEdit
sys.modules["PyQt6.QtWidgets"].QVBoxLayout = MagicMock()
sys.modules["PyQt6.QtWidgets"].QHBoxLayout = MagicMock()
sys.modules["PyQt6.QtWidgets"].QStackedWidget = MagicMock()
sys.modules["PyQt6.QtWidgets"].QGroupBox = MagicMock()

# Setup paths
sys.path.append(os.getcwd())

# Import code under test
from dm_toolkit.gui.editor.forms.command_form import CommandEditForm
from dm_toolkit.gui.editor.forms.unified_action_form import UnifiedActionForm
from dm_toolkit.gui.editor.widget_factory import WidgetFactory

class TestCommandFormRefactor(unittest.TestCase):
    def test_instantiation(self):
        """Test that CommandEditForm can be instantiated."""
        form = CommandEditForm()
        self.assertIsInstance(form, CommandEditForm)
        self.assertIsInstance(form, UnifiedActionForm)
        print("CommandEditForm instantiated successfully.")

    def test_widget_creation(self):
        """Test that WidgetFactory creates widgets."""
        mock_callback = MagicMock()

        # Test text widget
        w = WidgetFactory.create_widget(None, {"widget": "text", "key": "test"}, mock_callback)
        self.assertIsInstance(w, QLineEdit)

        # Test spinbox
        w = WidgetFactory.create_widget(None, {"widget": "spinbox", "key": "test"}, mock_callback)
        self.assertIsInstance(w, QSpinBox)

        print("WidgetFactory created widgets successfully.")

    def test_schema_loading(self):
        """Test that UnifiedActionForm loads schema."""
        form = UnifiedActionForm()
        # Verify schema is loaded (COMMAND_SCHEMA should be populated)
        from dm_toolkit.gui.editor.forms.unified_action_form import COMMAND_SCHEMA
        self.assertIn("DRAW_CARD", COMMAND_SCHEMA)
        print("Schema loaded successfully.")

if __name__ == "__main__":
    unittest.main()
