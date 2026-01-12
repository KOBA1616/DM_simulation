import sys
import os
from unittest.mock import MagicMock

# Add project root to sys.path
sys.path.append(os.getcwd())

# Mock pydantic
sys.modules["pydantic"] = MagicMock()
class MockBaseModel:
    pass
sys.modules["pydantic"].BaseModel = MockBaseModel

# Mock PyQt6 modules
mock_qt_widgets = MagicMock()
mock_qt_gui = MagicMock()
mock_qt_core = MagicMock()

sys.modules["PyQt6"] = MagicMock()
sys.modules["PyQt6.QtWidgets"] = mock_qt_widgets
sys.modules["PyQt6.QtGui"] = mock_qt_gui
sys.modules["PyQt6.QtCore"] = mock_qt_core

# Define dummy base classes for inheritance
# They must be classes, not mocks, to avoid metaclass conflicts with Mixins
class MockQWidget:
    def __init__(self, *args, **kwargs): pass
    def setMinimumWidth(self, *args): pass
    def setSizePolicy(self, *args): pass
    def setLayout(self, *args): pass

class MockQMainWindow(MockQWidget):
    def setWindowTitle(self, *args): pass
    def resize(self, *args): pass
    def addToolBar(self, *args): pass
    def setCentralWidget(self, *args): pass
    def statusBar(self): return MagicMock()

class MockQTreeView(MockQWidget):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.customContextMenuRequested = MagicMock()

    def setModel(self, *args): pass
    def setHeaderHidden(self, *args): pass
    def setSelectionMode(self, *args): pass
    def setEditTriggers(self, *args): pass
    def setDragEnabled(self, *args): pass
    def setAcceptDrops(self, *args): pass
    def setDropIndicatorShown(self, *args): pass
    def setDragDropMode(self, *args): pass
    def setContextMenuPolicy(self, *args): pass
    def selectionModel(self): return MagicMock()

class MockQComboBox(MockQWidget):
    def setEditable(self, *args): pass
    def addItems(self, *args): pass
    def currentText(self): return ""
    def setCurrentText(self, *args): pass
    def clear(self): pass

class MockQSplitter(MockQWidget):
    def addWidget(self, *args): pass
    def setStretchFactor(self, *args): pass

class MockQToolBar(MockQWidget):
    def setIconSize(self, *args): pass
    def setStyleSheet(self, *args): pass
    def addAction(self, *args): pass
    def addWidget(self, *args): pass

# Assign classes
mock_qt_widgets.QMainWindow = MockQMainWindow
mock_qt_widgets.QTreeView = MockQTreeView
mock_qt_widgets.QSplitter = MockQSplitter
mock_qt_widgets.QWidget = MockQWidget
mock_qt_widgets.QToolBar = MockQToolBar
mock_qt_widgets.QComboBox = MockQComboBox
mock_qt_widgets.QMessageBox = MagicMock()
mock_qt_widgets.QFileDialog = MagicMock()
mock_qt_widgets.QSizePolicy = MagicMock()
mock_qt_widgets.QMenu = MagicMock()

# Explicitly mock QAction as a class if subclassed, but typically it's just instantiated
class MockQAction:
    def __init__(self, *args, **kwargs):
        self.triggered = MagicMock()
    def setShortcut(self, *args): pass
    def setStatusTip(self, *args): pass
    def setText(self, *args): pass
    def setEnabled(self, *args): pass

mock_qt_gui.QAction = MockQAction
mock_qt_widgets.QAction = MockQAction # For safety

class MockQStandardItemModel:
    def __init__(self, *args, **kwargs): pass
    def invisibleRootItem(self): return MagicMock()
    def appendRow(self, *args): pass
    def clear(self): pass
    def setHorizontalHeaderLabels(self, *args): pass

class MockQStandardItem:
    def __init__(self, *args, **kwargs): pass
    def setData(self, *args): pass

mock_qt_gui.QStandardItemModel = MockQStandardItemModel
mock_qt_gui.QStandardItem = MockQStandardItem
mock_qt_gui.QKeySequence = MagicMock()

# Mock Qt constants
mock_qt_core.Qt.Orientation.Horizontal = 1
mock_qt_core.Qt.ItemDataRole.UserRole = 256
mock_qt_core.Qt.ContextMenuPolicy.CustomContextMenu = 1
mock_qt_core.Qt.Key.Key_Delete = 1

# Mock pyqtSignal
mock_qt_core.pyqtSignal = MagicMock(return_value=MagicMock())
mock_qt_core.QSize = MagicMock()

# Helper for layouts which are sometimes subclassed or used as mixins? No, usually not.
# But widgets/common.py might have other widgets.
class MockQLineEdit(MockQWidget): pass
class MockQSpinBox(MockQWidget): pass
class MockQCheckBox(MockQWidget): pass
class MockQLabel(MockQWidget): pass
class MockQGroupBox(MockQWidget): pass
class MockQScrollArea(MockQWidget): pass
class MockQLayout:
    def __init__(self, *args, **kwargs): pass
    def addWidget(self, *args): pass
    def addLayout(self, *args): pass
    def addStretch(self, *args): pass

mock_qt_widgets.QLineEdit = MockQLineEdit
mock_qt_widgets.QSpinBox = MockQSpinBox
mock_qt_widgets.QCheckBox = MockQCheckBox
mock_qt_widgets.QLabel = MockQLabel
mock_qt_widgets.QGroupBox = MockQGroupBox
mock_qt_widgets.QScrollArea = MockQScrollArea
mock_qt_widgets.QVBoxLayout = MockQLayout
mock_qt_widgets.QHBoxLayout = MockQLayout
mock_qt_widgets.QGridLayout = MockQLayout
mock_qt_widgets.QFormLayout = MockQLayout

# Run imports
print("Starting imports...")
try:
    from dm_toolkit.gui.editor import window
    print("Imported window")
    from dm_toolkit.gui.editor import context_menus
    print("Imported context_menus")
    from dm_toolkit.gui.editor import logic_tree
    print("Imported logic_tree")

    # Instantiate LogicTreeWidget
    lt = logic_tree.LogicTreeWidget()
    print("Instantiated LogicTreeWidget")

    # Check context menu handler
    handler = lt.context_menu_handler
    if isinstance(handler, context_menus.LogicTreeContextMenuHandler):
        print("Handler is LogicTreeContextMenuHandler")
    else:
        print("ERROR: Handler mismatch")
        sys.exit(1)

    print("Verification successful.")

except ImportError as e:
    print(f"ImportError: {e}")
    sys.exit(1)
except Exception as e:
    print(f"An error occurred: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
