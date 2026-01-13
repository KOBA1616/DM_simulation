#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified Pytest Runner with PyQt Stubbing for Headless Environments.
This script sets up the environment to mock PyQt/PySide modules before running pytest,
allowing GUI tests to run in headless CI environments without requiring a display or Qt libraries.
"""
import sys
import os
import unittest.mock
import types
import importlib.abc
import importlib.machinery

class StubLoader(importlib.abc.Loader):
    def __init__(self, module):
        self.module = module

    def create_module(self, spec):
        return self.module

    def exec_module(self, module):
        # Module is already populated.
        # Ensure it matches what we expect
        if hasattr(self.module, '__spec__'):
             module.__spec__ = self.module.__spec__

class StubFinder(importlib.abc.MetaPathFinder):
    def __init__(self, mocks):
        self.mocks = mocks

    def find_spec(self, fullname, path, target=None):
        if fullname in self.mocks:
            spec = importlib.machinery.ModuleSpec(fullname, StubLoader(self.mocks[fullname]))
            # CRITICAL: If we set has_location=True, it might look for files.
            # If False, it behaves like a namespace or builtin.
            spec.has_location = False
            return spec
        return None

def setup_gui_stubs():
    """
    Injects mocks for PyQt6 and PySide6 modules into sys.modules.
    This prevents ImportErrors when tests try to import GUI components in a headless environment.
    """
    # Check if stubs are already properly set up (e.g., by conftest.py)
    qtwidgets_mod = sys.modules.get('PyQt6.QtWidgets')
    if qtwidgets_mod and hasattr(qtwidgets_mod, 'QMainWindow'):
        print(" [STUB] GUI stubs already configured (likely by conftest.py).")
        return  # Already set up, don't override

    # Create a functional Signal class that actually calls connected slots
    class MockSignal:
        def __init__(self):
            self._slots = []
        
        def connect(self, slot):
            self._slots.append(slot)
            return None
        
        def disconnect(self, slot=None):
            if slot is None:
                self._slots.clear()
            elif slot in self._slots:
                self._slots.remove(slot)
            return None
        
        def emit(self, *args, **kwargs):
            for slot in self._slots:
                slot(*args, **kwargs)
            return None

    # Create dummy classes for inheritance
    class DummyQWidget(object):
        def __init__(self, *args, **kwargs):
            # Add common signals as functional MockSignals
            self.clicked = MockSignal()
            self.textChanged = MockSignal()
            self.stateChanged = MockSignal()
            self.currentIndexChanged = MockSignal()
            self._items = []
            
        def setWindowTitle(self, title): pass
        def setLayout(self, layout): pass
        def setGeometry(self, *args): pass
        def resize(self, *args): pass
        def setObjectName(self, name): pass
        def setSelectionMode(self, mode): pass
        def scrollToBottom(self): pass
        def addAction(self, action): pass
        def setAllowedAreas(self, areas): pass
        def setWidget(self, widget): pass
        def setWidgetResizable(self, resizable): pass
        def setVerticalScrollBarPolicy(self, policy): pass
        def setHorizontalScrollBarPolicy(self, policy): pass
        def setHorizontalHeaderLabels(self, labels): pass
        def setHeaderLabels(self, labels): pass
        def header(self): return unittest.mock.MagicMock()
        def horizontalHeader(self): return unittest.mock.MagicMock()
        def setDragDropMode(self, mode): pass
        def model(self): return unittest.mock.MagicMock()
        def font(self): return unittest.mock.MagicMock()
        def setFont(self, font): pass
        def show(self): pass
        def showMaximized(self): pass
        def hide(self): pass
        def close(self): return True
        def addWidget(self, widget, *args, **kwargs): pass  # Accept extra args for Grid Layout
        def addLayout(self, layout, *args, **kwargs): pass  # Accept extra args for Grid Layout
        def setRowStretch(self, row, stretch): pass
        def setColumnStretch(self, col, stretch): pass
        def setColumnCount(self, count): pass
        def setRowCount(self, count): pass
        def setItem(self, row, col, item): pass
        def setColumnWidth(self, col, width): pass
        def setStretch(self, index, stretch): pass
        def addTab(self, widget, label): pass
        def setAlignment(self, alignment): pass
        def setScene(self, scene): pass
        def setDragMode(self, mode): pass
        def setTransformationAnchor(self, anchor): pass
        def setResizeAnchor(self, anchor): pass
        def setRenderHint(self, hint): pass
        def setWordWrap(self, wrap): pass
        def setRange(self, min_val, max_val): pass
        def setValue(self, value): self._value = value
        def setFormat(self, format): pass
        def setText(self, text): pass
        def text(self): return ""
        def setCheckState(self, state): pass
        def addItem(self, *args): self._items.append(args)
        def setCurrentIndex(self, index): pass
        def currentIndex(self): return 0
        def blockSignals(self, block): return False
        def count(self): return len(self._items)
        def setItemData(self, index, data, role=None): pass
        def itemData(self, index, role=None): return None
        def addButton(self, button, id=-1): pass
        def checkedId(self): return -1
        def id(self, button): return -1
        def setContentsMargins(self, *args): pass
        def setSpacing(self, spacing): pass
        def addStretch(self, *args): pass
        def setStyleSheet(self, style): pass
        def setMinimumWidth(self, width): pass
        def setFixedWidth(self, width): pass
        def setMinimumHeight(self, height): pass
        def setMaximumWidth(self, width): pass
        def setMaximumHeight(self, height): pass
        def setFixedHeight(self, height): pass
        def setFixedSize(self, w, h): pass
        def setShortcut(self, shortcut): pass
        def clear(self): self._items = []
        def setToolTip(self, text): pass
        def setCursor(self, cursor): pass
        def setEnabled(self, enabled): pass
        def isEnabled(self): return True
        def setReadOnly(self, ro): pass
        def setVisible(self, visible): pass
        def isVisible(self): return True
        def setExclusive(self, exclusive): pass
        def insertItem(self, index, text): pass
        def setFlat(self, flat): pass
        def setSpacing(self, spacing): pass
        def setContentsMargins(self, *args): pass

    class DummyQMainWindow(DummyQWidget):
        def setCentralWidget(self, widget): pass
        def setMenuBar(self, menu): pass
        def addDockWidget(self, area, dock): pass
        def addToolBar(self, toolbar): pass
        def splitDockWidget(self, *args): pass
        def setStatusBar(self, bar): pass

    class DummyQDialog(DummyQWidget):
        def exec(self): return 1
        def accept(self): pass
        def reject(self): pass

    class DummyQApplication:
        def __init__(self, args): pass
        def exec(self): return 0
        @staticmethod
        def instance(): return None

    # Enums
    class DummyQt:
        class ItemDataRole:
            DisplayRole = 0
            UserRole = 256
            ForegroundRole = 9
            BackgroundRole = 8
            EditRole = 2

        class AlignmentFlag:
            AlignCenter = 0x0084
            AlignLeft = 0x0001
            AlignRight = 0x0002
            AlignTop = 0x0020
            AlignBottom = 0x0040
            AlignVCenter = 0x0080

        class WindowType:
            Window = 0x00000001

        class MatchFlag:
            MatchContains = 1
            MatchFixedString = 8

        class CursorShape:
            PointingHandCursor = 13

        class DockWidgetArea:
             LeftDockWidgetArea = 1
             RightDockWidgetArea = 2
             AllDockWidgetAreas = 15

        class ScrollBarPolicy:
            ScrollBarAlwaysOff = 1
            ScrollBarAlwaysOn = 2
            ScrollBarAsNeeded = 0

        class Orientation:
            Horizontal = 1
            Vertical = 2

        class GlobalColor:
            red = 1

        SolidPattern = 1
        Horizontal = 1
        Vertical = 2
        Checked = 2
        Unchecked = 0

    # Helper to create mock module
    def create_mock_module(name, is_package=False):
        m = types.ModuleType(name)
        if is_package:
            m.__path__ = []
        return m

    # Prepare mock modules map
    mocks = {}

    # 1. Mock the top-level packages
    pyqt6 = create_mock_module('PyQt6', is_package=True)
    mocks['PyQt6'] = pyqt6

    pyside6 = create_mock_module('PySide6', is_package=True)
    mocks['PySide6'] = pyside6

    # 2. Mock submodules
    # QtWidgets
    qt_widgets = create_mock_module('PyQt6.QtWidgets')
    mocks['PyQt6.QtWidgets'] = qt_widgets
    pyqt6.QtWidgets = qt_widgets
    pyqt6.__dict__['QtWidgets'] = qt_widgets

    # QtCore
    qt_core = create_mock_module('PyQt6.QtCore')
    mocks['PyQt6.QtCore'] = qt_core
    pyqt6.QtCore = qt_core
    pyqt6.__dict__['QtCore'] = qt_core

    # QtGui
    qt_gui = create_mock_module('PyQt6.QtGui')
    mocks['PyQt6.QtGui'] = qt_gui
    pyqt6.QtGui = qt_gui
    pyqt6.__dict__['QtGui'] = qt_gui

    # QtTest
    qt_test = create_mock_module('PyQt6.QtTest')
    mocks['PyQt6.QtTest'] = qt_test
    pyqt6.QtTest = qt_test
    pyqt6.__dict__['QtTest'] = qt_test

    # PySide6 mirrors
    ps_core = create_mock_module('PySide6.QtCore')
    ps_gui = create_mock_module('PySide6.QtGui')
    ps_widgets = create_mock_module('PySide6.QtWidgets')
    mocks['PySide6.QtCore'] = ps_core
    mocks['PySide6.QtGui'] = ps_gui
    mocks['PySide6.QtWidgets'] = ps_widgets
    pyside6.QtCore = ps_core
    pyside6.QtGui = ps_gui
    pyside6.QtWidgets = ps_widgets
    pyside6.__dict__['QtCore'] = ps_core
    pyside6.__dict__['QtGui'] = ps_gui
    pyside6.__dict__['QtWidgets'] = ps_widgets

    # 3. Inject attributes into QtWidgets
    qt_widgets.QMainWindow = DummyQMainWindow
    qt_widgets.QWidget = DummyQWidget
    qt_widgets.QDialog = DummyQDialog
    qt_widgets.QApplication = DummyQApplication

    # Create enhanced widget classes with signals
    class EnhancedButton(DummyQWidget):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.clicked = MockSignal()
        def setCheckable(self, checkable): pass
        def isChecked(self): return False
        def setChecked(self, checked): pass
        def setFlat(self, flat): pass
        def setStyleSheet(self, style): pass
        def setMinimumWidth(self, width): pass
        def setCursor(self, cursor): pass
    
    class EnhancedComboBox(DummyQWidget):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.currentIndexChanged = MockSignal()
            self._current_index = 0

        def blockSignals(self, block): return False
            
        def addItem(self, *args): self._items.append(args)
        def setCurrentIndex(self, index): self._current_index = index
        def currentIndex(self): return self._current_index
        def currentText(self):
            if 0 <= self._current_index < len(self._items):
                return self._items[self._current_index][0]
            return ""
        def currentData(self):
            if 0 <= self._current_index < len(self._items):
                item = self._items[self._current_index]
                if len(item) > 1: return item[-1]
            return None

        def setEditable(self, editable): pass
        def setEnabled(self, enabled): pass

        def findData(self, data):
             for i, item in enumerate(self._items):
                 if len(item) > 1 and item[-1] == data:
                     return i
             return -1

        def itemData(self, index, role=None):
            if 0 <= index < len(self._items):
                item = self._items[index]
                if len(item) > 1:
                    return item[-1]
            return None
    
    class EnhancedLineEdit(DummyQWidget):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.textChanged = MockSignal()
            self.textEdited = MockSignal()
            
        def setText(self, text): pass
        def text(self): return ""
        def setPlaceholderText(self, text): pass
    
    class EnhancedCheckBox(DummyQWidget):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.stateChanged = MockSignal()
            
        def setCheckState(self, state): pass
        def checkState(self): return 0
        def isChecked(self): return False
        def setChecked(self, checked): pass

    class EnhancedSpinBox(DummyQWidget):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.valueChanged = MockSignal()
            self._value = 0

        def setRange(self, min_val, max_val): pass
        def setMinimum(self, min_val): pass
        def setMaximum(self, max_val): pass
        def setValue(self, value): self._value = value
        def value(self): return self._value
        def setSpecialValueText(self, text): pass
        def setSingleStep(self, step): pass
        def setVisible(self, visible): pass

    qt_widgets.QSpinBox = EnhancedSpinBox
    qt_widgets.QDoubleSpinBox = EnhancedSpinBox # Reuse spinbox mock for double

    class EnhancedFormLayout(DummyQWidget):
        def addRow(self, *args): pass

    qt_widgets.QFormLayout = EnhancedFormLayout

    class EnhancedRadioButton(DummyQWidget):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.toggled = MockSignal()
            self.clicked = MockSignal()
        def setChecked(self, checked): pass
        def isChecked(self): return False

    qt_widgets.QRadioButton = EnhancedRadioButton

    class EnhancedAction(DummyQWidget):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.triggered = MockSignal()
        def setCheckable(self, checkable): pass
    qt_widgets.QAction = EnhancedAction

    class DummyQSizePolicy:
        def __init__(self, *args): pass
        def setHorizontalStretch(self, stretch): pass
        def setVerticalStretch(self, stretch): pass

        Fixed = 0
        Minimum = 1
        Maximum = 4
        Preferred = 5
        Expanding = 7
        MinimumExpanding = 3
        Ignored = 13

    qt_widgets.QSizePolicy = DummyQSizePolicy

    class DummyQListWidgetItem:
        def __init__(self, *args): pass
        def setText(self, text): pass
        def text(self): return ""
        def setData(self, role, data): pass
        def data(self, role): return None
        def setForeground(self, brush): pass

    qt_widgets.QListWidgetItem = DummyQListWidgetItem

    class DummyQMessageBox(DummyQWidget):
        @staticmethod
        def information(*args): return
        @staticmethod
        def warning(*args): return
        @staticmethod
        def critical(*args): return
        @staticmethod
        def question(*args): return 16384 # Yes

        NoButton = 0
        Ok = 1024
        Save = 2048
        SaveAll = 4096
        Open = 8192
        Yes = 16384
        YesToAll = 32768
        No = 65536
        NoToAll = 131072
        Abort = 262144
        Retry = 524288
        Ignore = 1048576
        Close = 2097152
        Cancel = 4194304

    qt_widgets.QMessageBox = DummyQMessageBox

    class DummyQFileDialog(DummyQWidget):
        @staticmethod
        def getOpenFileName(*args): return ("", "")
        @staticmethod
        def getSaveFileName(*args): return ("", "")
        @staticmethod
        def getExistingDirectory(*args): return ""

    qt_widgets.QFileDialog = DummyQFileDialog

    class DummyQInputDialog(DummyQWidget):
        @staticmethod
        def getText(*args): return ("", False)
        @staticmethod
        def getItem(*args): return ("", False)
        @staticmethod
        def getInt(*args): return (0, False)

    qt_widgets.QInputDialog = DummyQInputDialog

    class DummyQAbstractItemView(DummyQWidget):
         class SelectionMode:
             NoSelection = 0
             SingleSelection = 1
             MultiSelection = 2
             ExtendedSelection = 3
             ContiguousSelection = 4
         class DragDropMode:
             NoDragDrop = 0
             DragOnly = 1
             DropOnly = 2
             DragDrop = 3
             InternalMove = 4
         NoSelection = 0
         SingleSelection = 1
         MultiSelection = 2
         ExtendedSelection = 3
         ContiguousSelection = 4

    qt_widgets.QAbstractItemView = DummyQAbstractItemView

    class EnhancedButtonGroup(DummyQWidget):
        def setExclusive(self, exclusive): pass
        def addButton(self, button, id=-1): pass

    qt_widgets.QButtonGroup = EnhancedButtonGroup

    # Map widgets with enhanced signal support
    qt_widgets.QPushButton = EnhancedButton
    qt_widgets.QComboBox = EnhancedComboBox
    qt_widgets.QLineEdit = EnhancedLineEdit
    qt_widgets.QCheckBox = EnhancedCheckBox
    
    # Map other common widgets without special signals
    for w in ['QLabel', 'QVBoxLayout', 'QHBoxLayout', 'QSplitter',
              'QTreeWidget', 'QTreeWidgetItem', 'QTextEdit',
              'QGroupBox', 'QScrollArea', 'QTabWidget', 'QDockWidget', 'QStatusBar', 'QMenuBar', 'QMenu',
              'QGridLayout', 'QTreeView', 'QToolBar', 'QStackedWidget', 'QGraphicsDropShadowEffect', 'QProgressBar',
              'QGraphicsScene', 'QGraphicsEllipseItem', 'QGraphicsLineItem', 'QGraphicsTextItem',
              'QTableWidget', 'QTableWidgetItem']:
         setattr(qt_widgets, w, type(w, (DummyQWidget,), {}))

    class EnhancedGraphicsView(DummyQWidget):
        class DragMode:
            ScrollHandDrag = 1
            RubberBandDrag = 2
            NoDrag = 0
        class ViewportAnchor:
            AnchorUnderMouse = 1
            AnchorViewCenter = 2
    qt_widgets.QGraphicsView = EnhancedGraphicsView

    class EnhancedFrame(DummyQWidget):
        class Shape:
            StyledPanel = 6
        class Shadow:
            Raised = 32
        def setFrameShape(self, shape): pass
        def setFrameShadow(self, shadow): pass
        def setLineWidth(self, width): pass
    qt_widgets.QFrame = EnhancedFrame

    qt_widgets.QHeaderView = type('QHeaderView', (DummyQWidget,), {
        'ResizeMode': type('ResizeMode', (object,), {'ResizeToContents': 0, 'Stretch': 1})
    })

    qt_widgets.QListWidget = type('QListWidget', (DummyQAbstractItemView,), {})

    # 4. Inject into QtCore
    qt_core.Qt = DummyQt
    qt_core.QObject = type('QObject', (object,), {'__init__': lambda s, *a: None, 'blockSignals': lambda s, b: False})
    qt_core.QModelIndex = type('QModelIndex', (object,), {})
    qt_core.QThread = type('QThread', (object,), {'start': lambda s: None, 'wait': lambda s: None, 'quit': lambda s: None, 'isRunning': lambda s: False})
    qt_core.pyqtSignal = lambda *args: unittest.mock.MagicMock(emit=lambda *a: None, connect=lambda *a: None)
    class MockTimer:
        def __init__(self, *args): self.timeout = qt_core.pyqtSignal()
        def singleShot(self, *args): pass
        def start(self, *args): pass
        def stop(self): pass
    qt_core.QTimer = MockTimer
    qt_core.QSize = type('QSize', (object,), {'__init__': lambda s, w, h: None, 'width': lambda s: 0, 'height': lambda s: 0})
    qt_core.QRect = type('QRect', (object,), {'__init__': lambda s, *a: None})
    qt_core.QRectF = type('QRectF', (object,), {'__init__': lambda s, *a: None})
    qt_core.QMimeData = type('QMimeData', (object,), {'__init__': lambda s, *a: None})

    # 4.1 Inject QModelIndex into QtCore (was missing)
    qt_core.QModelIndex = type('QModelIndex', (object,), {})

    # 5. Inject into QtGui
    qt_gui.QColor = lambda *a: None
    qt_gui.QIcon = lambda *a: None
    qt_gui.QCursor = lambda *a: None
    qt_gui.QFont = lambda *a: None
    class MockPainter:
        class RenderHint:
            Antialiasing = 1
    qt_gui.QPainter = MockPainter
    qt_gui.QPen = lambda *a: None
    qt_gui.QBrush = lambda *a: None
    qt_gui.QDrag = lambda *a: None
    qt_gui.QAction = qt_widgets.QAction # Alias
    qt_gui.QStandardItemModel = type('QStandardItemModel', (object,), {'__init__': lambda s, *a: None, 'invisibleRootItem': lambda s: unittest.mock.MagicMock()})
    qt_gui.QStandardItem = type('QStandardItem', (object,), {'__init__': lambda s, *a: None})
    qt_gui.QKeySequence = type('QKeySequence', (object,), {'__init__': lambda s, *a: None, 'StandardKey': type('StandardKey', (object,), {'Save': 1, 'Copy': 2})})

    # Add __all__ to package modules to support "from X import Y"
    pyqt6.__all__ = ['QtWidgets', 'QtCore', 'QtGui', 'QtTest']
    pyside6.__all__ = ['QtWidgets', 'QtCore', 'QtGui']

    # CRITICAL: Populate sys.modules BEFORE installing MetaPathFinder
    # This ensures modules are already loaded when pytest starts collecting tests
    for name, m in mocks.items():
        sys.modules[name] = m

    # Install MetaPathFinder as fallback
    # We must insert it before standard importers
    sys.meta_path.insert(0, StubFinder(mocks))

    print(" [STUB] GUI libraries mocked for headless execution (Custom Dummies via MetaPathFinder).")

def main():
    """
    Main entry point: Setup stubs and run pytest.
    """
    # 1. Setup Stubs
    setup_gui_stubs()

    # 2. Ensure current directory and python/ are in path
    sys.path.insert(0, os.getcwd())
    if os.path.isdir("python"):
        sys.path.insert(0, os.path.abspath("python"))

    # 3. Run Pytest
    import pytest

    # Pass all arguments to pytest
    pytest_args = sys.argv[1:]

    print(f" [RUN] Starting pytest with args: {pytest_args}")
    result = pytest.main(pytest_args)

    sys.exit(result)

if __name__ == "__main__":
    main()
