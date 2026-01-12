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
            
        def setWindowTitle(self, title): pass
        def setLayout(self, layout): pass
        def setGeometry(self, *args): pass
        def show(self): pass
        def close(self): return True
        def addWidget(self, widget): pass
        def addLayout(self, layout): pass
        def setText(self, text): pass
        def text(self): return ""
        def setCheckState(self, state): pass
        def addItem(self, *args): pass
        def setCurrentIndex(self, index): pass
        def currentIndex(self): return 0

    class DummyQMainWindow(DummyQWidget):
        def setCentralWidget(self, widget): pass
        def setMenuBar(self, menu): pass
        def addDockWidget(self, area, dock): pass
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

        class WindowType:
            Window = 0x00000001

        class MatchFlag:
            MatchContains = 1
            MatchFixedString = 8

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
    
    class EnhancedComboBox(DummyQWidget):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.currentIndexChanged = MockSignal()
            
        def addItem(self, *args): pass
        def setCurrentIndex(self, index): pass
        def currentIndex(self): return 0
        def currentText(self): return ""
    
    class EnhancedLineEdit(DummyQWidget):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.textChanged = MockSignal()
            
        def setText(self, text): pass
        def text(self): return ""
    
    class EnhancedCheckBox(DummyQWidget):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.stateChanged = MockSignal()
            
        def setCheckState(self, state): pass
        def checkState(self): return 0
        def isChecked(self): return False

    # Map widgets with enhanced signal support
    qt_widgets.QPushButton = EnhancedButton
    qt_widgets.QComboBox = EnhancedComboBox
    qt_widgets.QLineEdit = EnhancedLineEdit
    qt_widgets.QCheckBox = EnhancedCheckBox
    
    # Map other common widgets without special signals
    for w in ['QLabel', 'QVBoxLayout', 'QHBoxLayout', 'QSplitter',
              'QTreeWidget', 'QTreeWidgetItem', 'QTextEdit',
              'QGroupBox', 'QScrollArea', 'QTabWidget', 'QDockWidget', 'QStatusBar', 'QMenuBar', 'QMenu', 'QAction']:
         setattr(qt_widgets, w, type(w, (DummyQWidget,), {}))

    # 4. Inject into QtCore
    qt_core.Qt = DummyQt
    qt_core.QObject = type('QObject', (object,), {'__init__': lambda s, *a: None, 'blockSignals': lambda s, b: False})
    qt_core.QThread = type('QThread', (object,), {'start': lambda s: None, 'wait': lambda s: None, 'quit': lambda s: None, 'isRunning': lambda s: False})
    qt_core.pyqtSignal = lambda *args: unittest.mock.MagicMock(emit=lambda *a: None, connect=lambda *a: None)
    qt_core.QTimer = type('QTimer', (object,), {'singleShot': lambda *a: None, 'start': lambda s, t: None, 'stop': lambda s: None})

    # 5. Inject into QtGui
    qt_gui.QColor = lambda *a: None
    qt_gui.QIcon = lambda *a: None
    qt_gui.QAction = qt_widgets.QAction # Alias
    qt_gui.QStandardItemModel = type('QStandardItemModel', (object,), {'__init__': lambda s, *a: None, 'invisibleRootItem': lambda s: unittest.mock.MagicMock()})
    qt_gui.QStandardItem = type('QStandardItem', (object,), {'__init__': lambda s, *a: None})

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
