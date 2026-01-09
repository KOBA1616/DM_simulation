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

def setup_gui_stubs():
    """
    Injects mocks for PyQt6 and PySide6 modules into sys.modules.
    This prevents ImportErrors when tests try to import GUI components in a headless environment.
    """
    # Create dummy classes for inheritance
    class DummyQWidget(object):
        def __init__(self, *args, **kwargs): pass
        def setWindowTitle(self, title): pass
        def setLayout(self, layout): pass
        def setGeometry(self, *args): pass
        def show(self): pass
        def close(self): return True

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

    # Modules to mock
    modules_to_mock = [
        'PyQt6', 'PyQt6.QtCore', 'PyQt6.QtGui', 'PyQt6.QtWidgets', 'PyQt6.QtTest',
        'PySide6', 'PySide6.QtCore', 'PySide6.QtGui', 'PySide6.QtWidgets'
    ]

    for mod_name in modules_to_mock:
        if mod_name not in sys.modules:
            sys.modules[mod_name] = unittest.mock.MagicMock()

    # Inject specific dummies into QtWidgets
    mock_widgets = sys.modules['PyQt6.QtWidgets']
    mock_widgets.QMainWindow = DummyQMainWindow
    mock_widgets.QWidget = DummyQWidget
    mock_widgets.QDialog = DummyQDialog
    mock_widgets.QApplication = DummyQApplication
    # Map common widgets to DummyQWidget or MagicMock as needed
    for w in ['QLabel', 'QPushButton', 'QVBoxLayout', 'QHBoxLayout', 'QSplitter',
              'QTreeWidget', 'QTreeWidgetItem', 'QComboBox', 'QLineEdit', 'QTextEdit',
              'QCheckBox', 'QGroupBox', 'QScrollArea', 'QTabWidget', 'QDockWidget', 'QStatusBar', 'QMenuBar', 'QMenu', 'QAction']:
         setattr(mock_widgets, w, type(w, (DummyQWidget,), {}))

    # Inject into QtCore
    mock_core = sys.modules['PyQt6.QtCore']
    mock_core.Qt = DummyQt
    mock_core.QObject = type('QObject', (object,), {'__init__': lambda s, *a: None, 'blockSignals': lambda s, b: False})
    mock_core.QThread = type('QThread', (object,), {'start': lambda s: None, 'wait': lambda s: None, 'quit': lambda s: None, 'isRunning': lambda s: False})
    mock_core.pyqtSignal = lambda *args: unittest.mock.MagicMock(emit=lambda *a: None, connect=lambda *a: None)
    mock_core.QTimer = type('QTimer', (object,), {'singleShot': lambda *a: None, 'start': lambda s, t: None, 'stop': lambda s: None})

    # Inject into QtGui
    mock_gui = sys.modules['PyQt6.QtGui']
    mock_gui.QColor = lambda *a: None
    mock_gui.QIcon = lambda *a: None
    mock_gui.QAction = type('QAction', (DummyQWidget,), {})
    mock_gui.QStandardItemModel = type('QStandardItemModel', (object,), {'__init__': lambda s, *a: None, 'invisibleRootItem': lambda s: unittest.mock.MagicMock()})
    mock_gui.QStandardItem = type('QStandardItem', (object,), {'__init__': lambda s, *a: None})

    print(" [STUB] GUI libraries mocked for headless execution (Custom Dummies).")

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
