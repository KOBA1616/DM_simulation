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
    modules_to_mock = [
        'PyQt6', 'PyQt6.QtCore', 'PyQt6.QtGui', 'PyQt6.QtWidgets', 'PyQt6.QtTest',
        'PySide6', 'PySide6.QtCore', 'PySide6.QtGui', 'PySide6.QtWidgets'
    ]

    for mod_name in modules_to_mock:
        if mod_name not in sys.modules:
            sys.modules[mod_name] = unittest.mock.MagicMock()

    # Specific mocks for commonly used classes to avoid attribute errors
    # if tests try to inherit from them or access static members.

    # Mock QMainWindow, QWidget, QDialog for inheritance
    mock_widgets = sys.modules['PyQt6.QtWidgets']
    mock_widgets.QMainWindow = unittest.mock.MagicMock
    mock_widgets.QWidget = unittest.mock.MagicMock
    mock_widgets.QDialog = unittest.mock.MagicMock
    mock_widgets.QApplication = unittest.mock.MagicMock

    # Mock QtCore enums and classes
    mock_core = sys.modules['PyQt6.QtCore']
    mock_core.Qt = unittest.mock.MagicMock()
    mock_core.Qt.ItemDataRole = unittest.mock.MagicMock()
    mock_core.Qt.ItemDataRole.DisplayRole = 0
    mock_core.Qt.ItemDataRole.UserRole = 256
    mock_core.QObject = unittest.mock.MagicMock

    print(" [STUB] GUI libraries mocked for headless execution.")

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
