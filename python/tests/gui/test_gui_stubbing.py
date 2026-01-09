# -*- coding: utf-8 -*-
import sys
import pytest

def test_gui_libraries_are_stubbed():
    """
    Verifies that PyQt6 modules are mocked and do not raise ImportErrors.
    This test confirms that stubbing infrastructure (conftest.py or run_pytest_with_pyqt_stub.py) is working.
    
    Note: Due to test collection order, PyQt6 modules may already be in sys.modules.
    We verify they can be imported and used, even if they were cached.
    """
    # Attempt to import PyQt6 (which should be mocked)
    try:
        # Force re-import to get fresh modules
        if 'PyQt6.QtWidgets' in sys.modules:
            del sys.modules['PyQt6.QtWidgets']
        if 'PyQt6.QtCore' in sys.modules:
            del sys.modules['PyQt6.QtCore']
        if 'PyQt6' in sys.modules:
            del sys.modules['PyQt6']
        
        # Re-setup stubs if needed (call the conftest function)
        from conftest import _setup_minimal_gui_stubs
        _setup_minimal_gui_stubs()
        
        # Now import
        from PyQt6 import QtWidgets, QtCore
        
        # Access classes directly from the imported modules
        QMainWindow = QtWidgets.QMainWindow
        QWidget = QtWidgets.QWidget
        Qt = QtCore.Qt

    except ImportError as e:
        pytest.fail(f"PyQt6 import failed: {e}. Stubbing harness might be inactive.")
    except AttributeError as e:
        # Detailed debugging info
        pyqt6_mod = sys.modules.get('PyQt6')
        qtwidgets_mod = sys.modules.get('PyQt6.QtWidgets')
        pytest.fail(
            f"PyQt6 attribute missing: {e}.\n"
            f"PyQt6 in sys.modules: {pyqt6_mod is not None}\n"
            f"PyQt6.__dict__ keys: {list(pyqt6_mod.__dict__.keys()) if pyqt6_mod else 'N/A'}\n"
            f"QtWidgets in sys.modules: {qtwidgets_mod is not None}\n"
            f"QtWidgets.QMainWindow exists: {hasattr(qtwidgets_mod, 'QMainWindow') if qtwidgets_mod else 'N/A'}"
        )

    # Verify that these are indeed mocks (or at least present)
    assert QMainWindow is not None
    assert QWidget is not None
    assert Qt is not None

    # Check that we can instantiate/inherit without error (basic check)
    class MyWindow(QMainWindow):
        pass

    w = MyWindow()
    assert w is not None

    print("\n [OK] PyQt6 stubbing verified successfully.")
