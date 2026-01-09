# -*- coding: utf-8 -*-
import sys
import pytest

def test_gui_libraries_are_stubbed():
    """
    Verifies that PyQt6 modules are mocked and do not raise ImportErrors.
    This test confirms that the run_pytest_with_pyqt_stub.py harness is working.
    """
    # Attempt to import PyQt6 (which should be mocked)
    try:
        # Import directly first
        from PyQt6 import QtWidgets, QtCore

        # Check if attribute exists on parent
        if hasattr(QtWidgets, 'QMainWindow'):
            QMainWindow = QtWidgets.QMainWindow
        else:
            # Fallback to direct submodule import if parent package linking is broken
            # BUT avoid using import if we are in a fragile stub state.
            # Check sys.modules directly
            mod = sys.modules.get('PyQt6.QtWidgets')
            if mod and hasattr(mod, 'QMainWindow'):
                QMainWindow = mod.QMainWindow
            else:
                # Last ditch: try import
                import PyQt6.QtWidgets
                QMainWindow = PyQt6.QtWidgets.QMainWindow

        # Same for QWidget etc
        QWidget = QtWidgets.QWidget if hasattr(QtWidgets, 'QWidget') else sys.modules['PyQt6.QtWidgets'].QWidget
        Qt = QtCore.Qt if hasattr(QtCore, 'Qt') else sys.modules['PyQt6.QtCore'].Qt

    except ImportError as e:
        pytest.fail(f"PyQt6 import failed: {e}. Stubbing harness might be inactive.")
    except AttributeError as e:
         pytest.fail(f"PyQt6 attribute missing: {e}. Stubbing harness incomplete. Dir: {dir(QtWidgets)}")

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
