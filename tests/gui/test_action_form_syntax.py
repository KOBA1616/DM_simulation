
import sys
import pytest

# Mock BaseEditForm if needed for testing environment, but it seems available.
# We just want to ensure syntax is correct and classes load.

def test_syntax():
    try:
        from dm_toolkit.gui.editor.forms.action_form import ActionEditForm
        print("ActionEditForm syntax check passed.")
    except ImportError as e:
        if "libEGL" in str(e) or "cannot open shared object file" in str(e) or "PyQt6" in str(e):
            pytest.skip("Skipping GUI test due to missing libraries in headless environment: " + str(e))
        else:
            raise e
    except Exception as e:
        pytest.fail(f"Syntax check failed with error: {e}")

if __name__ == "__main__":
    test_syntax()
