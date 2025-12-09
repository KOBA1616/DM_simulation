import sys
import unittest
import importlib.util

class TestCardEditorSyntax(unittest.TestCase):
    def test_import(self):
        # check if PyQt6 is installed
        if importlib.util.find_spec("PyQt6") is None:
            print("PyQt6 not installed, skipping test")
            return

        sys.path.append('python')
        try:
            from dm_toolkit.gui.card_editor import CardEditor
        except ImportError as e:
            self.fail(f"Failed to import CardEditor: {e}")
        except Exception as e:
             self.fail(f"Importing CardEditor caused exception: {e}")

if __name__ == '__main__':
    unittest.main()
