import sys
import unittest
try:
    from PyQt6.QtWidgets import QApplication
    from python.gui.card_editor import CardEditor
except ImportError:
    # If PyQt6 is not installed in environment, we might skip or fail.
    # But usually in this sandbox it is installed.
    # The 'python' path might need adjustment.
    pass

# We need to make sure python/ is in sys.path
sys.path.append('python')

class TestCardEditorSyntax(unittest.TestCase):
    def test_import(self):
        try:
            from gui.card_editor import CardEditor
        except ImportError as e:
            self.fail(f"Failed to import CardEditor: {e}")
        except Exception as e:
             self.fail(f"Importing CardEditor caused exception: {e}")

if __name__ == '__main__':
    unittest.main()
