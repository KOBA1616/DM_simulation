import sys
import os
from unittest.mock import MagicMock

# Add root directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Define dummy base classes BEFORE imports
class MockQWidget:
    def __init__(self, parent=None):
        pass
    def setLayout(self, layout):
        pass

class MockQVBoxLayout:
    def __init__(self, parent=None):
        pass
    def setContentsMargins(self, *args):
        pass
    def addWidget(self, widget):
        pass

# Mock modules
sys.modules['PyQt6'] = MagicMock()
sys.modules['PyQt6.QtWidgets'] = MagicMock()
sys.modules['PyQt6.QtWidgets'].QWidget = MockQWidget
sys.modules['PyQt6.QtWidgets'].QVBoxLayout = MockQVBoxLayout

sys.modules['PyQt6.QtCore'] = MagicMock()
sys.modules['PyQt6.QtGui'] = MagicMock()
sys.modules['dm_toolkit.gui.i18n'] = MagicMock()
sys.modules['dm_toolkit.gui.utils'] = MagicMock()
sys.modules['dm_toolkit.gui.utils.card_helpers'] = MagicMock()

# Define CardPreviewWidget mock
class MockPreviewWidget:
    def __init__(self, parent=None):
        self.last_rendered_data = None
        print("MockPreviewWidget initialized")

    def render_card(self, data):
        print(f"Mock render_card called with: {data}")
        self.last_rendered_data = data

    def clear_preview(self):
        self.last_rendered_data = None

# Mock the module that contains CardPreviewWidget
preview_pane_mock = MagicMock()
preview_pane_mock.CardPreviewWidget = MockPreviewWidget
sys.modules['dm_toolkit.gui.editor.preview_pane'] = preview_pane_mock

# Now import the class to test
from dm_toolkit.gui.widgets.card_detail_panel import CardDetailPanel

def test_update_card_with_dict():
    print("Initializing CardDetailPanel...")
    panel = CardDetailPanel()

    mock_data = {
        'id': 1,
        'name': 'Test Card',
        'effects': [{'type': 'test'}],
        'triggers': []
    }

    print(f"Updating card with: {mock_data}")
    panel.update_card(mock_data)

    rendered = panel.preview.last_rendered_data
    print(f"Rendered data: {rendered}")

    assert rendered is not None
    assert rendered['id'] == 1
    assert rendered['name'] == 'Test Card'
    assert 'effects' in rendered
    assert len(rendered['effects']) == 1
    print("Test passed: Dictionary data passed correctly to render_card")

if __name__ == "__main__":
    test_update_card_with_dict()
