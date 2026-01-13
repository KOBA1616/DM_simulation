import sys
from unittest.mock import MagicMock

# Mock PyQt6 modules
sys.modules["PyQt6"] = MagicMock()
sys.modules["PyQt6.QtWidgets"] = MagicMock()
sys.modules["PyQt6.QtCore"] = MagicMock()
sys.modules["PyQt6.QtGui"] = MagicMock()

# Now import the target module
from dm_toolkit.gui.editor.text_generator import CardTextGenerator

def verify_legacy_removal():
    # 1. Test TextGenerator with Legacy Actions
    effect_data = {
        "actions": [{"type": "DRAW_CARD", "value1": 1}]
    }

    # Current behavior: It should produce text because legacy support is still there.
    # After my changes: It should NOT produce text or should ignore 'actions'.

    try:
        text = CardTextGenerator._format_effect(effect_data)
        print(f"Legacy Action Text: '{text}'")
    except Exception as e:
        print(f"Legacy Action Text Error: {e}")

    # 2. Test TextGenerator with Commands
    command_data = {
        "commands": [{"type": "DRAW_CARD", "amount": 1}]
    }
    try:
        text_cmd = CardTextGenerator._format_effect(command_data)
        print(f"Command Text: '{text_cmd}'")
    except Exception as e:
        print(f"Command Text Error: {e}")

if __name__ == "__main__":
    verify_legacy_removal()
