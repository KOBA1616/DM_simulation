
import sys
import os

# Ensure we can import from repo root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from dm_toolkit.gui.editor.text_generator import CardTextGenerator
from dm_toolkit.command_builders import build_destroy_command

def test_trigger_text_generation():
    # Define a card with the new trigger and target
    card_data = {
        "name": "Test Card",
        "civilizations": ["FIRE"],
        "type": "CREATURE",
        "effects": [
            {
                "trigger": "ON_OPPONENT_CREATURE_ENTER",
                "commands": [
                    build_destroy_command(
                        native=False,
                        target_filter={
                            "is_trigger_source": True,
                            "types": ["CREATURE"]
                        }
                    )
                ]
            }
        ]
    }

    text = CardTextGenerator.generate_text(card_data)
    print("Generated Text:\n", text)

    expected_trigger = "相手のクリーチャーが出た時"
    expected_target = "そのクリーチャーを破壊する"

    if expected_trigger not in text:
        print(f"FAIL: Expected trigger text '{expected_trigger}' not found.")
        sys.exit(1)

    if expected_target not in text:
        print(f"FAIL: Expected target text '{expected_target}' not found.")
        sys.exit(1)

    print("SUCCESS: Text generation verified.")

if __name__ == "__main__":
    test_trigger_text_generation()
