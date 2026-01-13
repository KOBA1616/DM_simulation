import sys
import os

# Add project root to path
sys.path.append(os.getcwd())

from dm_toolkit.gui.editor.text_generator import CardTextGenerator

# Mock dm_ai_module
class MockDM:
    pass
sys.modules["dm_ai_module"] = MockDM

# Mock CardTextResources if needed, but it should be fine.
# We test generate_text via _format_command (implicitly)

def test_select_number():
    command = {
        "type": "SELECT_NUMBER",
        "amount": 1,
        "val2": 6
    }

    # We call _format_command directly to test the text generation for a single command
    text = CardTextGenerator._format_command(command)
    print(f"Generated Text: {text}")

    expected = "1～6の数字を1つ選ぶ。"
    if text == expected:
        print("PASS")
    else:
        print(f"FAIL: Expected '{expected}', got '{text}'")
        sys.exit(1)

def test_select_number_defaults():
    # If val2 is missing (legacy data), we expect behavior (currently failing or empty)
    # With new code, val2 default is 0 if not present in dict.
    # Text gen `if val1 > 0 and val2 > 0` requires both.

    command = {
        "type": "SELECT_NUMBER",
        "amount": 5
        # val2 missing
    }
    text = CardTextGenerator._format_command(command)
    print(f"Generated Text (Missing val2): {text}")

    # Updated expectation: Falls back to the default translation for SELECT_NUMBER
    expected_fallback = "数字を1つ選ぶ。"

    if text == expected_fallback:
        print("PASS (Fallback)")
    else:
         print(f"FAIL (Fallback): Expected '{expected_fallback}', Got '{text}'")

if __name__ == "__main__":
    test_select_number()
    test_select_number_defaults()
