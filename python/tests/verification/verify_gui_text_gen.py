import sys
import os

# Add root directory to path to allow imports
sys.path.append(os.getcwd())

from dm_toolkit.gui.editor.text_generator import CardTextGenerator

def test_reveal_cards():
    print("Testing REVEAL_CARDS...")
    # Case 1: Static amount
    cmd_static = {
        "type": "REVEAL_CARDS",
        "amount": 3
    }
    text_static = CardTextGenerator._format_command(cmd_static)
    print(f"Static (3): {text_static}")
    assert "3枚を表向きにする" in text_static

    # Case 2: Input Link
    cmd_linked = {
        "type": "REVEAL_CARDS",
        "amount": 0,
        "input_link": "prev_output",
        "input_usage": "COUNT"
    }
    text_linked = CardTextGenerator._format_command(cmd_linked)
    print(f"Linked: {text_linked}")
    assert "その数だけ表向きにする" in text_linked

def test_play_from_zone_linked():
    print("\nTesting PLAY_FROM_ZONE with Input Link...")
    # Case: Play creature with max cost determined by previous input
    cmd = {
        "type": "PLAY_FROM_ZONE",
        "from_zone": "MANA_ZONE",
        "to_zone": "BATTLE_ZONE",
        "amount": 0, # Should be ignored due to input link
        "input_link": "prev_output",
        "input_usage": "MAX_COST",
        "filter": {
            "types": ["CREATURE"],
            "max_cost": {
                "input_value_usage": "MAX_COST"
            }
        }
    }
    text = CardTextGenerator._format_command(cmd)
    print(f"Linked: {text}")
    # Expect: "マナゾーンからコストその数以下のクリーチャーを召喚する。"
    # Note: MAX_COST usage label is intentionally suppressed in text_generator.py
    assert "コストその数以下の" in text
    # assert "最大コストとして使用" in text  <-- Suppressed for MAX_COST

def test_mekraid_linked():
    print("\nTesting MEKRAID with Input Link...")
    cmd = {
        "type": "MEKRAID",
        "amount": 0,
        "input_link": "prev_output",
        "input_usage": "MAX_COST"
    }
    text = CardTextGenerator._format_command(cmd)
    print(f"Linked: {text}")
    assert "メクレイドその数" in text

if __name__ == "__main__":
    try:
        test_reveal_cards()
        test_play_from_zone_linked()
        test_mekraid_linked()
        print("\nAll verification tests passed!")
    except AssertionError as e:
        print(f"\nFAILURE: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\nERROR: {e}")
        sys.exit(1)
