
import pytest
from dm_toolkit.gui.editor.text_generator import CardTextGenerator

def test_friend_burst_text():
    action = {
        "type": "FRIEND_BURST",
        "str_val": "Fire Bird"
    }

    expected = "＜Fire Bird＞のフレンド・バースト（このクリーチャーが出た時、自分の他のFire Bird・クリーチャーを1体タップしてもよい。そうしたら、このクリーチャーの呪文側をバトルゾーンに置いたまま、コストを支払わずに唱える。）"
    generated = CardTextGenerator._format_action(action)

    assert generated == expected


def test_replace_card_move_text():
    cmd = {
        "type": "REPLACE_CARD_MOVE",
        "from_zone": "GRAVEYARD",
        "to_zone": "DECK_BOTTOM",
        "input_value_key": "card_ref"
    }

    generated = CardTextGenerator._format_command(cmd)

    assert "墓地に置くかわりに" in generated
    assert "山札の下" in generated


def test_replace_card_move_with_input_value_key():
    """Test REPLACE_CARD_MOVE with input_value_key for card reference."""
    cmd = {
        "type": "REPLACE_CARD_MOVE",
        "from_zone": "GRAVEYARD",
        "to_zone": "DECK_BOTTOM",
        "input_value_key": "selected_card",
        "amount": 1
    }

    generated = CardTextGenerator._format_command(cmd)

    # Should reference "そのカード" when input_value_key is present
    assert "そのカード" in generated
    assert "墓地に置くかわりに" in generated
    assert "山札の下に置く" in generated


def test_replace_card_move_deck_top():
    """Test REPLACE_CARD_MOVE moving to deck top instead of graveyard."""
    cmd = {
        "type": "REPLACE_CARD_MOVE",
        "from_zone": "GRAVEYARD",
        "to_zone": "DECK",
        "input_value_key": "card_ref"
    }

    generated = CardTextGenerator._format_command(cmd)

    assert "墓地に置くかわりに" in generated
    # DECK should be localized to "山札" or "デッキ" depending on the translation
    # For now, accept either
    assert ("山札" in generated or "デッキ" in generated or "DECK" in generated)
