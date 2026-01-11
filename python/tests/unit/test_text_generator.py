
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
