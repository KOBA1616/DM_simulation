
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
