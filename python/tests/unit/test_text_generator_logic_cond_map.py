from dm_toolkit.gui.editor.text_generator import CardTextGenerator


def test_opponent_draw_count_condition():
    action = {"condition": {"type": "OPPONENT_DRAW_COUNT", "value": 3}}
    # atype 'IF' triggers reading if_true etc; we only need cond formatting
    res = CardTextGenerator._format_logic_command("IF", {"condition": action["condition"], "if_true": []}, False, None, False)
    assert "相手がカードを3枚目以上引いたなら" in res
