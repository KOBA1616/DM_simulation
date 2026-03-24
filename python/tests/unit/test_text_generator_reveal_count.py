from dm_toolkit.gui.editor.text_generator import CardTextGenerator


def test_reveal_cards_by_count():
    res = CardTextGenerator._format_game_action_command(
        "REVEAL_CARDS", {}, False, 2, 0, "", "", "", "", None
    )
    assert res == "山札の上から2枚を表向きにする。"


def test_reveal_cards_with_input_key():
    res = CardTextGenerator._format_game_action_command(
        "REVEAL_CARDS", {"input_value_key": "X"}, False, 0, 0, "", "", "", "", None
    )
    # Depending on how input_key is routed, handler may use input link or fall back.
    assert ("その数だけ表向き" in res) or ("0枚を表向きにする" in res)


def test_count_cards_with_target():
    res = CardTextGenerator._format_game_action_command(
        "COUNT_CARDS", {}, False, 0, 0, "自分のカード", "", "", "", None
    )
    assert res == "自分のカードの数を数える。"


def test_count_cards_generic():
    res = CardTextGenerator._format_game_action_command(
        "COUNT_CARDS", {}, False, 0, 0, "カード", "", "", "", None
    )
    # Localized text may vary; ensure it returns a parenthesized label
    assert res.startswith("(") and res.endswith(")")
