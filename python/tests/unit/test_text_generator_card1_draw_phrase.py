# -*- coding: utf-8 -*-
import json
import pathlib

from dm_toolkit.gui.editor.text_generator import CardTextGenerator


def _load_card_1() -> dict:
    root = pathlib.Path(__file__).resolve().parents[3]
    cards_path = root / "data" / "cards.json"
    cards = json.loads(cards_path.read_text(encoding="utf-8"))
    for card in cards:
        if card.get("id") == 1:
            return card
    raise AssertionError("card id=1 not found")


def test_card1_draw_phrase_uses_mana_civ_count_and_linked_hand_to_deck_bottom_text():
    card = _load_card_1()
    body = CardTextGenerator.generate_body_text(card)

    assert "マナゾーンの文明数を数える。" in body
    assert "山札からカードをマナゾーンの文明数まで引いてもよい。" in body
    assert "自分の手札からカードを引いた枚数だけ選び、山札の下に置く。" in body
