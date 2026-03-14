# -*- coding: utf-8 -*-
from dm_toolkit.gui.editor.window import CardEditor
from PyQt6.QtCore import Qt


def test_find_card_item_from_item_card():
    win = object.__new__(CardEditor)

    class CardItem:
        def data(self, role):
            return "CARD"

    card = CardItem()
    assert win._find_card_item_from_item(card) is card


def test_find_card_item_from_item_effect():
    win = object.__new__(CardEditor)

    class Parent:
        pass

    parent = Parent()

    class EffectItem:
        def parent(self):
            return parent
        def data(self, role):
            return "EFFECT"

    eff = EffectItem()
    assert win._find_card_item_from_item(eff) is parent


def test_find_card_item_from_item_command():
    win = object.__new__(CardEditor)

    class Grand:
        pass

    grand = Grand()

    class Parent:
        def parent(self):
            return grand

    class CmdItem:
        def parent(self):
            return Parent()
        def data(self, role):
            return "COMMAND"

    cmd = CmdItem()
    assert win._find_card_item_from_item(cmd) is grand
