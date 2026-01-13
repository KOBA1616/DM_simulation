# -*- coding: utf-8 -*-
"""
set_deckの動作を確認するテスト
"""
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    import dm_ai_module
except ImportError:
    print("Warning: dm_ai_module not available, skipping test")
    sys.exit(0)


def test_set_deck_basic():
    """set_deckが正しくデッキを設定することを確認"""
    print("Testing set_deck...")
    
    gs = dm_ai_module.GameState(42)
    gs.setup_test_duel()
    
    # デッキを設定前
    print(f"Before set_deck - P0 deck: {len(gs.players[0].deck)}, P1 deck: {len(gs.players[1].deck)}")
    
    # P0のデッキを設定
    deck0 = [1] * 40
    gs.set_deck(0, deck0)
    print(f"\nAfter P0 set_deck - P0 deck: {len(gs.players[0].deck)}, P1 deck: {len(gs.players[1].deck)}")
    
    # P0のデッキの最初の3枚を確認
    print(f"P0 deck[0]: card_id={gs.players[0].deck[0].card_id}, instance_id={gs.players[0].deck[0].instance_id}, owner={gs.players[0].deck[0].owner}")
    print(f"P0 deck[1]: card_id={gs.players[0].deck[1].card_id}, instance_id={gs.players[0].deck[1].instance_id}, owner={gs.players[0].deck[1].owner}")
    print(f"P0 deck[39]: card_id={gs.players[0].deck[39].card_id}, instance_id={gs.players[0].deck[39].instance_id}, owner={gs.players[0].deck[39].owner}")
    
    # P1のデッキを設定
    deck1 = [1] * 40
    gs.set_deck(1, deck1)
    print(f"\nAfter P1 set_deck - P0 deck: {len(gs.players[0].deck)}, P1 deck: {len(gs.players[1].deck)}")
    
    # P1のデッキの最初の3枚を確認
    print(f"P1 deck[0]: card_id={gs.players[1].deck[0].card_id}, instance_id={gs.players[1].deck[0].instance_id}, owner={gs.players[1].deck[0].owner}")
    print(f"P1 deck[1]: card_id={gs.players[1].deck[1].card_id}, instance_id={gs.players[1].deck[1].instance_id}, owner={gs.players[1].deck[1].owner}")
    print(f"P1 deck[39]: card_id={gs.players[1].deck[39].card_id}, instance_id={gs.players[1].deck[39].instance_id}, owner={gs.players[1].deck[39].owner}")
    
    # Assertions
    assert len(gs.players[0].deck) == 40, f"P0 deck should have 40 cards, got {len(gs.players[0].deck)}"
    assert len(gs.players[1].deck) == 40, f"P1 deck should have 40 cards, got {len(gs.players[1].deck)}"
    assert gs.players[0].deck[0].owner == 0, f"P0 deck[0] owner should be 0, got {gs.players[0].deck[0].owner}"
    assert gs.players[1].deck[0].owner == 1, f"P1 deck[0] owner should be 1, got {gs.players[1].deck[0].owner}"
    assert gs.players[0].deck[0].instance_id == 0, f"P0 deck[0] instance_id should be 0"
    assert gs.players[1].deck[0].instance_id == 40, f"P1 deck[0] instance_id should be 40"
    
    print("\n✓ Test passed!")


if __name__ == "__main__":
    test_set_deck_basic()
