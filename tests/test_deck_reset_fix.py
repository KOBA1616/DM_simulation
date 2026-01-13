# -*- coding: utf-8 -*-
"""
デッキ読み込みとリセット時のP1手札/シールド自動配置の修正を検証するテスト
"""
import sys
import os

# パスを追加
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    import dm_ai_module
except ImportError:
    print("Warning: dm_ai_module not available, skipping test")
    sys.exit(0)


def test_reset_with_no_decks():
    """
    デッキを指定せずにリセットした場合、両プレイヤーに手札とシールドが配置されることを確認
    """
    from dm_toolkit.gui.game_session import GameSession
    from dm_toolkit.engine.compat import EngineCompat
    
    def dummy_update_ui():
        pass
    
    def dummy_log(msg):
        print(f"[LOG] {msg}")
    
    # GameSessionを作成
    session = GameSession(
        callback_update_ui=dummy_update_ui,
        callback_log=dummy_log
    )
    
    # カードDBをロード
    card_db = EngineCompat.load_cards_robust("data/cards.json")
    session.initialize_game(card_db, seed=42)
    
    # ゲーム状態を取得
    gs = session.gs
    
    # デッキサイズを確認
    print(f"P0 Deck size before start_game: {len(gs.players[0].deck)} cards")
    print(f"P1 Deck size before start_game: {len(gs.players[1].deck)} cards")
    
    # 両プレイヤーの手札とシールドを確認
    print(f"P0 Hand: {len(gs.players[0].hand)} cards")
    print(f"P0 Shields: {len(gs.players[0].shield_zone)} cards")
    print(f"P1 Hand: {len(gs.players[1].hand)} cards")
    print(f"P1 Shields: {len(gs.players[1].shield_zone)} cards")
    
    # アサーション
    assert len(gs.players[0].hand) == 5, f"P0 should have 5 cards in hand, got {len(gs.players[0].hand)}"
    assert len(gs.players[0].shield_zone) == 5, f"P0 should have 5 shields, got {len(gs.players[0].shield_zone)}"
    assert len(gs.players[1].hand) == 5, f"P1 should have 5 cards in hand, got {len(gs.players[1].hand)}"
    assert len(gs.players[1].shield_zone) == 5, f"P1 should have 5 shields, got {len(gs.players[1].shield_zone)}"
    
    print("✓ Test passed: Both players have correct hand and shields")


def test_reset_with_p0_deck_only():
    """
    P0のデッキのみを指定してリセットした場合、P1にもデフォルトデッキが設定されることを確認
    """
    from dm_toolkit.gui.game_session import GameSession
    from dm_toolkit.engine.compat import EngineCompat
    
    def dummy_update_ui():
        pass
    
    def dummy_log(msg):
        print(f"[LOG] {msg}")
    
    # GameSessionを作成
    session = GameSession(
        callback_update_ui=dummy_update_ui,
        callback_log=dummy_log
    )
    
    # カードDBをロード
    card_db = EngineCompat.load_cards_robust("data/cards.json")
    session.card_db = card_db
    
    # P0のデッキのみを指定してリセット
    p0_deck = [1] * 40
    session.reset_game(p0_deck=p0_deck, p1_deck=None)
    
    # ゲーム状態を取得
    gs = session.gs
    
    # 両プレイヤーの手札とシールドを確認
    print(f"P0 Hand: {len(gs.players[0].hand)} cards")
    print(f"P0 Shields: {len(gs.players[0].shield_zone)} cards")
    print(f"P1 Hand: {len(gs.players[1].hand)} cards")
    print(f"P1 Shields: {len(gs.players[1].shield_zone)} cards")
    
    # アサーション
    assert len(gs.players[0].hand) == 5, f"P0 should have 5 cards in hand, got {len(gs.players[0].hand)}"
    assert len(gs.players[0].shield_zone) == 5, f"P0 should have 5 shields, got {len(gs.players[0].shield_zone)}"
    assert len(gs.players[1].hand) == 5, f"P1 should have 5 cards in hand, got {len(gs.players[1].hand)}"
    assert len(gs.players[1].shield_zone) == 5, f"P1 should have 5 shields, got {len(gs.players[1].shield_zone)}"
    
    print("✓ Test passed: P1 uses default deck when not specified")


if __name__ == "__main__":
    print("Testing deck reset fix...")
    print("=" * 60)
    
    print("\nTest 1: Reset with no decks")
    print("-" * 60)
    test_reset_with_no_decks()
    
    print("\nTest 2: Reset with P0 deck only")
    print("-" * 60)
    test_reset_with_p0_deck_only()
    
    print("\n" + "=" * 60)
    print("All tests passed! ✓")
