"""
Check initial deck placement and game state consistency.
Verifies:
1. Deck size and composition
2. Player initial state (hand, mana, battle zones)
3. Game phase and active player
4. Player modes
"""

import sys
import dm_ai_module

def check_deck_consistency():
    """Check that deck is properly set and initialized."""
    print("=" * 70)
    print("DECK CONSISTENCY CHECK")
    print("=" * 70)
    
    # Load card database
    try:
        card_db = dm_ai_module.JsonLoader.load_cards("data/cards.json")
        print("✅ Card database loaded")
    except Exception as e:
        print(f"❌ Failed to load card database: {e}")
        return False
    
    # Create GameInstance
    try:
        gi = dm_ai_module.GameInstance(42, card_db)
        gs = gi.state
        print("✅ GameInstance created")
    except Exception as e:
        print(f"❌ Failed to create GameInstance: {e}")
        return False
    
    # Setup test duel
    try:
        gs.setup_test_duel()
        print("✅ setup_test_duel() executed")
    except Exception as e:
        print(f"❌ Failed to setup_test_duel: {e}")
        return False
    
    # Verify initial state after setup_test_duel
    print("\n--- Initial State (after setup_test_duel) ---")
    print(f"Turn number: {gs.turn_number}")
    print(f"Active player: {gs.active_player_id}")
    print(f"Current phase: {gs.current_phase}")
    print(f"Game over: {gs.game_over}")
    
    # Check player state before deck set
    for pid in range(2):
        p = gs.players[pid]
        print(f"\nPlayer {pid} (before deck set):")
        print(f"  Hand: {len(p.hand)} cards")
        print(f"  Mana zone: {len(p.mana_zone)} cards")
        print(f"  Battle zone: {len(p.battle_zone)} cards")
        print(f"  Shield zone: {len(p.shield_zone)} cards")
        print(f"  Graveyard: {len(p.graveyard)} cards")
        print(f"  Deck: {len(p.deck)} cards")
    
    # Set decks (default: 40 cards each)
    default_deck = [1,2,3,4,5,6,7,8,9,10]*4
    print(f"\n--- Setting Decks (40 cards each) ---")
    print(f"Default deck composition: {default_deck[:10]}...{default_deck[-10:]}")
    print(f"Default deck size: {len(default_deck)}")
    
    try:
        gs.set_deck(0, default_deck)
        gs.set_deck(1, default_deck)
        print("✅ Decks set for both players")
    except Exception as e:
        print(f"❌ Failed to set decks: {e}")
        return False
    
    # Check player state after deck set
    print("\n--- Player State (after deck set) ---")
    for pid in range(2):
        p = gs.players[pid]
        print(f"Player {pid}:")
        print(f"  Hand: {len(p.hand)} cards")
        print(f"  Mana zone: {len(p.mana_zone)} cards")
        print(f"  Battle zone: {len(p.battle_zone)} cards")
        print(f"  Shield zone: {len(p.shield_zone)} cards")
        print(f"  Graveyard: {len(p.graveyard)} cards")
        print(f"  Deck: {len(p.deck)} cards (should be 40)")
        
        # Verify deck size
        if len(p.deck) != 40:
            print(f"  ❌ ERROR: Expected 40 cards in deck, got {len(p.deck)}")
            return False
        else:
            print(f"  ✅ Deck size correct")
    
    # Start game
    print("\n--- Starting Game ---")
    try:
        dm_ai_module.PhaseManager.start_game(gs, card_db)
        print("✅ PhaseManager.start_game() executed")
    except Exception as e:
        print(f"⚠️  start_game failed (may not be implemented): {e}")
    
    # Fast forward
    print("\n--- Fast Forwarding ---")
    try:
        dm_ai_module.PhaseManager.fast_forward(gs, card_db)
        print("✅ PhaseManager.fast_forward() executed")
    except Exception as e:
        print(f"⚠️  fast_forward failed (may not be implemented): {e}")
    
    # Check final state
    print("\n--- Game State (after fast_forward) ---")
    print(f"Turn number: {gs.turn_number}")
    print(f"Active player: {gs.active_player_id}")
    print(f"Current phase: {gs.current_phase}")
    print(f"Game over: {gs.game_over}")
    
    # Check player state after game start
    print("\n--- Player State (after game start) ---")
    for pid in range(2):
        p = gs.players[pid]
        print(f"Player {pid}:")
        print(f"  Hand: {len(p.hand)} cards")
        print(f"  Mana zone: {len(p.mana_zone)} cards")
        print(f"  Battle zone: {len(p.battle_zone)} cards")
        print(f"  Shield zone: {len(p.shield_zone)} cards")
        print(f"  Graveyard: {len(p.graveyard)} cards")
        print(f"  Deck: {len(p.deck)} cards")
        
        # Basic consistency check: total cards should be 40
        total_cards = (len(p.hand) + len(p.mana_zone) + len(p.battle_zone) + 
                      len(p.shield_zone) + len(p.graveyard) + len(p.deck))
        print(f"  Total cards: {total_cards} (should be 40)")
        if total_cards != 40:
            print(f"  ⚠️  WARNING: Expected 40 total cards, got {total_cards}")
    
    # Check player modes
    print("\n--- Player Modes ---")
    try:
        for pid in range(2):
            mode = gs.player_modes[pid]
            is_human = gs.is_human_player(pid)
            mode_name = "HUMAN" if is_human else "AI"
            print(f"Player {pid}: {mode_name} (value: {mode})")
        print("✅ Player modes accessible")
    except Exception as e:
        print(f"⚠️  Could not check player modes: {e}")
    
    print("\n" + "=" * 70)
    print("CONSISTENCY CHECK COMPLETE")
    print("=" * 70)
    return True

if __name__ == "__main__":
    try:
        success = check_deck_consistency()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"FATAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
