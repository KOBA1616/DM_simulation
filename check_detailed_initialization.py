"""
Detailed game initialization flow check.
Traces step-by-step initialization including:
1. Initial state
2. Game start sequence
3. Phase transitions
4. Card movements
"""

import dm_ai_module

def check_detailed_initialization():
    """Trace detailed initialization flow."""
    print("=" * 70)
    print("DETAILED GAME INITIALIZATION FLOW")
    print("=" * 70)
    
    # Load card database
    try:
        card_db = dm_ai_module.JsonLoader.load_cards("data/cards.json")
        print("\n✅ Card database loaded")
    except Exception as e:
        print(f"❌ Failed to load card database: {e}")
        return False
    
    # Create GameInstance
    gi = dm_ai_module.GameInstance(42, card_db)
    gs = gi.state
    print("✅ GameInstance created")
    
    # Setup test duel
    gs.setup_test_duel()
    print("✅ setup_test_duel() executed")
    
    # Set decks
    default_deck = [1,2,3,4,5,6,7,8,9,10]*4
    gs.set_deck(0, default_deck)
    gs.set_deck(1, default_deck)
    print("✅ Decks set (40 cards each)")
    
    # Record initial state
    def print_state(title):
        print(f"\n--- {title} ---")
        print(f"Turn: {gs.turn_number}, Active: P{gs.active_player_id}, Phase: {gs.current_phase}")
        for pid in range(2):
            p = gs.players[pid]
            total = (len(p.hand) + len(p.mana_zone) + len(p.battle_zone) + 
                    len(p.shield_zone) + len(p.graveyard) + len(p.deck))
            print(f"P{pid}: Hand={len(p.hand):2d} Mana={len(p.mana_zone):2d} Battle={len(p.battle_zone):2d} " +
                  f"Shield={len(p.shield_zone):2d} Grave={len(p.graveyard):2d} Deck={len(p.deck):2d} Total={total:2d}")
    
    print_state("1. After setup_test_duel()")
    print_state("2. After set_deck()")
    
    # Start game
    print("\n--- Starting C++ Game Engine ---")
    try:
        dm_ai_module.PhaseManager.start_game(gs, card_db)
        print("✅ PhaseManager.start_game() called")
    except Exception as e:
        print(f"⚠️  start_game error: {e}")
    
    print_state("3. After start_game()")
    
    # Call fast_forward
    try:
        dm_ai_module.PhaseManager.fast_forward(gs, card_db)
        print("✅ PhaseManager.fast_forward() called")
    except Exception as e:
        print(f"⚠️  fast_forward error: {e}")
    
    print_state("4. After fast_forward()")
    
    # Try calling step() on GameInstance to advance
    print("\n--- Trying to step through C++ engine ---")
    for step_num in range(1, 6):
        try:
            result = gi.step()
            print(f"✅ Step {step_num} executed (result: {result})")
            print_state(f"5.{step_num} After step() #{step_num}")
        except Exception as e:
            print(f"⚠️  step() {step_num} failed: {e}")
            break
    
    # Final summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    final_hand_cards = sum(len(gs.players[pid].hand) for pid in range(2))
    final_mana_cards = sum(len(gs.players[pid].mana_zone) for pid in range(2))
    final_shield_cards = sum(len(gs.players[pid].shield_zone) for pid in range(2))
    final_deck_cards = sum(len(gs.players[pid].deck) for pid in range(2))
    
    print(f"\nFinal card distribution:")
    print(f"  Total Hand cards: {final_hand_cards} (Expected: 0-10)")
    print(f"  Total Mana cards: {final_mana_cards} (Expected: 0-2)")
    print(f"  Total Shield cards: {final_shield_cards} (Expected: 0-10)")
    print(f"  Total Deck cards: {final_deck_cards} (Expected: 70-80)")
    
    if final_hand_cards > 0 or final_mana_cards > 0:
        print("\n✅ Cards have been initialized (hand/mana moved from deck)")
    else:
        print("\n⚠️  No cards initialized yet (cards still in deck only)")
    
    print(f"\nGame state:")
    print(f"  Turn: {gs.turn_number}")
    print(f"  Phase: {gs.current_phase}")
    print(f"  Game over: {gs.game_over}")
    
    return True

if __name__ == "__main__":
    try:
        check_detailed_initialization()
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
