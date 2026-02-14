import dm_ai_module
import os
import sys

# Ensure current directory and dm_toolkit are in path for module import
sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), 'dm_toolkit'))

def run_reproduction():
    print("Initializing Game for Card ID 1 Reproduction...")
    try:
        from dm_ai_module import GameInstance, JsonLoader, PhaseManager, Phase, ScenarioConfig, PlayerIntent
    except ImportError:
        print("Error: Could not import dm_ai_module. Ensure you are running from project root.")
        return

    game = GameInstance(42)
    card_db = JsonLoader.load_cards("data/cards.json")
    
    # Setup - Player 0 has Card ID 1 in hand and enough    # Card ID 1 cost is 2, Water.
    # We need 2 Water mana.
    # We can use card ID 1 itself as mana if it is Water.
    # my_mana_zone expects a list of card IDs. 
    # Let's assume Card ID 1 is Water (Aqua Hulk).
    config = ScenarioConfig()
    config.my_hand_cards = [1, 2, 3, 4, 5] 
    config.my_mana_zone = [1, 1] # 2 Water mana
    config.my_deck = [2] * 10
    config.enemy_hand_cards = [1]
    config.enemy_deck = [1] * 10
    
    PhaseManager.setup_scenario(game.state, config, card_db)

    # Force Main Phase
    game.state.current_phase = Phase.MAIN
    game.state.active_player_id = 0
    
    print(f"Initial Hand Size: {len(game.state.players[0].hand)}")
    print(f"Initial Mana: {len(game.state.players[0].mana_zone)}")
    
    # Generate Actions
    actions = dm_ai_module.IntentGenerator.generate_legal_actions(game.state, card_db)
    
    play_action = None
    print(f"Total Actions Generated: {len(actions)}")
    for a in actions:
        print(f"Action: Type={a.type}, Source={a.source_instance_id}, CardID={a.card_id}")
        if a.type == PlayerIntent.PLAY_CARD or a.type == PlayerIntent.DECLARE_PLAY:
            # We want to play card ID 1. We need to check the source card ID.
            # a.source_instance_id gives us the instance ID. We can look it up in the state.
            card_instance = next((c for c in game.state.players[0].hand if c.instance_id == a.source_instance_id), None)
            if card_instance and card_instance.card_id == 1:
                play_action = a
                break
    
    if play_action:
        print(f"Found Play Action for Card ID 1 (Instance {play_action.source_instance_id})")
        game.resolve_action(play_action)
        print("Action Resolved.")
    else:
        print("FAIL: Could not find action to play Card ID 1")
        return

    # Verify State
    hand_size = len(game.state.players[0].hand)
    graveyard_size = len(game.state.players[0].graveyard)
    deck_size = len(game.state.players[0].deck)
    battle_zone_size = len(game.state.players[0].battle_zone)
    
    print(f"Final Hand Size: {hand_size}")
    print(f"Final Graveyard Size: {graveyard_size}")
    print(f"Final Deck Size: {deck_size}")
    print(f"Final Battle Zone Size: {battle_zone_size}")
    print(f"Final Stack Size: {len(game.state.players[0].stack)}")

    # Expected behavior for Card ID 1 (Oboro Kagerou):
    # 1. On Play: Mana Charge Count (2) -> Draw 2 cards.
    # 2. Put 2 cards from hand to bottom of deck.
    # Net change in hand size: -1 (played card) + 2 (draw) - 2 (put back) = -1.
    # Initial hand: 5. Final hand should be 4.
    # Played card should be in Battle Zone (creature).
    
    if battle_zone_size != 1:
        print("ERROR: Card ID 1 (Creature) not in Battle Zone.")
        
    if hand_size == 0 and graveyard_size > 0:
        print("BUG REPRODUCED: Hand is empty and cards are in graveyard!")
    elif hand_size == 4:
         print("SUCCESS: Hand size is correct (4).")
    else:
         print(f"UNEXPECTED STATE: Hand size {hand_size} (Expected 4).")

if __name__ == "__main__":
    run_reproduction()
