
import os
import sys

# Ensure current directory is in path
sys.path.append(os.getcwd())
# sys.path.append(os.path.join(os.getcwd(), 'dm_toolkit')) # Removed to avoid conflict

def run_reproduction():
    print("Initializing Game for Card ID 1 Reproduction...")
    # Check if already imported
    if 'dm_ai_module' in sys.modules:
        print("dm_ai_module already in sys.modules")
        import dm_ai_module
    else:
        try:
            import dm_ai_module
        except ImportError:
            # Try adding dm_toolkit if root fails, but be careful
            print("Error: Could not import dm_ai_module from root.")
            return

    from dm_ai_module import GameInstance, JsonLoader, PhaseManager, IntentGenerator, PlayerIntent, Phase, ScenarioConfig, Action

    game = GameInstance(42)
    card_db = JsonLoader.load_cards("data/cards.json")
    
    # HACK: Set Card 1 cost to 0 to bypass PAY_COST issues
    if 1 in card_db:
        print("HACK: Setting Card 1 cost to 0")
        card_db[1].cost = 0
    
    print("\n--- Game Start ---")
    
    # 2. Setup Scenario
    # Card ID 1: Oboro Kagerou
    print("Setting up scenario...")
    config = ScenarioConfig()
    # Player 0: Has Card 1 in hand, and some other cards
    # Use Card ID 5 (Water Creature) for mana to ensure Water Civ
    config.my_hand_cards = [1, 2, 3, 4] 
    config.my_mana_zone = [5, 5, 5] # 3 Water cards in Mana
    config.my_deck = [2, 3, 4, 5, 6, 7, 8, 9] * 3
    
    config.enemy_hand_cards = [1]
    config.enemy_deck = [1] * 20
    
    try:
        PhaseManager.setup_scenario(game.state, config, card_db)
    except Exception as e:
        print(f"Error in setup_scenario: {e}")
        return

    # Force MAIN phase
    print("Forcing MAIN phase...")
    game.state.current_phase = Phase.MAIN
    game.state.active_player_id = 0
    game.state.turn_number = 1
    
    # Verify Mana zone
    print(f"Player 0 Hand: {[c.card_id for c in game.state.players[0].hand]}")
    print(f"Player 0 Mana: {[c.card_id for c in game.state.players[0].mana_zone]}")
    for c in game.state.players[0].mana_zone:
        # print(f"  Mana Card {c.card_id} Instance: {c.instance_id} Tapped: {c.is_tapped}")
        c.is_tapped = False # Ensure they are untrapped
    
    # 3. Find Action to Play Card ID 1
    print("Generating legal actions...")
    actions = IntentGenerator.generate_legal_actions(game.state, card_db)
    play_action = None
    for a in actions:
        if a.type == PlayerIntent.DECLARE_PLAY:
            players = game.state.players
            # a.source_instance_id should be in hand
            for c in players[0].hand:
                if c.instance_id == a.source_instance_id and c.card_id == 1:
                    play_action = a
                    print(f"Found PlayAction: src={a.source_instance_id} for CardID=1")
                    break
        if play_action: break
    
    if not play_action:
        print("Error: Could not find action to play Card ID 1.")
        print("Legal actions:")
        for a in actions:
           print(f"  {a.type} src={a.source_instance_id} target={a.target_instance_id}")
        return

    print(f"Executing Play Action for Card ID 1 (Instance {play_action.source_instance_id})...")
    game.resolve_action(play_action)
    
    # Handle auto-resolutions (Resolve Play, etc)
    max_loops = 10
    loops = 0
    while loops < max_loops:
        actions = IntentGenerator.generate_legal_actions(game.state, card_db)
        if not actions:
            print("No more actions.")
            break
            
        first_action = actions[0]
        print(f"Next Action: {first_action.type} SlotIndex={first_action.slot_index}")
        
        if first_action.type == PlayerIntent.PAY_COST:
             print("Error: Still getting PAY_COST despite 0 cost!")
             game.resolve_action(first_action)
             loops += 1
        elif first_action.type == PlayerIntent.RESOLVE_PLAY:
             print("Auto-resolving RESOLVE_PLAY...")
             game.resolve_action(first_action)
             loops += 1
        else:
             break
    
    if loops >= max_loops:
        print("Warning: Reached max loops in auto-resolution.")

    # Check for Optional Draw (SELECT_NUMBER or similar)
    # actions is already updated from loop end
    print(f"Legal Actions after Play Resolution ({len(actions)}):")
    
    selection_action = None
    for a in actions:
        print(f"  Type: {a.type} SlotIndex: {a.slot_index} Target: {a.target_instance_id}")
        if a.type in [PlayerIntent.SELECT_NUMBER, PlayerIntent.SELECT_OPTION]:
             selection_action = a
        
    if selection_action:
        print(f"Selecting action (Draw Count): SlotIndex={selection_action.slot_index}")
        game.resolve_action(selection_action)
    else:
        print("No selection actions found. Did it skip optional draw?")
        
    # Now check for Hand Selection (Transition to Bottom Deck)
    print("\n--- After Draw, checking for Hand Selection ---")
    actions = IntentGenerator.generate_legal_actions(game.state, card_db)
    print(f"Legal Actions for Hand Selection ({len(actions)}):")
    
    # Check if actions target cards in hand
    targets_hand = False
    targets_self = False
    
    for a in actions:
        print(f"  Type: {a.type} Source: {a.source_instance_id} Target: {a.target_instance_id}")
        if a.target_instance_id != -1:
            # Check owner/zone
            hand_ids = [c.instance_id for c in game.state.players[0].hand]
            battle_ids = [c.instance_id for c in game.state.players[0].battle_zone]
            
            if a.target_instance_id in hand_ids:
                targets_hand = True
            if a.target_instance_id in battle_ids:
                targets_self = True

    if targets_hand:
        print("SUCCESS: Actions are targeting cards in HAND.")
    elif targets_self:
        print("FAILURE: Actions are targeting cards in BATTLE_ZONE (Self).")
    elif len(actions) == 0:
        print("FAILURE: No actions available.")
    else:
        print("FAILURE: No relevant targets found or targeting unknown zone.")

    print("Reproduction script finished.")

if __name__ == "__main__":
    run_reproduction()
