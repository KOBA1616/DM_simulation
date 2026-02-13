import dm_ai_module
import os
import sys

# Ensure current directory and dm_toolkit are in path for module import
sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), 'dm_toolkit'))

def run_reproduction():
    print("Initializing Game for Reproduction...")
    try:
        import dm_ai_module
        print(f"Loaded dm_ai_module from: {dm_ai_module.__file__}")
        print(f"Dir(dm_ai_module): {dir(dm_ai_module)}")
        from dm_ai_module import GameInstance, JsonLoader
    except ImportError:
        print("Error: Could not import dm_ai_module. Ensure you are running from project root.")
        return

    game = GameInstance(42)
    card_db = JsonLoader.load_cards("data/cards.json")
    
    print("\n--- Game Start ---")
    
    # 2. Setup Scenario
    print("Setting up scenario...")
    # Configure scenario to ensure cards are in hand for mana charge
    config = dm_ai_module.ScenarioConfig()
    config.my_hand_cards = [1, 2, 3, 4, 5] 
    config.enemy_hand_cards = [1, 2, 3, 4, 5]
    config.my_deck = [1] * 20
    config.enemy_deck = [1] * 20
    
    try:
        dm_ai_module.PhaseManager.setup_scenario(game.state, config, card_db)
    except Exception as e:
        print(f"Error in setup_scenario: {e}")
        return

    # 3. Start Game Override - Do NOT call start_game() as it resets state
    # Instead, ensure we are in MANA phase for testing
    print("Forcing MANA phase for reproduction...")
    game.state.current_phase = dm_ai_module.Phase.MANA
    game.state.active_player_id = 0
    game.state.turn_number = 1
    
    # Verify State
    print(f"Player 0 Hand: {len(game.state.players[0].hand)}")
    print(f"Player 0 Mana: {len(game.state.players[0].mana_zone)}")

    # 4. Game Loop
    for turn in range(1, 4):
        print(f"\n--- Turn {turn} ---")
        
        # Log Mana Status Start
        p0_mana = len(game.state.players[0].mana_zone)
        p1_mana = len(game.state.players[1].mana_zone)
        print(f"Start Mana: P0={p0_mana}, P1={p1_mana}")
        
        start_turn_num = game.state.turn_number
        
        while game.state.turn_number == start_turn_num:
            current_phase = game.state.current_phase
            active_player = game.state.active_player_id
            print(f"Phase: {current_phase}, Active Player: {active_player}")
            
            # Generate Actions
            try:
                actions = dm_ai_module.IntentGenerator.generate_legal_actions(game.state, card_db)
            except Exception as e:
                print(f"Error generating actions: {e}")
                break

            print(f"Legal Actions ({len(actions)}):")
            for i, a in enumerate(actions):
                print(f"  [{i}] Type: {a.type}")

            # Select Action
            selected_action = None
            
            # Priority: MANA_CHARGE > PASS (if no other moves)
            charged_this_turn = game.state.turn_stats.mana_charged_by_player[active_player]
            
            if current_phase == dm_ai_module.Phase.MANA:
                 if not charged_this_turn:
                     # Find first mana charge action
                     for a in actions:
                         if a.type == dm_ai_module.PlayerIntent.MANA_CHARGE:
                             selected_action = a
                             break
                 
                 if not selected_action:
                     # Pass if already charged or no charge possible
                     for a in actions:
                         if a.type == dm_ai_module.PlayerIntent.PASS: # PASS
                             selected_action = a
                             break
            
            elif current_phase == dm_ai_module.Phase.MAIN:
                 # Just pass to speed up test
                 for a in actions:
                     if a.type == dm_ai_module.PlayerIntent.PASS: # PASS
                         selected_action = a
                         break
                 if not selected_action and actions:
                      selected_action = actions[0]

            else:
                 # Other phases - Default first
                 if actions:
                      selected_action = actions[0]

            if selected_action:
                print(f"Executing Action: Type {selected_action.type} Source {selected_action.source_instance_id}")
                game.resolve_action(selected_action)
            else:
                print("No action selected (or no actions). Fast forwarding...")
                dm_ai_module.PhaseManager.fast_forward(game.state, card_db)
                
            # Check if turn changed to break inner loop
            if game.state.turn_number != start_turn_num:
                break
                
            # Stop if game over
            if game.state.game_over:
                break

        # Log Mana Status End
        p0_mana_end = len(game.state.players[0].mana_zone)
        p1_mana_end = len(game.state.players[1].mana_zone)
        end_str = f"End Mana: P0={p0_mana_end}, P1={p1_mana_end}"
        print(end_str)
        
        if game.state.game_over:
             print(f"Game Over! Winner: {game.state.winner}")
             break
        
    print("Reproduction script finished.")

if __name__ == "__main__":
    run_reproduction()
