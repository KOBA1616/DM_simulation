
import sys
import os
import torch
import numpy as np

# Ensure path is set up to import dm_toolkit
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
sys.path.append(project_root)

import dm_ai_module
from dm_ai_module import GameInstance, PhaseManager, Phase, ActionType
from dm_toolkit.ai.agent.transformer_model import DuelTransformer
from dm_toolkit.ai.agent.tokenization import StateTokenizer

def verify_integration():
    print("--- Starting Integration Diagnostics ---")

    # 1. Setup Game
    card_db = {} # Empty DB for fallback test
    instance = GameInstance(seed=42, card_db=card_db)
    PhaseManager.start_game(instance.state, card_db)
    print(f"Initial Phase: {instance.state.current_phase} (Expected: {Phase.MANA})")

    # 2. Setup Model
    vocab_size = 8192
    action_dim = 1024
    model = DuelTransformer(vocab_size=vocab_size, action_dim=action_dim)
    model.eval()
    print("DuelTransformer initialized.")

    # 3. Setup Tokenizer
    tokenizer = StateTokenizer(vocab_size=vocab_size, max_len=200)

    # 4. Setup MCTS with Tokenizer
    # MCTS expects state_converter(state, pid, db) -> tokens
    def state_converter_wrapper(state, pid, db):
        tokens = tokenizer.encode_state(state, pid, db)
        # Verify it returns Ints
        if len(tokens) > 0:
             if not isinstance(tokens[0], (int, np.integer)):
                 print(f"WARNING: Tokenizer returned non-int: {type(tokens[0])}")
        return tokens

    mcts = dm_ai_module.MCTS(
        network=model,
        card_db=card_db,
        simulations=5, # Small number for speed
        state_converter=state_converter_wrapper
    )
    print("MCTS initialized with Tokenizer.")

    # 5. Run Loop
    max_steps = 20
    print("\n--- Running Game Loop ---")

    # We track phases to ensure we cycle: MANA -> MAIN -> ATTACK -> END -> MANA
    phase_history = []

    for step in range(max_steps):
        current_phase = instance.state.current_phase
        active_player = instance.state.active_player_id
        phase_history.append(current_phase)

        print(f"Step {step}: Player {active_player} in Phase {current_phase.name if hasattr(current_phase, 'name') else current_phase}")

        # Run MCTS
        root = mcts.search(instance.state)
        best_child = None
        best_score = -float('inf')

        # Simple selection of best action from root
        for child in root.children:
            if child.visit_count > best_score:
                best_score = child.visit_count
                best_child = child

        if best_child:
            action = best_child.action
            action_type_name = action.type.name if hasattr(action.type, 'name') else str(action.type)
            print(f"  > Selected Action: {action_type_name} (Card: {action.card_id})")

            # Execute Action in Real State
            # Note: MCTS.search uses clones. We must apply to instance.state.

            # Logic to apply action:
            if action.type == ActionType.PASS:
                print("  > Executing PASS -> Next Phase")
                PhaseManager.next_phase(instance.state, card_db)
            elif action.type == ActionType.MANA_CHARGE:
                 print("  > Executing MANA_CHARGE")
                 # Emulate mana charge logic
                 instance.state.players[active_player].mana_zone.append(dm_ai_module.CardStub(action.card_id))
                 # Fallback generator creates dummy actions that don't remove from hand?
                 # Assuming fallback logic just appends.
            elif action.type == ActionType.PLAY_CARD:
                 print("  > Executing PLAY_CARD")
                 instance.state.players[active_player].battle_zone.append(dm_ai_module.CardStub(action.card_id))

            # If we are in END phase, Pass should trigger turn change
            if current_phase == Phase.END and action.type == ActionType.PASS:
                print("  > End of Turn -> Switching Player")

        else:
            print("  > No legal actions found (MCTS failed?)")
            break

        # Check for game over
        is_over, _ = PhaseManager.check_game_over(instance.state)
        if is_over:
            print("Game Over")
            break

    print("\n--- Phase History ---")
    print([p.name if hasattr(p, 'name') else p for p in phase_history])

    # Check for phase diversity
    unique_phases = set(phase_history)
    if len(unique_phases) >= 4:
         print("\nSUCCESS: Cycled through multiple phases.")
    else:
         print("\nWARNING: Did not cycle through all phases. Loop might be stuck.")

if __name__ == "__main__":
    verify_integration()
