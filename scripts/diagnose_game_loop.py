
import sys
import os
import torch
import traceback

# Ensure project root is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from dm_toolkit.ai.agent.mcts import MCTS
from dm_toolkit.ai.agent.transformer_model import DuelTransformer
from dm_toolkit.ai.agent.tokenization import StateTokenizer, ActionEncoder
from dm_toolkit import dm_ai_module
from dm_toolkit.dm_ai_module import GameState, PhaseManager, Phase, GameResult

def run():
    print("--- Diagnostic Script: Game Loop Phases (Correct Configuration) ---")

    # 1. Setup Dependencies
    try:
        card_db = dm_ai_module.JsonLoader.load_cards("data/cards.json")
    except Exception:
        card_db = {}

    vocab_size = 2000
    action_dim = 600

    # 2. Instantiate Model & Tokenizer
    print("Initializing DuelTransformer & Tokenizer...")
    model = DuelTransformer(vocab_size=vocab_size, action_dim=action_dim)
    model.eval()

    tokenizer = StateTokenizer(vocab_size=vocab_size)
    encoder = ActionEncoder(action_dim=action_dim)

    # 3. Instantiate MCTS (WITH state_converter)
    print("Initializing MCTS (With Tokenizer)...")
    mcts = MCTS(
        network=model,
        card_db=card_db,
        simulations=2,
        state_converter=tokenizer.encode_state,
        action_encoder=encoder.encode_action
    )

    # 4. Create State & Start Game
    state = GameState()
    deck = [1] * 40 # Dummy deck
    state.set_deck(0, deck)
    state.set_deck(1, deck)

    print("Starting Game...")
    PhaseManager.start_game(state, card_db)

    print(f"Initial Phase: {Phase(state.current_phase).name}")

    # 5. Run Game Loop (Max 20 steps)
    max_steps = 20
    step = 0

    turn_counter = state.turn_number

    while step < max_steps:
        is_over, result = PhaseManager.check_game_over(state)
        if is_over:
            print(f"Game Over! Result: {result}")
            break

        print(f"[Step {step}] Turn {state.turn_number} - Player {state.active_player_id} - Phase: {Phase(state.current_phase).name}")

        # Verify Phase Logic
        if state.turn_number != turn_counter:
            print(f"--> Turn Advanced to {state.turn_number}")
            turn_counter = state.turn_number

        # Select Action via MCTS
        # Note: MCTS.search returns a root node. We need to pick the best child.
        try:
            root = mcts.search(state)
            best_child = None
            best_val = -float('inf')

            # Simple policy: pick most visited
            for child in root.children:
                if child.visit_count > best_val:
                    best_val = child.visit_count
                    best_child = child

            if best_child:
                action = best_child.action
                act_str = str(action)
                if hasattr(action, 'to_dict'):
                    act_str = str(action.to_dict())
                elif hasattr(action, '__dict__'):
                    act_str = str(action.__dict__)
                print(f"    AI Action: {act_str}")

                # Apply Action
                # MCTS search doesn't modify the input state, so we must apply the action to the real state
                # However, MCTS logic includes applying the action AND potentially auto-passing phases.
                # To be safe, we replicate the execute/next_phase logic here or assume MCTS verified it.

                # Execution Logic (Simplified from MCTS._expand)
                from dm_toolkit.engine.compat import EngineCompat

                # We need to execute the command/action
                executed = False
                try:
                    if hasattr(action, 'execute'):
                         action.execute(state)
                         executed = True
                    else:
                         EngineCompat.ExecuteCommand(state, action, card_db)
                         executed = True
                except Exception as e:
                    print(f"    Execution Error: {e}")

                # Check for Phase Transition explicitly if action didn't do it
                # Logic: If PASS, then next_phase.
                # Or if auto-transition logic exists.
                is_pass = False
                try:
                    # Check Enum
                    if hasattr(action, 'type') and action.type == dm_ai_module.ActionType.PASS:
                        is_pass = True
                    # Check string 'PASS'
                    elif hasattr(action, 'type') and str(action.type) == 'PASS':
                        is_pass = True
                    # Check dict
                    elif isinstance(action, dict) and action.get('type') == 'PASS':
                        is_pass = True
                    # Check wrapper
                    elif hasattr(action, 'to_dict') and action.to_dict().get('type') == 'PASS':
                        is_pass = True
                except:
                    pass

                if is_pass:
                    print("    (Passing Turn/Phase)")
                    PhaseManager.next_phase(state, card_db)

                # Also handle implicit phase changes (e.g. if mana charged in mana phase?)
                # But fallback engine is dumb, so we rely on explicit PASS usually.

            else:
                print("    AI found no moves (PASS)")
                PhaseManager.next_phase(state, card_db)

        except Exception as e:
            print(f"CRITICAL ERROR in Search: {e}")
            traceback.print_exc()
            break

        step += 1

    print("Diagnostic Loop Completed.")

if __name__ == "__main__":
    run()
