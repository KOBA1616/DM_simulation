
import sys
import os
import torch
import unittest
from pathlib import Path
import random

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import copy
try:
    import dm_ai_module
    from dm_toolkit.ai.agent.mcts import MCTS
    from training.ai_player import AIPlayer
    from dm_toolkit.engine.compat import EngineCompat
except ImportError as e:
    print(f"Import Error: {e}")
    sys.exit(1)

# Mock Network for MCTS (AlphaZero style, expects tensor input)
class MockAlphaZeroNetwork(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        batch_size = x.shape[0]
        # Policy: 600 logits
        policy = torch.randn(batch_size, 600)
        # Value: 1 scalar
        value = torch.tanh(torch.randn(batch_size, 1))
        return policy, value

# Diagnostic Script
def run_diagnosis():
    print("Initializing Game...")
    game = dm_ai_module.GameInstance()
    game.start_game()

    # Setup MCTS Agent (Player 0)
    print("Setting up MCTS (P0)...")
    mcts_net = MockAlphaZeroNetwork()
    # Assuming card_db is available or empty dict is fine for basic logic
    card_db = {}
    mcts_agent = MCTS(network=mcts_net, card_db=card_db, simulations=10)

    # Setup AIPlayer (Player 1) - Transformer
    print("Setting up Transformer (P1)...")
    # We rely on AIPlayer's internal random init if model_path is missing
    ai_player = AIPlayer(model_path="non_existent.pth", device='cpu')

    print("Starting Game Loop...")

    max_turns = 20
    turn_count = 0
    consecutive_passes = 0

    while not game.state.game_over and turn_count < max_turns:
        active_pid = game.state.active_player_id
        phase = game.state.current_phase
        turn = game.state.turn_number

        print(f"Turn: {turn}, Phase: {phase}, Active Player: {active_pid}")

        command = None

        if active_pid == 0:
            # MCTS
            print("  MCTS Thinking...")
            try:
                root = mcts_agent.search(game.state)
                # Pick best child
                best_child = None
                best_visit = -1
                for child in root.children:
                    if child.visit_count > best_visit:
                        best_visit = child.visit_count
                        best_child = child

                if best_child:
                    command = best_child.action
                    cmd_str = str(command)
                    if hasattr(command, 'to_dict'):
                        cmd_str = str(command.to_dict())
                    print(f"  MCTS Selected: {cmd_str}")
                else:
                    print("  MCTS returned no children. Forcing Pass/EndTurn.")
                    # If MCTS fails, we might be stuck.
            except Exception as e:
                print(f"  MCTS Crash: {e}")
                break
        else:
            # Transformer
            print("  Transformer Thinking...")
            try:
                command = ai_player.get_action(game.state, active_pid)
                print(f"  Transformer Selected: {command}")
            except Exception as e:
                print(f"  Transformer Crash: {e}")
                break

        if command:
            prev_phase = game.state.current_phase
            p0_mana = len(game.state.players[0].mana_zone)

            # Execute
            try:
                # Handle different command types (Action vs dict vs ICommand)
                if hasattr(command, 'execute'):
                    command.execute(game.state)
                elif hasattr(command, 'type'): # GameCommand or Action
                    # AIPlayer returns GameCommand
                    # MCTS might return Action or GameCommand
                    from dm_toolkit.compat_wrappers import execute_action_compat
                    execute_action_compat(game.state, command, card_db)
                else:
                    # Dict?
                    print(f"  Unknown command type: {type(command)}")
            except Exception as e:
                print(f"  Execution Error: {e}")

            # Check Phase progression
            new_phase = game.state.current_phase
            new_p0_mana = len(game.state.players[0].mana_zone)
            if new_p0_mana != p0_mana:
                print(f"  Mana Changed: {p0_mana} -> {new_p0_mana}")

            # Manual Phase Advance (Auto Pass logic)
            # GameSession handles this, so we must too for verification.
            cmd_type = None
            if hasattr(command, 'to_dict'):
                cmd_type = command.to_dict().get('type')
            elif hasattr(command, 'type'):
                cmd_type = str(command.type)

            if cmd_type in ['PASS', 'MANA_CHARGE', 'ActionType.PASS', 'ActionType.MANA_CHARGE']:
                 # If MANA_CHARGE, we usually advance immediately in this simplified loop
                 # In real game, you can only charge once, then you MUST pass/advance.
                 # But if we don't force advance, we stick in Phase 2.
                 print(f"  Auto-Advancing Phase from {new_phase}")
                 dm_ai_module.PhaseManager.next_phase(game.state, card_db)
                 new_phase = game.state.current_phase
                 print(f"  New Phase: {new_phase}")

            # Detect Stuck in Phase (e.g. Main Phase loop)
            if new_phase == prev_phase and new_phase == dm_ai_module.Phase.MAIN:
                # Assuming PASS advances Main -> Attack/End
                # If command was PASS, it should have changed.
                pass

            # Special check: Did MCTS/AI generate PASS in Main Phase?
            # If so, did it change phase?

        else:
            print("  No command generated. Stalled.")
            break

        # Safety break for infinite loops within a turn
        # (Simplified: just counting total loop iterations as 'turns' is wrong, but good enough for crash test)
        turn_count += 1

    print("Diagnosis Complete.")

if __name__ == "__main__":
    run_diagnosis()
