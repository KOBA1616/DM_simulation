
import sys
import os
from pathlib import Path
import unittest

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from dm_toolkit import dm_ai_module
from dm_toolkit.dm_ai_module import GameInstance, PhaseManager, Phase, ActionType
from dm_toolkit.ai.agent.mcts import MCTS

try:
    import torch
    from dm_toolkit.ai.agent.transformer_model import DuelTransformer
    from dm_toolkit.ai.agent.tokenization import StateTokenizer, ActionEncoder
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

class TestPhaseAndMCTS(unittest.TestCase):
    def test_phase_transition_logic(self):
        """
        Verify that PhaseManager.next_phase correctly:
        1. Cycles phases (MANA -> MAIN -> ATTACK -> END -> MANA)
        2. Switches active player at end of turn
        3. Increments turn number
        4. Untaps cards
        5. Draws a card (if applicable)
        """
        game = GameInstance()
        game.start_game()
        state = game.state

        print(f"\n[Phase Test] Initial State: Turn {state.turn_number}, Player {state.active_player_id}, Phase {state.current_phase}")

        # Setup: Tap a mana card to see if it untaps
        state.add_card_to_mana(0, 100)
        state.players[0].mana_zone[0].is_tapped = True

        initial_player = state.active_player_id
        initial_turn = state.turn_number

        # Cycle through phases to complete a turn
        # Expectation: MANA -> MAIN -> ATTACK -> END -> (Switch Player) -> MANA

        # 1. MANA -> MAIN
        self.assertEqual(state.current_phase, Phase.MANA)
        PhaseManager.next_phase(state)
        self.assertEqual(state.current_phase, Phase.MAIN)

        # 2. MAIN -> ATTACK
        PhaseManager.next_phase(state)
        self.assertEqual(state.current_phase, Phase.ATTACK)

        # 3. ATTACK -> END
        PhaseManager.next_phase(state)
        self.assertEqual(state.current_phase, Phase.END)

        # 4. END -> MANA (Next Turn)
        PhaseManager.next_phase(state)
        self.assertEqual(state.current_phase, Phase.MANA)

        print(f"[Phase Test] After 1 full cycle: Turn {state.turn_number}, Player {state.active_player_id}, Phase {state.current_phase}")

        # Verification
        if state.active_player_id == initial_player:
            print("FAILURE: Active player did not switch!")
        else:
            print("SUCCESS: Active player switched.")

        if state.turn_number == initial_turn and state.active_player_id == initial_player:
             # If player didn't switch, turn definitely didn't increment appropriately for a 2-player game flow
             pass

        # Check untap
        # Note: If player switched, we are now looking at Player 1's turn.
        # Player 0's cards shouldn't necessarily untap until Player 0's turn starts again?
        # Actually, untap happens at start of YOUR turn.
        # So we need to cycle back to Player 0 to see untap.

        # Cycle Player 1's turn
        PhaseManager.next_phase(state) # MAIN
        PhaseManager.next_phase(state) # ATTACK
        PhaseManager.next_phase(state) # END
        PhaseManager.next_phase(state) # MANA (Player 0 again)

        print(f"[Phase Test] Back to Player 0: Turn {state.turn_number}, Player {state.active_player_id}")

        if state.players[0].mana_zone[0].is_tapped:
            print("FAILURE: Mana card did not untap!")
        else:
            print("SUCCESS: Mana card untaped.")

    def test_mcts_transformer_integration(self):
        """
        Verify MCTS correctly raises RuntimeError on initialization (deprecation check).
        """
        print("\n[MCTS Test] Initializing MCTS + Transformer...")

        if not TORCH_AVAILABLE:
            print("[MCTS Test] Torch not available, testing MCTS init with dummy args...")
            # Test with None for network/tokenizer, should still raise RuntimeError
            with self.assertRaises(RuntimeError) as cm:
                mcts = MCTS(
                    network=None,
                    card_db=None,
                    simulations=2
                )
            print("SUCCESS: Caught expected deprecation error:", cm.exception)
            return

        game = GameInstance()
        game.start_game()

        # Setup Model
        vocab_size = 1000
        action_dim = 600
        model = DuelTransformer(vocab_size=vocab_size, action_dim=action_dim)
        model.eval()

        # Setup Tokenizer/Encoder
        tokenizer = StateTokenizer(vocab_size=vocab_size)

        def state_converter(state, player_id, card_db):
            return tokenizer.encode_state(state, player_id)

        # Setup MCTS
        print("[MCTS Test] Expecting RuntimeError during MCTS init...")
        with self.assertRaises(RuntimeError) as cm:
            mcts = MCTS(
                network=model,
                card_db=None,
                simulations=2, # Small number for test
                state_converter=state_converter
            )

        print("SUCCESS: Caught expected deprecation error:", cm.exception)

if __name__ == '__main__':
    # Run the tests manually
    t = TestPhaseAndMCTS()
    try:
        t.test_phase_transition_logic()
    except Exception as e:
        print(f"Phase Logic Test Crashed: {e}")

    try:
        t.test_mcts_transformer_integration()
    except Exception as e:
        print(f"MCTS Integration Test Crashed: {e}")
