
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
# Conditional imports for Torch-dependent modules
try:
    import torch
    from dm_toolkit.ai.agent.transformer_model import DuelTransformer
    HAS_TORCH = True
except ImportError:
    torch = None
    HAS_TORCH = False
    DuelTransformer = None

from dm_toolkit.ai.agent.tokenization import StateTokenizer, ActionEncoder

class TestPhaseAndMCTS(unittest.TestCase):
    def test_phase_transition_logic(self):
        """
        Verify that PhaseManager.next_phase correctly cycles phases.
        """
        game = GameInstance()
        game.start_game()
        state = game.state

        print(f"\n[Phase Test] Initial State: Turn {state.turn_number}, Player {state.active_player_id}, Phase {state.current_phase}")

        # Setup: Tap a mana card to see if it untaps
        # Check if add_card_to_mana is available or use alternate
        try:
            if hasattr(state, 'add_card_to_mana'):
                state.add_card_to_mana(0, 100)
                if state.players[0].mana_zone:
                    state.players[0].mana_zone[0].is_tapped = True
        except Exception:
            pass

        initial_player = state.active_player_id

        # Cycle through phases to complete a turn
        # Expectation: MANA -> MAIN -> ATTACK -> END -> (Switch Player) -> MANA

        # 1. MANA -> MAIN
        self.assertEqual(int(state.current_phase), int(Phase.MANA))
        PhaseManager.next_phase(state, {})
        self.assertEqual(int(state.current_phase), int(Phase.MAIN))

        # 2. MAIN -> ATTACK
        PhaseManager.next_phase(state, {})
        self.assertEqual(int(state.current_phase), int(Phase.ATTACK))

        # 3. ATTACK -> END
        PhaseManager.next_phase(state, {})
        self.assertEqual(int(state.current_phase), int(Phase.END))

        # 4. END -> MANA (Next Turn)
        PhaseManager.next_phase(state, {})
        self.assertEqual(int(state.current_phase), int(Phase.MANA))

        print(f"[Phase Test] After 1 full cycle: Turn {state.turn_number}, Player {state.active_player_id}, Phase {state.current_phase}")

        self.assertNotEqual(state.active_player_id, initial_player)

    @unittest.skipUnless(HAS_TORCH, "Torch not available")
    def test_mcts_transformer_integration(self):
        """
        Verify MCTS initializes and runs (Restored).
        """
        print("\n[MCTS Test] Initializing MCTS + Transformer...")
        game = GameInstance()
        game.start_game()
        state = game.state

        # Setup Model
        vocab_size = 1000
        action_dim = 600
        model = DuelTransformer(vocab_size=vocab_size, action_dim=action_dim)
        model.eval()

        # Setup Tokenizer/Encoder
        tokenizer = StateTokenizer(vocab_size=vocab_size)

        def state_converter(state, player_id, card_db):
            return tokenizer.encode_state(state, player_id)

        print("[MCTS Test] Instantiating MCTS (Should succeed)...")
        mcts = MCTS(
            network=model,
            card_db={},
            simulations=2,
            state_converter=state_converter
        )
        print("SUCCESS: MCTS Instantiated.")

        # Verify search (basic run)
        try:
            best_action = mcts.search(state)
            print(f"MCTS Search Result: {best_action}")
        except Exception as e:
            self.fail(f"MCTS Search failed: {e}")

if __name__ == '__main__':
    unittest.main()
