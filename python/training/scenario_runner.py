import os
import sys
import numpy as np
import dm_ai_module
from training.scenario_definitions import get_scenario_config

# Helper to run a scenario
class ScenarioRunner:
    def __init__(self, card_db):
        self.card_db = card_db

    def run_scenario(self, scenario_name, agent, max_turns=20):
        config = get_scenario_config(scenario_name)

        # Initialize GameInstance
        # Seed doesn't matter much for deterministic scenario start, but matters for RNG later
        gi = dm_ai_module.GameInstance(np.random.randint(100000), self.card_db)
        gi.reset_with_scenario(config)

        state = gi.state

        # Initialize Stats Tracker for Loop Detection
        state_hashes = []
        loop_counter = 0

        for _ in range(max_turns * 2): # *2 because turns are shared or steps?
            # Check Game Over
            if state.winner != dm_ai_module.GameResult.NONE:
                return f"GAME_OVER: {state.winner}"

            # Loop Detection
            current_hash = state.calculate_hash()
            if state_hashes and current_hash == state_hashes[-1]:
                # Immediate repetition (e.g. invalid move loop?) - Engine should prevent this usually
                pass

            # Check for 3-fold repetition
            if state_hashes.count(current_hash) >= 2: # Currently seeing it for the 3rd time
                if config.loop_proof_mode:
                     return "LOOP_PROVEN"
                else:
                     # In normal game, draw?
                     return "DRAW_LOOP"

            state_hashes.append(current_hash)

            # Agent Move
            # We need to wrap state in list for batch API
            # Assuming agent is a function that takes [state] and returns ([policy], [value])
            # Or use MCTS directly

            # For simplicity, let's assume agent is an object with `select_action(state)` or similar
            # If agent is None, use random
            if agent is None:
                # Random legal action
                # We need to generate legal actions using ActionGenerator
                # But we don't have direct access to `ActionGenerator` instance easily from Python
                # unless we use `ActionGenerator.generate_legal_actions(state)`
                legal_actions = dm_ai_module.ActionGenerator.generate_legal_actions(state, self.card_db)
                if not legal_actions:
                    # Pass or lose?
                    dm_ai_module.PhaseManager.next_phase(state) # Force next phase if no actions?
                    continue

                action = legal_actions[np.random.randint(len(legal_actions))]
                dm_ai_module.EffectResolver.resolve_action(state, action, self.card_db)

            else:
                # Use Agent
                # TODO: Implement agent integration
                pass

            # Advance phase/turn logic is handled by `EffectResolver.resolve_action` and `PhaseManager`?
            # Actually `EffectResolver` handles the action. `PhaseManager` handles transition.
            # We need to loop until game over.

            # Check if turn changed?

        return "TIMEOUT"

if __name__ == "__main__":
    # Test
    try:
        # Load DB
        # Assuming run from root
        db_path = "data/cards.csv"
        if not os.path.exists(db_path):
            print("data/cards.csv not found, using empty db")
            card_db = {}
        else:
            card_db = dm_ai_module.CsvLoader.load_cards(db_path)

        runner = ScenarioRunner(card_db)
        result = runner.run_scenario("lethal_puzzle_easy", None)
        print(f"Scenario Result: {result}")

    except Exception as e:
        print(f"Error: {e}")
