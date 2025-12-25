import os
import sys
import numpy as np
import torch
import time
from typing import List, Tuple, Optional, Any

# Ensure bin is in path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
bin_path = os.path.join(project_root, 'bin')
if bin_path not in sys.path:
    sys.path.append(bin_path)

try:
    import dm_ai_module
except ImportError:
    print(f"Error: Could not import dm_ai_module. Make sure it is built and in {bin_path}")
    sys.exit(1)

from dm_toolkit.ai.agent.network import AlphaZeroNetwork
from dm_toolkit.types import CardDB, ResultsList

class SelfPlayRunner:
    def __init__(self, card_db: CardDB, device=None):
        self.card_db: CardDB = card_db
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize dimensions
        # GameInstance not exposed, using GameState directly
        dummy_state = dm_ai_module.GameState(len(self.card_db))
        # Note: instance.state returns a copy, but checking dimensions is fine
        dummy_vec = dm_ai_module.TensorConverter.convert_to_tensor(dummy_state, 0, self.card_db)
        self.input_size = len(dummy_vec)
        self.action_size = dm_ai_module.ActionEncoder.TOTAL_ACTION_SIZE

    def load_model(self, model_path: str):
        network = AlphaZeroNetwork(self.input_size, self.action_size).to(self.device)
        if model_path and os.path.exists(model_path):
            print(f"Loading model: {model_path}")
            network.load_state_dict(torch.load(model_path, map_location=self.device))
        else:
            print("Warning: Model path not found or None, using random initialization.")
        network.eval()
        return network

    def _prepare_initial_states(self, num_games: int, deck_card_id: int):
        states = []
        if deck_card_id == 0:
             print("Error: No valid card ID found for deck construction.")
             return []

        print(f"Constructing decks using CardID: {deck_card_id}")

        for i in range(num_games):
            # Create GameInstance just to get a seeded state logic if needed,
            # but we can also just create GameState directly.
            # GameInstance initializes RNG seed.
            # instance = dm_ai_module.GameInstance(int(time.time() * 1000 + i) % 1000000, self.card_db)
            state = dm_ai_module.GameState(len(self.card_db))

            # Capture the COPY of the state
            # state = instance.state

            # Setup decks using helper method (direct list modification won't work due to copy)
            deck_list = [deck_card_id] * 40
            state.set_deck(0, deck_list)
            state.set_deck(1, deck_list)

            # Initialize card stats
            state.initialize_card_stats(self.card_db, 40)

            # Start the game (modifies 'state')
            dm_ai_module.PhaseManager.start_game(state, self.card_db)

            # Verify game is not over immediately
            if state.winner != dm_ai_module.GameResult.NONE:
                 print("Warning: Game started in finished state!")

            states.append(state)
        return states

    def collect_data(self, model_path: str, output_path: str, episodes=10, sims=800, batch_size=32, threads=4):
        network = self.load_model(model_path)

        # Find a valid creature for dummy deck
        valid_cid = 0
        for cid, defn in self.card_db.items():
            if defn.type == dm_ai_module.CardType.CREATURE:
                valid_cid = cid
                break

        initial_states = self._prepare_initial_states(episodes, valid_cid)
        if not initial_states:
             print("Failed to prepare initial states.")
             return

        runner = dm_ai_module.ParallelRunner(self.card_db, sims, batch_size)

        # Evaluator callback
        def evaluator(states: List[dm_ai_module.GameState]):
            tensors = []
            for s in states:
                # Assuming s.active_player_id is valid
                t = dm_ai_module.TensorConverter.convert_to_tensor(s, s.active_player_id, self.card_db, True)
                tensors.append(t)

            if not tensors:
                 return [], []

            input_tensor = torch.tensor(np.array(tensors), dtype=torch.float32).to(self.device)

            with torch.no_grad():
                policy_logits, values = network(input_tensor)
                policies = torch.softmax(policy_logits, dim=1).cpu().numpy()
                vals = values.squeeze(1).cpu().numpy()

            return policies, vals

        print(f"Starting collection: {episodes} episodes, {sims} sims, {threads} threads")
        start_time = time.time()

        try:
            results = runner.play_games(initial_states, evaluator, 1.0, True, threads)
        except Exception as e:
            print(f"ParallelRunner failed: {e}")
            return

        duration = time.time() - start_time
        print(f"Collection finished in {duration:.2f}s")

        # Process results
        all_masked = []
        all_full = []
        all_policies = []
        all_values = []

        win_counts = {1:0, 2:0, 0:0, 3:0}

        for info in results:
            res = info.result
            win_counts[res] = win_counts.get(res, 0) + 1

            final_value_p0 = 0.0
            if res == 1: final_value_p0 = 1.0
            elif res == 2: final_value_p0 = -1.0
            # Draw = 0.0

            # Info.states is list of GameStates encountered during the game
            for i, state in enumerate(info.states):
                pid = state.active_player_id

                # Value for this state
                val = final_value_p0 if pid == 0 else -final_value_p0

                # Policy
                pol = info.policies[i]

                # Convert State
                masked = dm_ai_module.TensorConverter.convert_to_tensor(state, pid, self.card_db, True)
                full = dm_ai_module.TensorConverter.convert_to_tensor(state, pid, self.card_db, False)

                all_masked.append(masked)
                all_full.append(full)
                all_policies.append(pol)
                all_values.append(val)

        print(f"Results: {win_counts}")
        print(f"Samples collected: {len(all_masked)}")

        if len(all_masked) > 0:
            np.savez_compressed(output_path,
                states_masked=np.array(all_masked, dtype=np.float32),
                states_full=np.array(all_full, dtype=np.float32),
                policies=np.array(all_policies, dtype=np.float32),
                values=np.array(all_values, dtype=np.float32)
            )
            print(f"Saved to {output_path}")

    def evaluate_matchup(self, model1_path, model2_path, episodes=10, sims=800, batch_size=32, threads=4):
        # Model 1 plays as P1 (Player 0), Model 2 as P2 (Player 1)
        net1 = self.load_model(model1_path)
        net2 = self.load_model(model2_path)

        valid_cid = 0
        for cid, defn in self.card_db.items():
            if defn.type == dm_ai_module.CardType.CREATURE:
                valid_cid = cid
                break

        initial_states = self._prepare_initial_states(episodes, valid_cid)
        runner = dm_ai_module.ParallelRunner(self.card_db, sims, batch_size)

        def evaluator(states: List[dm_ai_module.GameState]):
            # Split by active player
            indices_p1 = []
            states_p1 = []
            indices_p2 = []
            states_p2 = []

            for i, s in enumerate(states):
                if s.active_player_id == 0:
                    indices_p1.append(i)
                    states_p1.append(s)
                else:
                    indices_p2.append(i)
                    states_p2.append(s)

            # Inference P1
            results_p1_pol: Any = []
            results_p1_val: Any = []
            if states_p1:
                tensors = [dm_ai_module.TensorConverter.convert_to_tensor(s, 0, self.card_db, True) for s in states_p1]
                t_in = torch.tensor(np.array(tensors), dtype=torch.float32).to(self.device)
                with torch.no_grad():
                    logits, vals = net1(t_in)
                    results_p1_pol = torch.softmax(logits, dim=1).cpu().numpy()
                    results_p1_val = vals.squeeze(1).cpu().numpy()

            # Inference P2
            results_p2_pol: Any = []
            results_p2_val: Any = []
            if states_p2:
                tensors = [dm_ai_module.TensorConverter.convert_to_tensor(s, 1, self.card_db, True) for s in states_p2]
                t_in = torch.tensor(np.array(tensors), dtype=torch.float32).to(self.device)
                with torch.no_grad():
                    logits, vals = net2(t_in)
                    results_p2_pol = torch.softmax(logits, dim=1).cpu().numpy()
                    results_p2_val = vals.squeeze(1).cpu().numpy()

            # Reassemble
            final_policies = [None] * len(states)
            final_values = [None] * len(states)

            for k, original_idx in enumerate(indices_p1):
                final_policies[original_idx] = results_p1_pol[k]
                final_values[original_idx] = results_p1_val[k]

            for k, original_idx in enumerate(indices_p2):
                final_policies[original_idx] = results_p2_pol[k]
                final_values[original_idx] = results_p2_val[k]

            return final_policies, final_values

        print(f"Evaluating Matchup: {episodes} games")
        results = runner.play_games(initial_states, evaluator, 1.0, False, threads)

        p1_wins = 0
        for info in results:
            if info.result == 1: p1_wins += 1

        win_rate = p1_wins / episodes
        print(f"Matchup Result: P1 Wins {p1_wins}/{episodes} ({win_rate*100:.1f}%)")
        return win_rate

if __name__ == "__main__":
    # Test script
    cards_path = os.path.join(project_root, 'data', 'cards.json')
    db = dm_ai_module.JsonLoader.load_cards(cards_path)
    sp = SelfPlayRunner(db)

    # sp.collect_data(None, "test_data.npz", episodes=2, sims=50)
