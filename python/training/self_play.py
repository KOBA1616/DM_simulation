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
if project_root not in sys.path:
    sys.path.append(project_root)

try:
    import dm_ai_module
except ImportError:
    print(f"Error: Could not import dm_ai_module. Make sure it is built and in {bin_path}")
    sys.exit(1)

from dm_toolkit.types import CardDB
from training.game_runner import GameRunner

class SelfPlayRunner:
    def __init__(self, card_db: CardDB, device: Optional[Any] = None) -> None:
        self.card_db = card_db
        self.device = device
        self.runner = GameRunner(card_db, device=device)

    def _prepare_initial_states(self, num_games: int, deck_card_id: int) -> List[Any]:
        states = []
        if deck_card_id == 0:
             print("Error: No valid card ID found for deck construction.")
             return []

        print(f"Constructing decks using CardID: {deck_card_id}")

        deck_list = [deck_card_id] * 40
        for _ in range(num_games):
            state = self.runner.prepare_state(deck_list, deck_list)
            states.append(state)
        return states

    def collect_data(self, model_path: Optional[str], output_path: str, episodes: int = 10, sims: int = 800, batch_size: int = 32, threads: int = 4) -> None:
        # Update model in runner
        self.runner.load_model(model_path)

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

        print(f"Starting collection: {episodes} episodes, {sims} sims, {threads} threads")
        start_time = time.time()

        try:
            # We want data collection enabled for self-play
            results = self.runner.run_games(initial_states, sims, batch_size, threads, temperature=1.0, add_noise=True, collect_data=True)
        except Exception as e:
            print(f"GameRunner execution failed: {e}")
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

    def evaluate_matchup(self, model1_path: Optional[str], model2_path: Optional[str], episodes: int = 10, sims: int = 800, batch_size: int = 32, threads: int = 4) -> float:
        # Note: GameRunner currently supports only ONE model (Self-Play).
        # Evaluating two DIFFERENT models (Asymmetric) requires custom callback logic not yet in standard GameRunner.
        # For now, this method will raise NotImplementedError or needs logic in GameRunner to handle 2 networks.
        # Given the "Code Duplication" task, moving the Symmetric logic is the priority.
        # I will leave the original implementation for evaluate_matchup here OR implement DualModel support in GameRunner.
        # Implementing DualModel support in GameRunner complicates it.
        # I will retain the custom logic here for asymmetric evaluation but use GameRunner helper for symmetrical parts if possible.

        # However, reusing GameRunner for this is tricky because the callback is bound to self.network.
        # So I will keep the custom evaluator here for now, but maybe GameRunner can accept a custom callback?
        # Yes, but GameRunner wraps ParallelRunner execution.

        # Let's re-implement evaluate_matchup using the original logic but adapted slightly if needed.
        # Or just keep it as is? But I want to remove duplication.
        # The duplication was mainly in self-play collection.
        # I'll keep the asymmetric logic separate for now as it's distinct.

        # Copied from original file, but cleaning up imports

        net1 = AlphaZeroNetwork(self.runner.input_size, self.runner.action_size).to(self.device)
        if model1_path: net1.load_state_dict(torch.load(model1_path, map_location=self.device))
        net1.eval()

        net2 = AlphaZeroNetwork(self.runner.input_size, self.runner.action_size).to(self.device)
        if model2_path: net2.load_state_dict(torch.load(model2_path, map_location=self.device))
        net2.eval()

        valid_cid = 0
        for cid, defn in self.card_db.items():
            if defn.type == dm_ai_module.CardType.CREATURE:
                valid_cid = cid
                break

        initial_states = self._prepare_initial_states(episodes, valid_cid)
        runner = dm_ai_module.ParallelRunner(self.card_db, sims, batch_size)

        from dm_toolkit.ai.agent.network import AlphaZeroNetwork # Re-import if needed or rely on top level

        def evaluator(states: List[Any]) -> Tuple[Any, Any]:
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
    cards_path = os.path.join(project_root, 'data', 'cards.json')
    if os.path.exists(cards_path):
        db = dm_ai_module.JsonLoader.load_cards(cards_path)
        sp = SelfPlayRunner(db)
        # sp.collect_data(None, "test_data.npz", episodes=2, sims=50)
