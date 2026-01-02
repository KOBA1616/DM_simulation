import os
import sys
import numpy as np
import torch
import time
import argparse
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
from dm_toolkit.types import CardDB
from python.training.game_runner import GameRunner

class SelfPlayRunner:
    def __init__(self, card_db: CardDB, device: Optional[Any] = None) -> None:
        self.card_db = card_db
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.runner = GameRunner(card_db, self.device)
        self.input_size = self.runner.input_size
        self.action_size = self.runner.action_size

    def load_model(self, model_path: Optional[str]) -> Any:
        network = AlphaZeroNetwork(self.input_size, self.action_size).to(self.device)
        if model_path and os.path.exists(model_path):
            print(f"Loading model: {model_path}")
            network.load_state_dict(torch.load(model_path, map_location=self.device))
        else:
            print("Warning: Model path not found or None, using random initialization.")
        network.eval()
        return network

    def get_valid_deck(self) -> List[int]:
        valid_cid = 0
        for cid, defn in self.card_db.items():
            if defn.type == dm_ai_module.CardType.CREATURE:
                valid_cid = cid
                break
        return [valid_cid] * 40

    def collect_data(self, model_path: Optional[str], output_path: str, episodes: int = 10, sims: int = 800, batch_size: int = 32, threads: int = 4) -> None:
        network = self.load_model(model_path)
        callback = self.runner.create_network_callback(network)

        deck = self.get_valid_deck()
        initial_states = self.runner.prepare_initial_states(episodes, deck, deck)

        if not initial_states:
             print("Failed to prepare initial states.")
             return

        print(f"Starting collection: {episodes} episodes, {sims} sims, {threads} threads")
        start_time = time.time()

        try:
            results = self.runner.run_games(
                initial_states, callback, sims, batch_size, threads,
                temperature=1.0, add_noise=True, collect_data=True
            )
        except Exception as e:
            print(f"Game execution failed: {e}")
            return

        duration = time.time() - start_time
        print(f"Collection finished in {duration:.2f}s")

        # Process results
        data = self.runner.process_results_to_data(results)

        # Additional formatting for legacy compatibility if needed (masked vs full)
        # GameRunner returns flat states (floats). If self_play needs legacy format, it might need adjustment.
        # But for now, let's stick to what GameRunner provides which is likely 'states' (flat tensor list).

        self.runner.save_data(data, output_path)

    def evaluate_matchup(self, model1_path: Optional[str], model2_path: Optional[str], episodes: int = 10, sims: int = 800, batch_size: int = 32, threads: int = 4) -> float:
        net1 = self.load_model(model1_path)
        net2 = self.load_model(model2_path)

        deck = self.get_valid_deck()
        initial_states = self.runner.prepare_initial_states(episodes, deck, deck)

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

            # Helper for inference
            def infer(net, s_list):
                if not s_list: return [], []
                # Use flat batch conversion
                flat = dm_ai_module.TensorConverter.convert_batch_flat(s_list, self.card_db, True)
                t_in = torch.tensor(flat, dtype=torch.float32).view(len(s_list), self.input_size).to(self.device)
                with torch.no_grad():
                    logits, vals = net(t_in)
                    p = torch.softmax(logits, dim=1).cpu().numpy()
                    v = vals.squeeze(1).cpu().numpy()
                return p, v

            p1_pol, p1_val = infer(net1, states_p1)
            p2_pol, p2_val = infer(net2, states_p2)

            # Reassemble
            final_policies = [None] * len(states)
            final_values = [None] * len(states)

            for k, original_idx in enumerate(indices_p1):
                final_policies[original_idx] = p1_pol[k]
                final_values[original_idx] = p1_val[k]

            for k, original_idx in enumerate(indices_p2):
                final_policies[original_idx] = p2_pol[k]
                final_values[original_idx] = p2_val[k]

            return final_policies, final_values

        print(f"Evaluating Matchup: {episodes} games")
        # Ensure we set the callback via set_flat_batch_callback if needed, or pass it directly.
        # However, evaluator here is complex (different models).
        # GameRunner.run_games uses whatever is passed to it.
        # BUT GameRunner.run_games also attempts to set_flat_batch_callback globally.
        # This might overwrite if we are not careful, but here we pass 'evaluator' which matches the signature run_games expects.

        # Note: run_games wraps the callback to ignore player_id if set_flat_batch_callback is used.
        # But here we handle player ID inside the evaluator by checking state.active_player_id.
        # The wrapped callback in GameRunner ignores the second arg.
        # This is fine.

        results = self.runner.run_games(
            initial_states, evaluator, sims, batch_size, threads,
            temperature=1.0, add_noise=False, collect_data=False
        )

        p1_wins = 0
        for info in results:
            if info.result == 1: p1_wins += 1

        win_rate = p1_wins / episodes
        print(f"Matchup Result: P1 Wins {p1_wins}/{episodes} ({win_rate*100:.1f}%)")
        return win_rate

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--sims", type=int, default=100)
    parser.add_argument("--output", type=str, default="test_data.npz")
    args = parser.parse_args()

    cards_path = os.path.join(project_root, 'data', 'cards.json')
    if os.path.exists(cards_path):
        db = dm_ai_module.JsonLoader.load_cards(cards_path)
        sp = SelfPlayRunner(db)
        sp.collect_data(None, args.output, episodes=args.episodes, sims=args.sims)
    else:
        print("cards.json not found")
