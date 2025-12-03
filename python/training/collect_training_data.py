
import os
import sys
import numpy as np
import argparse
import time
import random

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
bin_path = os.path.join(project_root, 'bin')
python_path = os.path.join(project_root, 'python')

if bin_path not in sys.path:
    sys.path.append(bin_path)
if python_path not in sys.path:
    sys.path.append(python_path)

try:
    import dm_ai_module
except ImportError:
    print(f"Error: Could not import dm_ai_module. Make sure it is built and in {bin_path}")
    sys.exit(1)

from py_ai.agent.heuristic_agent import HeuristicAgent

sys.path.append(os.path.dirname(__file__))

class HeuristicDataCollector:
    def __init__(self, card_db):
        self.card_db = card_db
        self.valid_card_id = 0
        for cid, defn in self.card_db.items():
            if defn.type == dm_ai_module.CardType.CREATURE:
                self.valid_card_id = cid
                break
        if self.valid_card_id == 0:
            if len(self.card_db) > 0:
                self.valid_card_id = list(self.card_db.keys())[0]

        print(f"Using Card ID {self.valid_card_id} for deck construction.")

        dummy_instance = dm_ai_module.GameInstance(42, self.card_db)
        dummy_vec = dm_ai_module.TensorConverter.convert_to_tensor(dummy_instance.state, 0, self.card_db, True)
        self.input_size = len(dummy_vec)
        self.action_size = dm_ai_module.ActionEncoder.TOTAL_ACTION_SIZE

        print(f"Config: Input Size={self.input_size}, Action Size={self.action_size}")

        self.agent1 = HeuristicAgent(0)
        self.agent2 = HeuristicAgent(1)

    def run_episode(self):
        instance = dm_ai_module.GameInstance(random.randint(0, 100000), self.card_db)
        state = instance.state

        deck = [self.valid_card_id] * 40
        state.set_deck(0, deck)
        state.set_deck(1, deck)

        dm_ai_module.PhaseManager.start_game(state, self.card_db)

        examples = []

        step_count = 0
        while True:
            is_over, result = dm_ai_module.PhaseManager.check_game_over(state)
            if is_over:
                return examples, result

            if step_count > 300:
                return examples, 3

            active_pid = state.active_player_id
            agent = self.agent1 if active_pid == 0 else self.agent2

            legal_actions = dm_ai_module.ActionGenerator.generate_legal_actions(state, self.card_db)

            chosen_action = agent.get_action(state, legal_actions, self.card_db)

            if not chosen_action:
                 break

            masked_tensor = dm_ai_module.TensorConverter.convert_to_tensor(state, active_pid, self.card_db, True)
            full_tensor = dm_ai_module.TensorConverter.convert_to_tensor(state, active_pid, self.card_db, False)

            policy_vec = np.zeros(self.action_size, dtype=np.float32)
            action_idx = dm_ai_module.ActionEncoder.action_to_index(chosen_action)
            if 0 <= action_idx < self.action_size:
                policy_vec[action_idx] = 1.0

            examples.append({
                "masked_state": masked_tensor,
                "full_state": full_tensor,
                "policy": policy_vec,
                "player": active_pid
            })

            dm_ai_module.EffectResolver.resolve_action(state, chosen_action, self.card_db)
            if chosen_action.type == dm_ai_module.ActionType.PASS:
                dm_ai_module.PhaseManager.next_phase(state, self.card_db)

            dm_ai_module.PhaseManager.fast_forward(state, self.card_db)
            step_count += 1

        return examples, 3

    def collect_data(self, episodes, output_file):
        print(f"Collecting data... Episodes={episodes}")
        all_data = []

        start_time = time.time()
        p1_wins = 0
        p2_wins = 0
        draws = 0

        for i in range(episodes):
            examples, result = self.run_episode()

            if result == 1: p1_wins += 1
            elif result == 2: p2_wins += 1
            else: draws += 1

            for ex in examples:
                player = ex["player"]
                value = 0.0
                if result == 1: value = 1.0 if player == 0 else -1.0
                elif result == 2: value = 1.0 if player == 1 else -1.0

                all_data.append({
                    "masked": ex["masked_state"],
                    "full": ex["full_state"],
                    "policy": ex["policy"],
                    "value": value
                })

            if (i+1) % 10 == 0:
                print(f"Progress: {i+1}/{episodes}")

        duration = time.time() - start_time
        print(f"Done. Time: {duration:.2f}s. Samples: {len(all_data)}")
        print(f"Results: P1={p1_wins}, P2={p2_wins}, Draw={draws}")

        # Save
        masked_states = np.array([x["masked"] for x in all_data], dtype=np.float32)
        full_states = np.array([x["full"] for x in all_data], dtype=np.float32)
        policies = np.array([x["policy"] for x in all_data], dtype=np.float32)
        values = np.array([x["value"] for x in all_data], dtype=np.float32)

        np.savez_compressed(output_file,
            states_masked=masked_states,
            states_full=full_states,
            policies=policies,
            values=values
        )
        print(f"Saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--output", type=str, default="training_data.npz")
    args = parser.parse_args()

    cards_path = os.path.join(project_root, 'data', 'cards.json')
    card_db = dm_ai_module.JsonLoader.load_cards(cards_path)

    collector = HeuristicDataCollector(card_db)
    collector.collect_data(args.episodes, args.output)
