
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

from dm_toolkit.ai.agent.heuristic_agent import HeuristicAgent

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

        self.dummy_state = dm_ai_module.GameState(100)
        self.input_size = dm_ai_module.TensorConverter.INPUT_SIZE
        self.action_size = dm_ai_module.ActionEncoder.TOTAL_ACTION_SIZE

        print(f"Config: Input Size={self.input_size}, Action Size={self.action_size}")

        self.agent1 = HeuristicAgent(0)
        self.agent2 = HeuristicAgent(1)

    def run_episode(self):
        state = dm_ai_module.GameState(1000)

        for i in range(40):
            iid_p1 = i
            iid_p2 = i + 40
            state.add_card_to_deck(0, self.valid_card_id, iid_p1)
            state.add_card_to_deck(1, self.valid_card_id, iid_p2)

        state.initialize_card_stats(self.card_db, 1000)
        dm_ai_module.PhaseManager.start_game(state, self.card_db)

        examples = []
        step_count = 0
        max_steps = 300

        while True:
            if step_count > max_steps:
                return examples, dm_ai_module.GameResult.DRAW

            # Correctly check winner enum
            if state.winner != dm_ai_module.GameResult.NONE:
                 return examples, state.winner

            active_pid = state.active_player_id
            agent = self.agent1 if active_pid == 0 else self.agent2

            legal_actions = dm_ai_module.ActionGenerator.generate_legal_actions(state, self.card_db)

            if not legal_actions:
                 dm_ai_module.PhaseManager.next_phase(state, self.card_db)
                 continue

            chosen_action = agent.get_action(state, legal_actions, self.card_db)

            if not chosen_action:
                 break

            masked_tensor = dm_ai_module.TensorConverter.convert_to_tensor(state, active_pid, self.card_db, True)
            full_tensor = dm_ai_module.TensorConverter.convert_to_tensor(state, active_pid, self.card_db, False)
            token_seq = dm_ai_module.TokenConverter.encode_state(state, active_pid, 200)

            policy_vec = np.zeros(self.action_size, dtype=np.float32)
            action_idx = dm_ai_module.ActionEncoder.action_to_index(chosen_action)
            if 0 <= action_idx < self.action_size:
                policy_vec[action_idx] = 1.0

            mask_vec = np.zeros(self.action_size, dtype=np.float32)
            for action in legal_actions:
                idx = dm_ai_module.ActionEncoder.action_to_index(action)
                if 0 <= idx < self.action_size:
                    mask_vec[idx] = 1.0

            examples.append({
                "masked_state": masked_tensor,
                "full_state": full_tensor,
                "tokens": token_seq,
                "policy": policy_vec,
                "mask": mask_vec,
                "player": active_pid
            })

            dm_ai_module.EffectResolver.resolve_action(state, chosen_action, self.card_db)

            if chosen_action.type == dm_ai_module.ActionType.PASS:
                dm_ai_module.PhaseManager.next_phase(state, self.card_db)

            dm_ai_module.trigger_loop_detection(state)
            step_count += 1

        return examples, dm_ai_module.GameResult.DRAW

    def collect_data(self, episodes, output_file):
        print(f"Collecting data... Episodes={episodes}")
        all_data = []

        start_time = time.time()
        p1_wins = 0
        p2_wins = 0
        draws = 0

        for i in range(episodes):
            examples, result = self.run_episode()

            if result == dm_ai_module.GameResult.P1_WIN: p1_wins += 1
            elif result == dm_ai_module.GameResult.P2_WIN: p2_wins += 1
            else: draws += 1

            if (i+1) % 10 == 0:
                print(f"Progress: {i+1}/{episodes} (Result: {result})")

            for ex in examples:
                player = ex["player"]
                value = 0.0
                if result == dm_ai_module.GameResult.P1_WIN: value = 1.0 if player == 0 else -1.0
                elif result == dm_ai_module.GameResult.P2_WIN: value = 1.0 if player == 1 else -1.0

                all_data.append({
                    "masked": ex["masked_state"],
                    "full": ex["full_state"],
                    "tokens": ex["tokens"],
                    "policy": ex["policy"],
                    "mask": ex["mask"],
                    "value": value
                })

        duration = time.time() - start_time
        print(f"Done. Time: {duration:.2f}s. Samples: {len(all_data)}")
        print(f"Results: P1={p1_wins}, P2={p2_wins}, Draw={draws}")

        if not all_data:
            print("No data collected.")
            return

        masked_states = np.array([x["masked"] for x in all_data], dtype=np.float32)
        full_states = np.array([x["full"] for x in all_data], dtype=np.float32)
        # Use object array for jagged tokens
        tokens = np.array([x["tokens"] for x in all_data], dtype=object)
        policies = np.array([x["policy"] for x in all_data], dtype=np.float32)
        masks = np.array([x["mask"] for x in all_data], dtype=np.float32)
        values = np.array([x["value"] for x in all_data], dtype=np.float32)

        np.savez_compressed(output_file,
            states_masked=masked_states,
            states_full=full_states,
            tokens=tokens,
            policies=policies,
            masks=masks,
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
