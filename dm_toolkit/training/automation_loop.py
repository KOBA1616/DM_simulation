import os
import sys
import argparse
import time
import shutil

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

from dm_toolkit.training.generation_manager import GenerationManager
from dm_toolkit.training.self_play import SelfPlayRunner
from dm_toolkit.training.train_simple import train_pipeline

class AutomationLoop:
    def __init__(self, card_db):
        self.card_db = card_db
        self.manager = GenerationManager()
        self.runner = SelfPlayRunner(card_db)

    def run(self, max_generations=10):
        current_gen = self.manager.get_latest_generation_number()
        print(f"Starting Automation Loop. Current Generation: {current_gen}")

        for gen in range(current_gen + 1, current_gen + max_generations + 1):
            print(f"\n=== Generation {gen} ===")

            production_model = self.manager.get_production_model_path()
            if not production_model:
                print("No production model found. Starting from scratch (Random).")
            else:
                print(f"Using production model: {production_model}")

            # 1. Collection
            training_data_path = self.manager.get_training_data_path(gen)
            print(f"Step 1: Collection -> {training_data_path}")
            # Use lower sims for speed in loop for now
            self.runner.collect_data(production_model, training_data_path, episodes=50, sims=200, threads=4)

            # 2. Training
            candidate_path = self.manager.create_candidate_path(gen)
            print(f"Step 2: Training -> {candidate_path}")

            # Train on new data + some old data?
            # For simplicity, train on just new data for this iteration
            # Ideally, we should maintain a buffer.
            # Let's grab last 3 generation data files
            data_files = []
            for g in range(max(1, gen-2), gen+1):
                p = self.manager.get_training_data_path(g)
                if os.path.exists(p):
                    data_files.append(p)

            print(f"Training on: {data_files}")
            train_pipeline(data_files, production_model, candidate_path, epochs=2)

            # 3. Gatekeeper
            print(f"Step 3: Gatekeeper Verification")
            if not production_model:
                # First generation always wins if no production model
                print("First generation automatically promoted.")
                self.manager.promote_candidate(candidate_path)
            else:
                # Compare Candidate (P1) vs Production (P2)
                # Need > 55% winrate
                win_rate = self.runner.evaluate_matchup(candidate_path, production_model, episodes=20, sims=200)

                print(f"Candidate Win Rate: {win_rate*100:.1f}%")
                if win_rate > 0.55:
                    print("Candidate PROMOTED!")
                    self.manager.promote_candidate(candidate_path)
                else:
                    print("Candidate REJECTED.")
                    # We might want to keep the candidate file for analysis but not promote

            # Cleanup
            # self.manager.cleanup_candidates() # Optional

if __name__ == "__main__":
    cards_path = os.path.join(project_root, 'data', 'cards.json')
    db = dm_ai_module.JsonLoader.load_cards(cards_path)

    loop = AutomationLoop(db)
    loop.run(max_generations=5)
