
import os
import sys
import shutil
import json
import subprocess
import argparse
from typing import List

# Add python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from evolution.pbt_config import PBTConfig

class PBTRunner:
    def __init__(self, config: PBTConfig):
        self.config = config
        self.python_exe = sys.executable

    def run_command(self, cmd: List[str]):
        """Runs a command and waits for it."""
        print(f"Running: {' '.join(cmd)}")
        subprocess.check_call(cmd)

    def initialize(self):
        """Initializes the population."""
        print("Initializing PBT Population...")
        # Check if model exists, if not, rely on random init in training/collection
        pass

    def run_generation(self, gen: int):
        print(f"=== Generation {gen} ===")

        # 1. Collect Data (Self-Play)
        # We use the current best model (or population models).
        # For simplicity in this PBT-lite: Single threaded pipeline.
        # Future: Run N workers.

        current_model = os.path.join(self.config.model_dir, f"model_gen_{gen-1}.pth") if gen > 0 else None
        data_file = os.path.join(self.config.data_dir, f"data_gen_{gen}.npz")

        cmd_collect = [
            self.python_exe, "python/evolution/collect_selfplay.py",
            "--episodes", str(self.config.episodes_per_gen),
            "--output", data_file,
            "--type", self.config.model_type,
            "--sims", str(self.config.mcts_sims)
        ]
        if current_model and os.path.exists(current_model):
            cmd_collect.extend(["--model", current_model])

        if os.path.exists(self.config.meta_deck_path):
            cmd_collect.extend(["--meta_decks", self.config.meta_deck_path])

        self.run_command(cmd_collect)

        # 2. Train
        new_model = os.path.join(self.config.model_dir, f"model_gen_{gen}.pth")

        cmd_train = [
            self.python_exe, "python/training/train_simple.py",
            "--data_files", data_file,
            "--save", new_model,
            "--network_type", self.config.model_type,
            "--epochs", str(self.config.epochs_per_gen),
            "--batch_size", str(self.config.batch_size)
        ]
        # In PBT, we might fine-tune the *previous* model.
        # train_simple currently initializes from scratch or loads?
        # train_simple initializes new model. We should modify it to load weights if we want continuity.
        # BUT, standard AlphaZero retrains from recent window.
        # PBT usually fine-tunes.
        # Let's assume train_simple starts fresh on the data.
        # For continuous learning, we should load state dict if possible.
        # But for this task (Integration), we link the steps.

        self.run_command(cmd_train)

        # 3. Verify / Evaluate
        # Verify against a benchmark
        result_file = os.path.join(self.config.data_dir, f"result_gen_{gen}.json")
        cmd_verify = [
            self.python_exe, "python/tests/verification/verify_performance.py",
            "--model", new_model,
            "--model_type", self.config.model_type,
            "--episodes", "10", # Quick check
            "--result_file", result_file
        ]
        self.run_command(cmd_verify)

        # Check result
        win_rate = 0.0
        if os.path.exists(result_file):
            with open(result_file, 'r') as f:
                res = json.load(f)
                win_rate = res.get("win_rate", 0.0)
        print(f"Generation {gen} Win Rate: {win_rate}%")

        # 4. Deck Evolution (Meta Update)
        # Evolve a deck and add to meta if successful
        new_deck_file = os.path.join(self.config.data_dir, f"deck_gen_{gen}.json")
        cmd_evolve = [
            self.python_exe, "python/evolution/evolve_decks.py",
            "--meta_decks", self.config.meta_deck_path,
            "--output", new_deck_file
        ]
        # Only if meta decks exist or we want to start one
        try:
            self.run_command(cmd_evolve)

            # If successful, add to meta_decks.json
            if os.path.exists(new_deck_file):
                with open(new_deck_file, 'r') as f:
                    new_deck = json.load(f)

                meta_decks = []
                if os.path.exists(self.config.meta_deck_path):
                    with open(self.config.meta_deck_path, 'r') as f:
                        try:
                            meta_decks = json.load(f)
                        except: pass

                # Add new deck
                meta_decks.append(new_deck)
                # Keep size manageable
                if len(meta_decks) > 10:
                    meta_decks.pop(0)

                with open(self.config.meta_deck_path, 'w') as f:
                    json.dump(meta_decks, f)
                print(f"Updated Meta Decks. Count: {len(meta_decks)}")

        except Exception as e:
            print(f"Deck evolution failed: {e}")

    def run(self):
        self.initialize()
        for gen in range(1, self.config.generations + 1):
            self.run_generation(gen)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gens", type=int, default=5)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--sims", type=int, default=800)
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()

    config = PBTConfig(
        generations=args.gens,
        episodes_per_gen=args.episodes,
        epochs_per_gen=1,
        mcts_sims=args.sims,
        batch_size=args.batch_size
    )
    runner = PBTRunner(config)
    runner.run()
