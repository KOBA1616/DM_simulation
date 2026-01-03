
import os
import sys
import json
import random
import time
import argparse
import subprocess
import logging
import shutil
from datetime import datetime
from typing import Dict, Any, List, Tuple, Optional

# Set up paths
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
bin_path = os.path.join(project_root, 'bin')
if bin_path not in sys.path:
    sys.path.append(bin_path)

try:
    import dm_ai_module
except ImportError:
    print("Error: Could not import dm_ai_module. Ensure it is built and in bin/")
    sys.exit(1)

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler("ecosystem.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("Ecosystem")

# --- Deck Evolution Logic (Existing) ---
class EvolutionEcosystem:
    def __init__(self, cards_path: str, meta_decks_path: str, output_path: Optional[str] = None) -> None:
        self.cards_path: str = cards_path
        self.meta_decks_path: str = meta_decks_path
        self.output_path: str = output_path or meta_decks_path

        # Load Data
        logger.info(f"Loading cards from {cards_path}...")
        self.card_db = dm_ai_module.JsonLoader.load_cards(cards_path)

        self.load_meta_decks()

        # Evolution Config
        self.evo_config = dm_ai_module.DeckEvolutionConfig()
        self.evo_config.target_deck_size = 40
        self.evo_config.mutation_rate = 0.2

        self.evolver = dm_ai_module.DeckEvolution(self.card_db)

        # Runner for evaluation
        self.runner = dm_ai_module.ParallelRunner(self.card_db, 50, 1)

    def load_meta_decks(self) -> None:
        if os.path.exists(self.meta_decks_path):
            with open(self.meta_decks_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.meta_decks = data.get("decks", [])
            logger.info(f"Loaded {len(self.meta_decks)} meta decks.")
        else:
            logger.info("Meta decks file not found. Starting with empty meta.")
            self.meta_decks = []

    def save_meta_decks(self) -> None:
        data = {"decks": self.meta_decks}
        # Ensure directory exists
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        with open(self.output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved {len(self.meta_decks)} decks to {self.output_path}")

    def generate_challenger(self) -> Tuple[List[Any], str]:
        if not self.meta_decks:
            valid_ids = list(self.card_db.keys())
            deck = [random.choice(valid_ids) for _ in range(40)]
            return deck, "Random_Gen0"

        parent_entry = random.choice(self.meta_decks)
        parent_deck = parent_entry["cards"]
        parent_name = parent_entry["name"]
        pool = list(self.card_db.keys())
        new_deck = self.evolver.evolve_deck(parent_deck, pool, self.evo_config)
        new_name = f"{parent_name}_v{int(time.time()) % 10000}"
        return new_deck, new_name

    def evaluate_deck(self, challenger_deck: List[Any], challenger_name: str, num_games: int = 10) -> float:
        if not self.meta_decks:
            return 1.0

        wins = 0
        total_games = 0
        opponents = random.sample(self.meta_decks, min(3, len(self.meta_decks)))

        for opp in opponents:
            opp_deck = opp["cards"]
            # Match 1: Challenger as P1
            results1 = self.runner.play_deck_matchup(challenger_deck, opp_deck, num_games // 2, 4)
            wins += results1.count(1)
            # Match 2: Challenger as P2
            results2 = self.runner.play_deck_matchup(opp_deck, challenger_deck, num_games - (num_games // 2), 4)
            wins += results2.count(2)
            total_games += num_games

        win_rate = wins / total_games
        logger.info(f"Deck '{challenger_name}' Win Rate: {win_rate*100:.1f}% ({wins}/{total_games})")
        return win_rate

    def run_generation(self, num_challengers: int = 5, games_per_match: int = 20, min_win_rate: float = 0.55) -> None:
        logger.info(f"--- Starting Deck Evolution Generation ---")
        challengers = []
        for _ in range(num_challengers):
            deck, name = self.generate_challenger()
            challengers.append((deck, name))

        accepted_count = 0
        for deck, name in challengers:
            wr = self.evaluate_deck(deck, name, games_per_match)
            if wr >= min_win_rate:
                logger.info(f"  [ACCEPTED] {name} enters the meta!")
                self.meta_decks.append({"name": name, "cards": deck})
                accepted_count += 1
            else:
                logger.info(f"  [REJECTED] {name} too weak.")

        if len(self.meta_decks) > 20:
            logger.info("Pruning meta decks...")
            kept = self.meta_decks[-10:]
            remaining = self.meta_decks[:-10]
            if remaining:
                kept.extend(random.sample(remaining, min(5, len(remaining))))
            self.meta_decks = kept

        if accepted_count > 0:
            self.save_meta_decks()

# --- Automation Pipeline Logic ---

class PipelineConfig:
    def __init__(self):
        self.data_dir = os.path.join(project_root, "data", "pipeline_data")
        self.models_dir = os.path.join(project_root, "models")
        self.current_model_path = os.path.join(self.models_dir, "model_current.pth")
        self.best_model_path = os.path.join(self.models_dir, "model_best.pth")
        self.meta_decks_path = os.path.join(project_root, "data", "meta_decks.json")
        self.cards_path = os.path.join(project_root, "data", "cards.json")

        # Default Params
        self.episodes_per_gen = 500
        self.train_epochs = 5
        self.verify_sims = 200
        self.verify_episodes = 50
        self.min_win_rate = 55.0  # % to promote
        self.deck_gen_challengers = 5
        self.retries = 3

class PipelineManager:
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.state_file = "ecosystem_state.json"

        os.makedirs(self.config.data_dir, exist_ok=True)
        os.makedirs(self.config.models_dir, exist_ok=True)

        self.state = {
            "generation": 1,
            "step": "COLLECT",
            "failures": 0
        }
        self.load_state()

    def load_state(self):
        if os.path.exists(self.state_file):
            try:
                with open(self.state_file, 'r') as f:
                    self.state = json.load(f)
                logger.info(f"Loaded state: {self.state}")
            except Exception as e:
                logger.error(f"Failed to load state: {e}")

    def save_state(self):
        with open(self.state_file, 'w') as f:
            json.dump(self.state, f, indent=2)

    def run_command(self, cmd: List[str], log_filename: str) -> bool:
        """Runs a subprocess command with logging."""
        logger.info(f"Running command: {' '.join(cmd)}")
        try:
            with open(log_filename, 'w') as log_f:
                result = subprocess.run(cmd, stdout=log_f, stderr=subprocess.STDOUT, check=True)
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Command failed with exit code {e.returncode}. See {log_filename}")
            return False
        except Exception as e:
            logger.error(f"Execution error: {e}")
            return False

    def step_collect(self):
        logger.info(">>> Step: COLLECT DATA")
        output_file = os.path.join(self.config.data_dir, f"gen_{self.state['generation']}.npz")

        cmd = [
            sys.executable,
            os.path.join(project_root, "python/training/collect_training_data.py"),
            "--episodes", str(self.config.episodes_per_gen),
            "--output", output_file,
            "--mode", "both"
        ]

        if self.run_command(cmd, "log_collect.txt"):
            self.state["step"] = "TRAIN"
            self.state["failures"] = 0
            self.save_state()
        else:
            self.handle_failure()

    def step_train(self):
        logger.info(">>> Step: TRAIN MODEL")
        data_file = os.path.join(self.config.data_dir, f"gen_{self.state['generation']}.npz")
        if not os.path.exists(data_file):
            logger.error(f"Data file missing: {data_file}. Rolling back to COLLECT.")
            self.state["step"] = "COLLECT"
            self.save_state()
            return

        cmd = [
            sys.executable,
            os.path.join(project_root, "dm_toolkit/training/train_simple.py"),
            "--data_files", data_file,
            "--save", self.config.current_model_path,
            "--epochs", str(self.config.train_epochs)
        ]

        # If best model exists, load it as base? Or current?
        # For iterative training, we usually load the previous model.
        # Check if we have a previous generation model.
        if os.path.exists(self.config.best_model_path):
             cmd.extend(["--model", self.config.best_model_path])
        elif os.path.exists(self.config.current_model_path):
             # If retrying
             cmd.extend(["--model", self.config.current_model_path])

        if self.run_command(cmd, "log_train.txt"):
            self.state["step"] = "VERIFY"
            self.state["failures"] = 0
            self.save_state()
        else:
            self.handle_failure()

    def step_verify(self):
        logger.info(">>> Step: VERIFY PERFORMANCE")
        if not os.path.exists(self.config.current_model_path):
            logger.error("Model missing. Rolling back to TRAIN.")
            self.state["step"] = "TRAIN"
            self.save_state()
            return

        result_file = "verify_result.json"
        cmd = [
            sys.executable,
            os.path.join(project_root, "python/training/verify_performance.py"),
            "--model", self.config.current_model_path,
            "--scenario", "lethal_puzzle_easy", # Standard benchmark for now
            "--episodes", str(self.config.verify_episodes),
            "--sims", str(self.config.verify_sims),
            "--result_file", result_file
        ]

        if self.run_command(cmd, "log_verify.txt"):
            # Check result
            if os.path.exists(result_file):
                try:
                    with open(result_file, 'r') as f:
                        res = json.load(f)
                    win_rate = res.get("win_rate", 0)
                    logger.info(f"Verification Win Rate: {win_rate}%")

                    if win_rate >= self.config.min_win_rate:
                        logger.info("Performance acceptable. Promoting model.")
                        shutil.copy(self.config.current_model_path, self.config.best_model_path)
                    else:
                        logger.info("Performance below threshold. Keeping previous best.")

                    self.state["step"] = "EVOLVE"
                    self.state["failures"] = 0
                    self.save_state()
                except Exception as e:
                    logger.error(f"Failed to read verification result: {e}")
                    self.handle_failure()
            else:
                logger.error("Verification result file not found.")
                self.handle_failure()
        else:
            self.handle_failure()

    def step_evolve(self):
        logger.info(">>> Step: DECK EVOLUTION")
        cmd = [
            sys.executable,
            os.path.join(project_root, "python/training/evolution_ecosystem.py"),
            "--mode", "decks_only",
            "--episodes", str(self.config.episodes_per_gen) # Not used in deck gen but harmless
        ]

        if self.run_command(cmd, "log_evolve.txt"):
            self.state["step"] = "COLLECT"
            self.state["generation"] += 1
            self.state["failures"] = 0
            self.save_state()
            logger.info(f"Gen {self.state['generation']-1} Complete. Starting Gen {self.state['generation']}")
        else:
            self.handle_failure()

    def handle_failure(self):
        self.state["failures"] += 1
        logger.warning(f"Failure occurred. Count: {self.state['failures']}")
        self.save_state()

        if self.state["failures"] >= self.config.retries:
            logger.critical("Max retries exceeded. Aborting pipeline.")
            sys.exit(1)

        logger.info("Retrying in 5 seconds...")
        time.sleep(5)

    def run_loop(self):
        logger.info("Starting Automated Pipeline Loop...")
        while True:
            step = self.state["step"]
            if step == "COLLECT":
                self.step_collect()
            elif step == "TRAIN":
                self.step_train()
            elif step == "VERIFY":
                self.step_verify()
            elif step == "EVOLVE":
                self.step_evolve()
            else:
                logger.error(f"Unknown step: {step}")
                break

def main():
    parser = argparse.ArgumentParser(description="Duel Masters AI Automated Ecosystem")
    parser.add_argument("--episodes", type=int, default=100, help="Episodes per generation")
    parser.add_argument("--epochs", type=int, default=5, help="Training epochs")
    parser.add_argument("--verify_episodes", type=int, default=50, help="Verification episodes")
    parser.add_argument("--verify_sims", type=int, default=200, help="Verification MCTS sims")
    parser.add_argument("--mode", type=str, default="pipeline", choices=["pipeline", "decks_only"], help="Operation mode")

    args = parser.parse_args()

    if args.mode == "decks_only":
        # Legacy mode
        ecosystem = EvolutionEcosystem("data/cards.json", "data/meta_decks.json")
        ecosystem.run_generation()
    else:
        # Pipeline mode
        config = PipelineConfig()
        config.episodes_per_gen = args.episodes
        config.train_epochs = args.epochs
        config.verify_episodes = args.verify_episodes
        config.verify_sims = args.verify_sims

        manager = PipelineManager(config)
        manager.run_loop()

if __name__ == "__main__":
    main()
