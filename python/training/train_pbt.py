
import os
import sys
import json
import random
import time
import argparse
import glob
import numpy as np
import torch
from typing import List, Tuple, Dict, Any, Optional

# Ensure bin and src are in path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
bin_path = os.path.join(project_root, 'bin')
python_path = os.path.join(project_root, 'python')

if bin_path not in sys.path:
    sys.path.append(bin_path)
if python_path not in sys.path:
    sys.path.append(python_path)
if project_root not in sys.path:
    sys.path.append(project_root)

try:
    import dm_ai_module
except ImportError:
    print(f"Error: Could not import dm_ai_module. Make sure it is built and in {bin_path}")
    sys.exit(1)

from dm_toolkit.ai.agent.network import AlphaZeroNetwork
from dm_toolkit.training.train_simple import Trainer
from training.game_runner import GameRunner

class PBTAgent:
    def __init__(self, agent_id: int, deck: List[int], name: str = "Agent") -> None:
        self.id: int = agent_id
        self.deck: List[int] = deck
        self.name: str = name
        self.score: float = 1200.0  # Elo rating
        self.wins: int = 0
        self.losses: int = 0
        self.matches: int = 0

    def update_elo(self, opponent_elo: float, result: float, k_factor: float = 32.0) -> None:
        expected = 1.0 / (1.0 + 10.0 ** ((opponent_elo - self.score) / 400.0))
        self.score += k_factor * (result - expected)
        self.matches += 1
        if result == 1.0:
            self.wins += 1
        elif result == 0.0:
            self.losses += 1
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "deck": self.deck,
            "score": self.score,
            "wins": self.wins,
            "losses": self.losses
        }

class PBTManager:
    def __init__(self, card_db: Any, population_size: int = 10, model_path: Optional[str] = None) -> None:
        self.card_db: Any = card_db
        self.population_size = population_size
        self.agents: List[PBTAgent] = []
        self.model_path = model_path

        # Use GameRunner for execution
        self.runner = GameRunner(card_db, model_path=model_path)

        # Evolution Config
        self.evo_config = dm_ai_module.DeckEvolutionConfig()
        self.evo_config.target_deck_size = 40
        self.evo_config.mutation_rate = 0.2
        self.evolver = dm_ai_module.DeckEvolution(self.card_db)

        # Initialize Population
        self.init_population()

    def init_population(self) -> None:
        # Try to load meta decks
        meta_decks_path = os.path.join(project_root, 'data', 'meta_decks.json')
        meta_decks = []
        if os.path.exists(meta_decks_path):
            with open(meta_decks_path, 'r') as f:
                try:
                    data = json.load(f)
                    meta_decks = data.get("decks", [])
                except Exception as e:
                    print(f"Error loading meta_decks.json: {e}")

        # Collect all valid card IDs for random generation
        valid_ids = []
        for cid, defn in self.card_db.items():
            if defn.cost <= 10:
                valid_ids.append(cid)

        if not valid_ids:
             print("Warning: No valid cards found for deck generation.")
             valid_ids = [1] # Dummy

        for i in range(self.population_size):
            if i < len(meta_decks):
                deck = meta_decks[i]["cards"]
                name = meta_decks[i]["name"]
            else:
                deck = []
                for _ in range(40):
                    deck.append(random.choice(valid_ids))
                name = f"Random_Deck_{i}"

            while len(deck) < 40:
                deck.append(valid_ids[0])
            deck = deck[:40]

            self.agents.append(PBTAgent(i, deck, name))

        print(f"Initialized population of {len(self.agents)} agents.")

    def run_generation(self, generation_id: int, matches_per_pair: int = 2) -> None:
        print(f"--- Generation {generation_id} ---")

        # 1. Matchmaking
        pairs = []
        if self.population_size <= 10:
            for i in range(len(self.agents)):
                for j in range(i + 1, len(self.agents)):
                    pairs.append((self.agents[i], self.agents[j]))
        else:
            pool = self.agents[:]
            random.shuffle(pool)
            for i in range(0, len(pool), 2):
                if i+1 < len(pool):
                    pairs.append((pool[i], pool[i+1]))

        print(f"Running {len(pairs)} matchups ({matches_per_pair} games each)...")

        # 2. Execution
        initial_states = []
        metadata = []

        for p1, p2 in pairs:
            for _ in range(matches_per_pair):
                # Use runner helper to prepare state
                state = self.runner.prepare_state(p1.deck, p2.deck)
                initial_states.append(state)
                metadata.append((p1, p2))

        print(f"Total games queued: {len(initial_states)}")

        if not initial_states:
            print("No games to play.")
            return

        # Use GameRunner to execute
        # Note: We rely on the internal callback logic of GameRunner (batch inference)
        results = self.runner.run_games(
            initial_states,
            sims=50,
            batch_size=32,
            threads=4,
            temperature=1.0,
            add_noise=True,
            alpha=0.5,
            collect_data=True
        )

        print("Games finished.")

        # 3. Process Results & Training Data
        training_data_batch: Dict[str, List[Any]] = {
            'states': [],
            'policies': [],
            'values': []
        }

        for idx, res in enumerate(results):
            p1, p2 = metadata[idx]

            # Update Elo
            winner = res.result
            if winner == dm_ai_module.GameResult.P1_WIN:
                s1, s2 = 1.0, 0.0
            elif winner == dm_ai_module.GameResult.P2_WIN:
                s1, s2 = 0.0, 1.0
            else:
                s1, s2 = 0.5, 0.5

            p1.update_elo(p2.score, s1)
            p2.update_elo(p1.score, s2)

            # Collect Training Data
            if res.states:
                training_data_batch['states'].extend(res.states)
                training_data_batch['policies'].extend(res.policies)

                step_values = []
                for player in res.active_players:
                    if winner == dm_ai_module.GameResult.P1_WIN:
                        v = 1.0 if player == 0 else -1.0
                    elif winner == dm_ai_module.GameResult.P2_WIN:
                        v = 1.0 if player == 1 else -1.0
                    else:
                        v = 0.0
                    step_values.append([v])

                training_data_batch['values'].extend(step_values)

        # Save Training Data
        data_path = f"pbt_gen_{generation_id}_data.npz"
        np.savez_compressed(
            data_path,
            states=np.array(training_data_batch['states'], dtype=np.float32),
            policies=np.array(training_data_batch['policies'], dtype=np.float32),
            values=np.array(training_data_batch['values'], dtype=np.float32)
        )
        print(f"Saved {len(training_data_batch['states'])} samples to {data_path}")

        # 4. Train Model
        if len(training_data_batch['states']) > 0:
            print("Training model...")
            # Note: self.model_path might be None, handle default
            model_out = self.model_path or "model_v1.pth"
            trainer = Trainer([data_path], model_out, model_out)
            trainer.train(epochs=1, batch_size=64)

            # Reload model in runner
            self.runner.load_model(model_out)
        else:
            print("No data collected, skipping training.")

        # 5. Evolution
        self.agents.sort(key=lambda a: a.score, reverse=True)

        print("\nLeaderboard:")
        for i, a in enumerate(self.agents[:5]):
            print(f"{i+1}. {a.name} (ID: {a.id}): {a.score:.1f} ({a.wins}W-{a.losses}L)")

        cutoff = int(self.population_size * 0.2)
        if cutoff > 0 and self.population_size >= 2:
            top_agents = self.agents[:cutoff]
            bottom_agents = self.agents[-cutoff:]

            candidate_pool = []
            for cid in self.card_db.keys():
                 if self.card_db[cid].type in [dm_ai_module.CardType.CREATURE, dm_ai_module.CardType.SPELL]:
                     candidate_pool.append(cid)

            for i, victim in enumerate(bottom_agents):
                if not top_agents:
                    break
                parent = random.choice(top_agents)
                print(f"Evolving Agent {victim.id} -> Child of {parent.name}")

                new_deck = self.evolver.evolve_deck(parent.deck, candidate_pool, self.evo_config)

                victim.deck = new_deck
                victim.score = parent.score
                victim.wins = 0
                victim.losses = 0
                victim.matches = 0
                victim.name = f"Gen{generation_id}_ChildOf_{parent.id}"

        print("Generation Complete.")

def main() -> None:
    parser = argparse.ArgumentParser(description="PBT Training for Duel Masters AI")
    parser.add_argument("--gens", type=int, default=10, help="Number of generations")
    parser.add_argument("--pop", type=int, default=10, help="Population size")
    parser.add_argument("--matches", type=int, default=2, help="Matches per pair")
    parser.add_argument("--model", type=str, default="model_v1.pth", help="Model path")
    args = parser.parse_args()

    cards_path = os.path.join(project_root, 'data', 'cards.json')
    if not os.path.exists(cards_path):
        print("data/cards.json not found.")
        sys.exit(1)

    card_db = dm_ai_module.JsonLoader.load_cards(cards_path)

    manager = PBTManager(card_db, population_size=args.pop, model_path=args.model)

    for g in range(args.gens):
        manager.run_generation(g, matches_per_pair=args.matches)

if __name__ == "__main__":
    main()
