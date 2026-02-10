
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
from dm_toolkit.training.train_simple import Trainer  # type: ignore[import-untyped]

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
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.network: Optional[AlphaZeroNetwork] = None
        self.input_size = dm_ai_module.TensorConverter.INPUT_SIZE
        try:
            self.action_size = int(getattr(dm_ai_module.CommandEncoder, 'TOTAL_COMMAND_SIZE'))
        except Exception:
            # Fallback: use historical default and warn
            self.action_size = 591
            print(f"Warning: dm_ai_module.CommandEncoder.TOTAL_COMMAND_SIZE not available, falling back to {self.action_size}")

        # Evolution Config
        self.evo_config = dm_ai_module.DeckEvolutionConfig()
        self.evo_config.target_deck_size = 40
        self.evo_config.mutation_rate = 0.2  # 20% mutation chance per card swap
        self.evolver = dm_ai_module.DeckEvolution(self.card_db)

        # Runner
        # Use positional arguments for C++ constructor
        self.runner = dm_ai_module.ParallelRunner(self.card_db, 50, 32)

        # Load Model
        self.load_model()

        # Initialize Population
        self.init_population()

    def load_model(self) -> None:
        self.network = AlphaZeroNetwork(self.input_size, self.action_size).to(self.device)
        if self.model_path and os.path.exists(self.model_path):
            print(f"Loading model from {self.model_path}")
            try:
                self.network.load_state_dict(torch.load(self.model_path, map_location=self.device))
            except Exception as e:
                print(f"Failed to load model: {e}. Starting fresh.")
        if self.network:
            self.network.eval()

    def init_population(self) -> None:
        # Try to load meta decks
        meta_decks_path = os.path.join(project_root, 'data', 'meta_decks.json')
        meta_decks = []
        if os.path.exists(meta_decks_path):
            with open(meta_decks_path, 'r', encoding='utf-8') as f:
                try:
                    data = json.load(f)
                    meta_decks = data.get("decks", [])
                except Exception as e:
                    print(f"Error loading meta_decks.json: {e}")

        # Collect all valid card IDs for random generation
        valid_ids = []
        for cid, defn in self.card_db.items():
            # Basic filter: Only creatures and spells, cost <= 10 (heuristic)
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
                # Random deck
                deck = []
                for _ in range(40):
                    deck.append(random.choice(valid_ids))
                name = f"Random_Deck_{i}"

            # Ensure deck is valid (40 cards)
            while len(deck) < 40:
                deck.append(valid_ids[0])
            deck = deck[:40]

            self.agents.append(PBTAgent(i, deck, name))

        print(f"Initialized population of {len(self.agents)} agents.")

    def evaluate_network_callback(self, states: List[Any], player_id: int) -> Tuple[List[List[float]], List[float]]:
        """
        Callback for C++ ParallelRunner.
        Receives batch of GameStates, returns (policies, values).
        """
        if not states:
            return [], []

        try:
            # Flatten batch tensor [B, InputSize]
            batch_tensor_list = dm_ai_module.TensorConverter.convert_batch_flat(states, self.card_db, True)

            # Convert to Torch
            # batch_tensor_list is a list of floats (flattened batch)
            input_data = torch.tensor(batch_tensor_list, dtype=torch.float32).view(len(states), self.input_size).to(self.device)

            if self.network is None:
                return [], []

            with torch.no_grad():
                policies, values = self.network(input_data)

            # Convert back to lists
            policies_list = policies.cpu().numpy().tolist()
            values_list = values.cpu().numpy().flatten().tolist()

            return policies_list, values_list

        except Exception as e:
            print(f"Error in callback: {e}")
            # Fallback
            return [], []

    def run_generation(self, generation_id: int, matches_per_pair: int = 2) -> None:
        print(f"--- Generation {generation_id} ---")

        # 1. Matchmaking (Round Robin or Random Pairs)
        pairs = []
        if self.population_size <= 10:
            for i in range(len(self.agents)):
                for j in range(i + 1, len(self.agents)):
                    pairs.append((self.agents[i], self.agents[j]))
        else:
            # Random pairing (N matches)
            pool = self.agents[:]
            random.shuffle(pool)
            for i in range(0, len(pool), 2):
                if i+1 < len(pool):
                    pairs.append((pool[i], pool[i+1]))

        print(f"Running {len(pairs)} matchups ({matches_per_pair} games each)...")

        # 2. Execution
        initial_states = []
        metadata = [] # stores (agent1_idx, agent2_idx) for each game

        for p1, p2 in pairs:
            for _ in range(matches_per_pair):
                state = dm_ai_module.GameState(1000)
                # Set decks
                state.set_deck(0, p1.deck)
                state.set_deck(1, p2.deck)

                # Setup
                state.initialize_card_stats(self.card_db, 1000)
                dm_ai_module.PhaseManager.start_game(state, self.card_db)

                initial_states.append(state)
                metadata.append((p1, p2)) # 0 vs 1

        print(f"Total games queued: {len(initial_states)}")

        if not initial_states:
            print("No games to play.")
            return

        # Run Games
        # Set the global callback
        dm_ai_module.set_flat_batch_callback(self.evaluate_network_callback)

        # Call play_games (positional args)
        # play_games(initial_states, evaluator, temp, noise, threads, alpha, collect_data)
        results = self.runner.play_games(
            initial_states,
            self.evaluate_network_callback,
            1.0, # temperature
            True, # add_noise
            4, # threads
            0.5, # alpha
            True # collect_data
        )

        dm_ai_module.clear_flat_batch_callback()

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
            winner = res.result # 0=NONE, 1=P1, 2=P2, 3=DRAW

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

        # 4. Train Model (Incremental)
        if len(training_data_batch['states']) > 0:
            print("Training model...")
            trainer = Trainer([data_path], self.model_path, self.model_path or "model_v1.pth")
            trainer.train(epochs=1, batch_size=64)

            # Reload model
            self.load_model()
        else:
            print("No data collected, skipping training.")

        # 5. Evolution (Selection & Mutation)
        self.agents.sort(key=lambda a: a.score, reverse=True)

        print("\nLeaderboard:")
        for i, a in enumerate(self.agents[:5]):
            print(f"{i+1}. {a.name} (ID: {a.id}): {a.score:.1f} ({a.wins}W-{a.losses}L)")

        # Bottom 20% are replaced by mutated top 20%
        cutoff = int(self.population_size * 0.2)
        if cutoff > 0 and self.population_size >= 2:
            top_agents = self.agents[:cutoff]
            bottom_agents = self.agents[-cutoff:]

            # Create candidate pool (all cards in existence or subset)
            candidate_pool = []
            for cid in self.card_db.keys():
                 if self.card_db[cid].type in [dm_ai_module.CardType.CREATURE, dm_ai_module.CardType.SPELL]:
                     candidate_pool.append(cid)

            for i, victim in enumerate(bottom_agents):
                # Ensure we have top agents to copy from
                if not top_agents:
                    break

                parent = random.choice(top_agents)
                print(f"Evolving Agent {victim.id} (Score {victim.score:.1f}) -> Child of {parent.name}")

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
