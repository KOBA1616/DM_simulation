
import os
import sys
import json
import random
import time
import argparse
from datetime import datetime
from typing import Dict, Any, List, Tuple, Optional, cast
from dm_toolkit.dm_types import CardCounts

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
from dm_toolkit import commands

class EvolutionEcosystem:
    def __init__(self, cards_path: str, meta_decks_path: str, output_path: Optional[str] = None) -> None:
        self.cards_path: str = cards_path
        self.meta_decks_path: str = meta_decks_path
        self.output_path: str = output_path or meta_decks_path

        # Load Data
        print(f"Loading cards from {cards_path}...")
        self.card_db = dm_ai_module.JsonLoader.load_cards(cards_path)
        print(f"Loaded {len(self.card_db)} cards.")

        self.load_meta_decks()

        # Evolution Config
        self.evo_config = dm_ai_module.DeckEvolutionConfig()
        self.evo_config.target_deck_size = 40
        self.evo_config.mutation_rate = 0.2  # 20% mutation chance per card slot

        self.evolver = dm_ai_module.DeckEvolution(self.card_db)

        # Runner
        # We use heuristic agent for deck evaluation for speed
        self.runner = dm_ai_module.ParallelRunner(self.card_db, 50, 1)

    def load_meta_decks(self) -> None:
        if os.path.exists(self.meta_decks_path):
            with open(self.meta_decks_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.meta_decks = data.get("decks", [])
            print(f"Loaded {len(self.meta_decks)} meta decks.")
        else:
            print("Meta decks file not found. Starting with empty meta.")
            self.meta_decks = []

    def save_meta_decks(self) -> None:
        data = {"decks": self.meta_decks}
        with open(self.output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"Saved {len(self.meta_decks)} decks to {self.output_path}")

    def generate_challenger(self) -> Tuple[List[Any], str]:
        """Generates a challenger deck by mutating an existing meta deck or random creation."""
        if not self.meta_decks:
            # Create random deck if no meta
            # Filter for valid cards (creatures/spells) - simplistic approach
            valid_ids = list(self.card_db.keys())
            # Basic deck: 40 random cards
            deck = [random.choice(valid_ids) for _ in range(40)]
            return deck, "Random_Gen0"

        # Select parent
        parent_entry = random.choice(self.meta_decks)
        parent_deck = parent_entry["cards"]
        parent_name = parent_entry["name"]

        # Create pool (all cards)
        pool = list(self.card_db.keys())

        # Evolve
        new_deck = self.evolver.evolve_deck(parent_deck, pool, self.evo_config)

        # Name
        new_name = f"{parent_name}_v{int(time.time()) % 10000}"
        return new_deck, new_name

    def collect_smart_stats(self, deck: List[Any], opponent_deck: List[Any], num_games: int = 5) -> Dict[Any, Dict[str, int]]:
        """
        Runs a few single-threaded games to collect detailed card statistics.
        Returns aggregated stats dictionary: {card_id: {play: int, resource: int, ...}}
        """
        aggregated = {}

        for i in range(num_games):
            seed = int(time.time()) + i * 1000
            instance = dm_ai_module.GameInstance(seed, self.card_db)

            # Initialize stats in C++
            # Assuming 40 cards deck size, though we pass actual lists.
            # GameInstance.initialize_card_stats takes only deck_size in bindings or automatically handles card_db if stored
            instance.initialize_card_stats(40)

            # Setup decks
            instance.state.set_deck(0, deck)
            instance.state.set_deck(1, opponent_deck)

            dm_ai_module.PhaseManager.start_game(instance.state, self.card_db)

            # Simple agents
            agent0 = dm_ai_module.HeuristicEvaluator(self.card_db)
            # Use same for opponent
            agent1 = dm_ai_module.HeuristicEvaluator(self.card_db)

            # Run Game Loop
            # We need to manually drive it since GameInstance.start_game just sets up state
            # ParallelRunner logic:
            steps = 0
            max_steps = 10000 # Increase limit to avoid premature termination

            while steps < max_steps:
                 if instance.state.game_over:
                     break

                 res = dm_ai_module.GameResult.NONE
                 if dm_ai_module.PhaseManager.check_game_over(instance.state, res):
                    break

                 try:
                     legal_actions = dm_ai_module.ActionGenerator.generate_legal_commands(instance.state, self.card_db) or []
                 except Exception:
                     legal_actions = []
                 try:
                     cmds = commands.generate_legal_commands(instance.state, self.card_db) or []
                 except Exception:
                     cmds = []
                 if not legal_actions and not cmds:
                     dm_ai_module.PhaseManager.next_phase(instance.state, self.card_db)
                     continue

                 # Simple greedy selection using HeuristicEvaluator (which returns float score)
                 # Actually HeuristicEvaluator evaluates state.
                 # We need an agent that picks action.
                 # Python binding for HeuristicAgent is not exposed directly?
                 # Ah, wait. ParallelRunner uses HeuristicAgent in C++.
                 # In Python, we can just pick random or use a simple heuristic.
                 # For stat collection, random might be too chaotic.
                 # Let's pick random for now as HeuristicAgent isn't exposed.
                 # Or better, just pick first action (often PASS if available, or first card).
                 # To get meaningful stats, we want somewhat reasonable play.
                 # Let's use a very simple heuristic: prioritized actions.

                 # Prefer an action that can be executed via command when available
                 best_action = legal_actions[0] if legal_actions else None
                 best_cmd = None
                 if cmds:
                     best_cmd = cmds[0]

                 # Improved simple heuristic for stat collection
                 # 1. Charge Mana (up to 7)
                 # 2. Play Card
                 # 3. Attack Player
                 # 4. Pass/Other

                 found = False
                 # Check for Mana Charge first
                 current_mana = len(instance.state.get_zone(instance.state.active_player_id, dm_ai_module.Zone.MANA))
                 if current_mana < 7:
                    # Use getattr guards for ActionType members to satisfy mypy
                    _MANA_CHARGE = getattr(dm_ai_module.ActionType, 'MANA_CHARGE', None)
                    _MOVE_CARD = getattr(dm_ai_module.ActionType, 'MOVE_CARD', None)
                    for act in legal_actions:
                        # Check for MANA_CHARGE or MOVE_CARD in Mana Phase
                        if act.type == _MANA_CHARGE:
                            best_action = act
                            found = True
                            break
                        if instance.state.current_phase == dm_ai_module.Phase.MANA and act.type == _MOVE_CARD:
                            best_action = act
                            found = True
                            break

                 if not found:
                     for act in legal_actions:
                         if act.type == dm_ai_module.ActionType.PLAY_CARD:
                             best_action = act
                             found = True
                             break

                 if not found:
                     for act in legal_actions:
                         if act.type == dm_ai_module.ActionType.ATTACK_PLAYER:
                             best_action = act
                             found = True
                             break

                 # Execute via unified command path; convert action if needed
                 from dm_toolkit.unified_execution import ensure_executable_command
                 from dm_toolkit.engine.compat import EngineCompat
                 if best_cmd is not None:
                    try:
                        cast(Any, instance.state).execute_command(best_cmd)
                    except Exception:
                        try:
                            best_cmd.execute(instance.state)
                        except Exception:
                            try:
                                EngineCompat.ExecuteCommand(instance.state, best_cmd)
                            except Exception:
                                pass
                 else:
                     if best_action is not None:
                         try:
                            cmd = ensure_executable_command(best_action)
                            EngineCompat.ExecuteCommand(instance.state, cmd)
                         except Exception:
                             # Last resort: try central compat wrapper, then fall back to EngineCompat resolver
                             try:
                                 from dm_toolkit.compat_wrappers import execute_action_compat
                                 execute_action_compat(instance.state, best_action, cast(Dict[int, Any], self.card_db))
                             except Exception:
                                 try:
                                     EngineCompat.EffectResolver_resolve_action(instance.state, best_action, cast(Dict[int, Any], self.card_db))
                                 except Exception:
                                     pass
                 steps += 1

            # Game finished (or max steps), collect stats
            # get_card_stats returns {cid: {play_count, win_count, sum_cost_discount, sum_early_usage, ...}}
            game_stats = dm_ai_module.get_card_stats(instance.state)

            for cid_obj, stats in game_stats.items():
                cid = int(cid_obj) # pybind might return int or object
                if cid not in aggregated:
                    aggregated[cid] = {'play': 0, 'resource': 0}

                aggregated[cid]['play'] += stats.get('play_count', 0)
                # 'sum_early_usage' tracks turns where played early, which is not exactly "resource use".
                # But 'sum_cost_discount' tracks mana savings.
                # If we want "Resource Use (Mana/Cost)", we usually mean "put into mana zone".
                # The C++ CardStats doesn't explicitly track "times put into mana".
                # However, cards in mana zone are tracked in the state.
                # We can scan the mana zone at end of game!
                # Since we want "usage frequency", scanning mana zone at end tells us if it was used as resource.
                # Note: this only counts if it *ended* in mana. If it was burnt, it's missed.
                # But acceptable for now.

            # Scan mana zones for resource usage
            # Player 0 is our challenger
            mana_zone = instance.state.get_zone(0, dm_ai_module.Zone.MANA)
            for iid in mana_zone:
                card_inst = instance.state.get_card_instance(iid)
                if card_inst:
                    real_cid = card_inst.card_id
                    if real_cid not in aggregated:
                        aggregated[real_cid] = {'play': 0, 'resource': 0}
                    aggregated[real_cid]['resource'] += 1

        return aggregated

    def evaluate_deck(self, challenger_deck: List[Any], challenger_name: str, num_games: int = 10) -> Tuple[float, float]:
        """Runs the challenger against a sample of the meta."""
        # card_counts will be populated below
        if not self.meta_decks:
            # If no meta, it wins by default, but we should still score it
            # Run self-play or dummy play for stats
            dummy_opp = challenger_deck
            smart_stats = self.collect_smart_stats(challenger_deck, dummy_opp, num_games=2) # Few games for stats

            # Calculate Score
            total_score: float = 0.0
            deck_count = len(challenger_deck)
            # Count copies of each card in deck for normalization
            card_counts: Dict[Any, int] = {}
            for cid in challenger_deck:
                card_counts[cid] = card_counts.get(cid, 0) + 1

            for cid, count in card_counts.items():
                stats = smart_stats.get(cid, {'play': 0, 'resource': 0})
                # Formula: (5 * Play + 2 * Resource) / (Appearance Count)
                # Appearance Count approx = num_games * count
                appearance = 2 * count
                if appearance > 0:
                    score = (5 * stats['play'] + 2 * stats['resource']) / appearance
                    total_score += score

            # Normalize deck score (avg per card?) or sum?
            # Requirement says "Score by card usage". Usually we want the deck score to be the sum or avg.
            # Let's print it.
            print(f"Deck '{challenger_name}' Smart Score: {total_score:.2f} (Win Rate: 100% - Default)")
            return 1.0, total_score

        wins = 0
        total_games = 0

        # Sample opponents
        opponents = random.sample(self.meta_decks, min(3, len(self.meta_decks)))

        # We also collect stats against the first opponent for "Smart Scoring"
        # Running 2 games (1 as P1, 1 as P2) purely for stats collection using the python loop
        # Note: This adds overhead but fulfills the requirement without C++ hacks.
        stats_opp = opponents[0]["cards"]
        smart_stats = self.collect_smart_stats(challenger_deck, stats_opp, num_games=2)

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

        # Calculate Deck Smart Score
        total_smart_score: float = 0.0
        card_counts = {}
        for cid in challenger_deck:
            card_counts[cid] = card_counts.get(cid, 0) + 1

        for cid, count in card_counts.items():
            stats = smart_stats.get(cid, {'play': 0, 'resource': 0})
            # Appearance count for the stat collection runs (2 games)
            appearance = 2 * count
            if appearance > 0:
                # 5 pts for Play, 2 pts for Resource
                score = (5 * stats['play'] + 2 * stats['resource']) / appearance
                total_smart_score += score

        print(f"Deck '{challenger_name}' Win Rate: {win_rate*100:.1f}% ({wins}/{total_games}), Smart Score: {total_smart_score:.2f}")
        return win_rate, total_smart_score

    def run_generation(self, num_challengers: int = 5, games_per_match: int = 20, min_win_rate: float = 0.55) -> None:
        print(f"--- Starting Generation ---")
        challengers = []
        for _ in range(num_challengers):
            deck, name = self.generate_challenger()
            challengers.append((deck, name))

        accepted_count = 0
        for deck, name in challengers:
            wr, score = self.evaluate_deck(deck, name, games_per_match)

            # Acceptance criteria: Mainly Win Rate, but we could use Score as tie breaker or bonus
            # For now, keep Win Rate as primary gate
            if wr >= min_win_rate:
                print(f"  [ACCEPTED] {name} enters the meta! (Score: {score:.2f})")
                self.meta_decks.append({
                    "name": name,
                    "cards": deck,
                    "score": score
                })
                accepted_count += 1
            else:
                print(f"  [REJECTED] {name} too weak (WR: {wr*100:.1f}%, Score: {score:.2f})")

        # Pruning
        if len(self.meta_decks) > 20:
            print("Pruning meta decks...")
            # Keep top 10 newest
            kept = self.meta_decks[-10:]
            remaining = self.meta_decks[:-10]
            if remaining:
                kept.extend(random.sample(remaining, min(5, len(remaining))))
            self.meta_decks = kept

        if accepted_count > 0:
            self.save_meta_decks()

def main() -> None:
    parser = argparse.ArgumentParser(description="Self-Evolution Ecosystem")
    parser.add_argument("--cards", default="data/cards.json", help="Path to cards.json")
    parser.add_argument("--meta", default="data/meta_decks.json", help="Path to meta_decks.json")
    parser.add_argument("--generations", type=int, default=1, help="Number of generations to run")
    parser.add_argument("--challengers", type=int, default=5, help="Challengers per generation")

    args = parser.parse_args()

    if not os.path.exists(args.cards):
        print(f"Cards file not found: {args.cards}")
        return

    ecosystem = EvolutionEcosystem(args.cards, args.meta)

    for i in range(args.generations):
        print(f"\n=== Generation {i+1}/{args.generations} ===")
        ecosystem.run_generation(num_challengers=args.challengers)

if __name__ == "__main__":
    main()
