
import sys
import os

# Add bin directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../../bin'))
sys.path.append('bin')

try:
    import dm_ai_module
except ImportError:
    print("Error: dm_ai_module not found. Please build the project first.")
    sys.exit(1)

import time
import numpy as np

def verify_data_collector():
    print("Verifying DataCollector.collect_data_batch_heuristic...")
    card_db = dm_ai_module.JsonLoader.load_cards("data/cards.json")
    collector = dm_ai_module.DataCollector(card_db)

    start_time = time.time()
    batch = collector.collect_data_batch_heuristic(10) # 10 episodes
    end_time = time.time()

    print(f"Collected {len(batch.states)} steps in {end_time - start_time:.4f} seconds.")
    print(f"States shape: {len(batch.states)} x {len(batch.states[0]) if batch.states else 0}")
    print(f"Policies shape: {len(batch.policies)} x {len(batch.policies[0]) if batch.policies else 0}")
    print(f"Values count: {len(batch.values)}")

    assert len(batch.states) > 0
    assert len(batch.states) == len(batch.policies)
    assert len(batch.states) == len(batch.values)
    print("DataCollector verification passed.")

def verify_parallel_scenario():
    print("Verifying ParallelRunner.play_scenario_match...")
    card_db = dm_ai_module.JsonLoader.load_cards("data/cards.json")
    runner = dm_ai_module.ParallelRunner(card_db, 10, 1)

    config = dm_ai_module.ScenarioConfig()
    config.my_mana = 5
    config.my_hand_cards = [1, 2, 3] # IDs
    config.enemy_shield_count = 0 # Easy win
    # config.my_battle_zone in C++ ScenarioConfig expects vector<int> (CardIDs), NOT CardInstance objects.
    # The current simplified ScenarioConfig struct just takes card IDs and assumes default properties (untapped etc)
    # or it might need expansion if we want tapped state.
    # Looking at scenario_config.hpp, it is std::vector<int>.
    config.my_battle_zone = [1] # ID 1 creature

    start_time = time.time()
    # Run 10 games
    results = runner.play_scenario_match(config, 10, 2)
    end_time = time.time()

    print(f"Run 10 scenario matches in {end_time - start_time:.4f} seconds.")
    print(f"Results: {results}")

    assert len(results) == 10
    print("ParallelRunner Scenario verification passed.")

def verify_parallel_deck():
    print("Verifying ParallelRunner.play_deck_matchup...")
    card_db = dm_ai_module.JsonLoader.load_cards("data/cards.json")
    runner = dm_ai_module.ParallelRunner(card_db, 10, 1)

    deck1 = [1] * 40
    deck2 = [1] * 40

    start_time = time.time()
    results = runner.play_deck_matchup(deck1, deck2, 10, 2)
    end_time = time.time()

    print(f"Run 10 deck matches in {end_time - start_time:.4f} seconds.")
    print(f"Results: {results}")

    assert len(results) == 10
    print("ParallelRunner Deck verification passed.")

if __name__ == "__main__":
    verify_data_collector()
    verify_parallel_scenario()
    verify_parallel_deck()
