import os
import sys
import json
import time

# Add the bin directory to sys.path
sys.path.append(os.path.join(os.getcwd(), 'bin'))

try:
    import dm_ai_module
except ImportError:
    print("Error: Could not import dm_ai_module. Make sure it is built and in bin/.")
    sys.exit(1)

def run_games(num_games, output_file):
    print(f"Starting stats collection: {num_games} games...")

    # We will aggregate stats in Python
    # Structure: card_id -> {play_count: int, sum_early_usage: float, ...}
    master_stats = {}

    start_time = time.time()

    # Load Card DB (Minimal for now, or load from JSON if available)
    # We need at least the card definitions to run the game
    # For this test, we might rely on CsvLoader if available or just hardcode some cards
    # If CsvLoader is available in bindings, use it.

    card_db = {}
    try:
        # Assuming cards.csv is in data/cards.csv
        # We need to construct map<CardID, CardDefinition>
        # dm_ai_module.CsvLoader.load_cards("data/cards.csv") returns std::map
        # Check if load_cards is exposed
        if hasattr(dm_ai_module.CsvLoader, 'load_cards'):
             print("Loading cards from data/cards.csv...")
             card_db = dm_ai_module.CsvLoader.load_cards("data/cards.csv")
             print(f"Loaded {len(card_db)} cards.")
    except Exception as e:
        print(f"Failed to load cards: {e}. Using empty DB (might crash engine).")

    for i in range(num_games):
        seed = 42 + i
        gs = dm_ai_module.GameState(seed)

        # Setup decks
        # We need a valid deck.
        # If we have card_db, pick some random cards.
        # Otherwise use setup_test_duel() which adds card ID 1.
        if len(card_db) > 0:
             # Just pick first 40 cards or random
             deck_ids = list(card_db.keys())[:40]
             # Make sure we have 40
             while len(deck_ids) < 40:
                 deck_ids.append(deck_ids[0])

             # Need to convert to uint16 vector
             # But binding expects vector<uint16_t>
             # Python list of ints should work if pybind11/stl is included
             gs.set_deck(0, deck_ids)
             gs.set_deck(1, deck_ids)
        else:
             gs.setup_test_duel()

        # Initialize stats tracking in GameState
        dm_ai_module.initialize_card_stats(gs, card_db, 40)

        # Run Game
        dm_ai_module.PhaseManager.start_game(gs, card_db)

        step_count = 0
        while True:
            is_over, result = dm_ai_module.PhaseManager.check_game_over(gs)
            if is_over:
                break

            # Simple Random Agent
            active_pid = gs.active_player_id
            legal_actions = dm_ai_module.ActionGenerator.generate_legal_actions(gs, card_db)

            if not legal_actions:
                # Should pass?
                # Usually generate_legal_actions returns PASS if no other action
                # If empty, maybe something wrong or just pass
                # Create PASS action
                pass_action = dm_ai_module.Action()
                pass_action.type = dm_ai_module.ActionType.PASS
                dm_ai_module.EffectResolver.resolve_action(gs, pass_action, card_db)
            else:
                # Random choice
                import random
                action = random.choice(legal_actions)
                dm_ai_module.EffectResolver.resolve_action(gs, action, card_db)

            # Phase transition
            dm_ai_module.PhaseManager.next_phase(gs, card_db)

            step_count += 1
            if step_count > 1000: # Limit max steps
                break

        # Game Over or Limit Reached
        # Collect Stats
        game_stats = dm_ai_module.get_card_stats(gs)

        # Merge into master_stats
        for cid, stats in game_stats.items():
            if stats['play_count'] > 0:
                if cid not in master_stats:
                    master_stats[cid] = {
                        'play_count': 0, 'win_count': 0,
                        'sum_early_usage': 0.0, 'sum_late_usage': 0.0,
                        'sum_trigger_rate': 0.0, 'sum_cost_discount': 0.0
                    }

                m = master_stats[cid]
                m['play_count'] += stats['play_count']
                m['win_count'] += stats['win_count']
                m['sum_early_usage'] += stats['sum_early_usage']
                m['sum_late_usage'] += stats['sum_late_usage']
                m['sum_trigger_rate'] += stats['sum_trigger_rate']
                m['sum_cost_discount'] += stats['sum_cost_discount']

        if (i+1) % 10 == 0:
            print(f"Finished game {i+1}/{num_games}")

    elapsed = time.time() - start_time
    print(f"Completed {num_games} games in {elapsed:.2f} seconds.")

    # Save to JSON
    # Format: array of {id: ..., play_count: ..., sums: [...]}
    output_data = []
    for cid, m in master_stats.items():
        # Create sums array (16 dims, only first 4 implemented so far)
        sums = [0.0] * 16
        sums[0] = m['sum_early_usage']
        sums[1] = m['sum_late_usage']
        sums[2] = m['sum_trigger_rate']
        sums[3] = m['sum_cost_discount']

        output_data.append({
            "id": cid,
            "play_count": m['play_count'],
            "win_count": m['win_count'],
            "sums": sums
        })

    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"Stats saved to {output_file}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        num_games = int(sys.argv[1])
    else:
        num_games = 10

    output_file = "data/card_stats_collected.json"
    run_games(num_games, output_file)
