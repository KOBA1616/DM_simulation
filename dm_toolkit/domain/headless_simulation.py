"""
Run a headless simulation using SimulationRunner with C++ MCTS and Transformer Model.
"""
import os
import random
import sys
import time

def run_simulation(args):
    # Check for dm_ai_module inside the function to prevent import-time crash
    try:
        import dm_ai_module
        # The python fallback explicitly sets IS_NATIVE = False.
        # If IS_NATIVE is present and False, we abort.
        if getattr(dm_ai_module, 'IS_NATIVE', True) is False:
            print("Error: dm_ai_module is loaded but IS_NATIVE is False.")
            print("The Native C++ engine is required for high-performance simulation.")
            sys.exit(1)
    except ImportError:
        print("Error: dm_ai_module not available. Native engine is required for this simulation.")
        sys.exit(1)

    from dm_toolkit.domain.simulation import SimulationRunner

    # Determine paths
    cards_path = getattr(args, 'cards', os.path.join("data", "cards.json"))
    games_count = getattr(args, 'games', 100)
    seed = getattr(args, 'seed', None)
    quiet = getattr(args, 'quiet', False)
    model_path = getattr(args, 'model', None)

    if not os.path.exists(cards_path):
        print(f"Error: Cards file not found at {cards_path}")
        return

    print(f"Loading cards from {cards_path}...")
    card_db = dm_ai_module.JsonLoader.load_cards(cards_path)
    print(f"Loaded {len(card_db)} cards.")

    # Setup decks
    # Preserve the behavior: Generate a random valid deck and play against itself (Mirror Match)
    valid_ids = list(card_db.keys())
    if not valid_ids:
        print("No cards found in card DB.")
        return

    if seed is not None:
        random.seed(seed)

    # Simple random deck
    deck = [random.choice(valid_ids) for _ in range(40)]

    # We will run a mirror match
    deck_lists = [deck, deck]

    print(f"Starting simulation: {games_count} games")
    print("Evaluator: Model (C++ MCTS + Transformer)")
    if model_path:
        print(f"Using model: {model_path}")
    else:
        print("Using initialized model (Untrained)")

    # Callbacks for progress
    def progress_callback(progress, message):
        if not quiet:
            # Overwrite line to show progress
            sys.stdout.write(f"\r[{progress}%] {message}")
            sys.stdout.flush()
            if progress >= 100:
                print() # Newline at end

    def finished_callback(win_rate, summary):
        if not quiet:
            print() # Ensure newline
        print("\nSimulation Finished.")
        print(summary)

    # Determine threads: leave one core free if possible, min 1
    cpu_count = os.cpu_count() or 1
    threads = max(1, cpu_count - 1)

    runner = SimulationRunner(
        card_db=card_db,
        scenario_name="standard", # Unused when deck_lists is provided
        episodes=games_count,
        threads=threads,
        sims=100, # Standard MCTS simulation count
        evaluator_type="Model", # Force Model
        model_path=model_path,
        deck_lists=deck_lists
    )

    try:
        runner.run(progress_callback, finished_callback)
    except KeyboardInterrupt:
        runner.cancel()
        print("\nSimulation cancelled by user.")
    except Exception as e:
        print(f"\nError running simulation: {e}")
