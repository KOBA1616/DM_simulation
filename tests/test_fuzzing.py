
import os
import sys
import random
import time
import pytest

# Add the bin directory to sys.path
sys.path.append(os.path.join(os.getcwd(), 'bin'))

try:
    import dm_ai_module
except ImportError:
    pass

@pytest.mark.skipif('dm_ai_module' not in sys.modules, reason="requires dm_ai_module C++ extension")
def test_random_actions_fuzzing():
    """
    Run N random games and ensure no crashes occur.
    Migrated from legacy_tests/test_fuzzing_random_actions.py.
    """
    print("Starting Fuzzing Test (Random Actions)...")

    # Load Cards
    if not os.path.exists('data/cards.json'):
        pytest.skip("data/cards.json not found")

    card_db = dm_ai_module.JsonLoader.load_cards('data/cards.json')

    # Settings
    # Reduce for CI/Standard test suite. Use environment var for deep testing.
    if os.environ.get('FULL_FUZZ_TEST'):
        NUM_GAMES = 100
        MAX_TURNS = 50
    else:
        NUM_GAMES = 5
        MAX_TURNS = 20

    crashes = 0
    start_time = time.time()

    for game_idx in range(NUM_GAMES):
        try:
            gi = dm_ai_module.GameInstance(game_idx, card_db)
            state = gi.state

            # Simple Loop
            while state.game_over == False and state.turn_number < MAX_TURNS:
                # Generate Actions
                actions = dm_ai_module.ActionGenerator.generate_legal_actions(state, card_db)

                if not actions:
                    # Check if game really over? Or assume pass/stuck
                    # If start of turn, might need to start phase?
                    # Usually ActionGenerator handles phase logic if configured,
                    # or we assume engine handles it.
                    # If empty, force next phase or break
                    break

                # Pick Random
                action = random.choice(actions)

                # Resolve
                dm_ai_module.EffectResolver.resolve_action(state, action, card_db)

        except Exception as e:
            print(f"CRASH in Game {game_idx}: {e}")
            crashes += 1

    end_time = time.time()
    duration = end_time - start_time
    print(f"Fuzzing Complete. Games: {NUM_GAMES}, Crashes: {crashes}, Time: {duration:.2f}s")

    assert crashes == 0, f"Fuzzing failed with {crashes} crashes."

if __name__ == "__main__":
    test_random_actions_fuzzing()
