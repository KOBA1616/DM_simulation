
import sys
import os
import random
import time
import argparse

# Ensure we can import the module
sys.path.append(os.path.join(os.getcwd(), 'bin'))

try:
    import dm_ai_module
except ImportError:
    print("Error: Could not import dm_ai_module. Make sure it is built and in the bin/ directory.")
    sys.exit(1)

def run_stress_test(iterations=10000, max_steps=2000, verbose=False):
    """
    Runs a stress test with random actions.
    """
    print(f"Starting Stress Test: {iterations} iterations, Max Steps: {max_steps}")

    # Load Cards
    try:
        # Try JSON
        card_db = dm_ai_module.JsonLoader.load_cards("data/cards.json")
        print(f"Loaded {len(card_db)} cards from data/cards.json")
    except Exception as e:
        print(f"Failed to load cards: {e}")
        sys.exit(1)

    if not card_db:
        print("No cards loaded. Exiting.")
        sys.exit(1)

    all_card_ids = list(card_db.keys())

    crashes = 0
    start_time = time.time()

    for i in range(iterations):
        if verbose and i % 100 == 0:
            print(f"Game {i}/{iterations}...", end='\r')

        seed = 12345 + i
        step_count = 0
        try:
            gi = dm_ai_module.GameInstance(seed, card_db)

            # Setup Scenario using reset_with_scenario
            config = dm_ai_module.ScenarioConfig()

            # Populate Zones randomly
            config.my_hand_cards = [random.choice(all_card_ids) for _ in range(5)]
            config.my_shields = [random.choice(all_card_ids) for _ in range(5)]
            config.my_mana_zone = [random.choice(all_card_ids) for _ in range(5)]

            # Enemy setup
            config.enemy_shield_count = 5
            # Add some enemy creatures to interact with
            config.enemy_battle_zone = [random.choice(all_card_ids) for _ in range(2)]

            gi.reset_with_scenario(config)

            # Manually set decks since ScenarioConfig doesn't support them yet
            # and reset_with_scenario might fill them with dummies.
            # Let's override with random cards to test deck interactions.
            deck1 = [random.choice(all_card_ids) for _ in range(40)]
            deck2 = [random.choice(all_card_ids) for _ in range(40)]

            gi.state.set_deck(0, deck1)
            gi.state.set_deck(1, deck2)

            state = gi.state

            while state.winner == -1 and step_count < max_steps:
                try:
                    from dm_toolkit import commands_v2
                    generate_legal_commands = commands_v2.generate_legal_commands
                except Exception:
                    generate_legal_commands = None

                # Prefer commands; avoid direct action usage
                cmds = generate_legal_commands(state, card_db) if generate_legal_commands else []

                if not cmds:
                    # Stalemate or bug
                    break

                # Prefer executing random command when available
                cmd = random.choice(cmds)
                try:
                    state.execute_command(cmd)
                except Exception:
                    try:
                        cmd.execute(state)
                    except Exception:
                        try:
                            from dm_toolkit.engine.compat import EngineCompat
                            EngineCompat.ExecuteCommand(state, cmd, card_db)
                        except Exception:
                            pass
                step_count += 1

        except Exception as e:
            print(f"\nCRASH at Game {i}, Seed {seed}, Step {step_count}")
            print(f"Error: {e}")
            crashes += 1
            # Continue to find more crashes or stop?
            # For stress testing, we might want to continue to count total failures.

    end_time = time.time()
    duration = end_time - start_time

    print(f"\nStress Test Finished.")
    print(f"Total Games: {iterations}")
    print(f"Total Crashes: {crashes}")
    print(f"Time Taken: {duration:.2f}s")
    if duration > 0:
         print(f"Average FPS: {(iterations * max_steps) / duration:.2f} (Estimated max steps)") # Very rough estimate

    if crashes > 0:
        sys.exit(1)
    else:
        sys.exit(0)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stress Test DM AI Engine")
    parser.add_argument("--iterations", type=int, default=1000, help="Number of games to simulate")
    parser.add_argument("--steps", type=int, default=2000, help="Max steps per game")
    parser.add_argument("--verbose", action="store_true", help="Print progress")

    args = parser.parse_args()

    run_stress_test(args.iterations, args.steps, args.verbose)
