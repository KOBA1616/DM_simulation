import sys
import os
import pytest
import random
import time

# Add the bin directory to sys.path
sys.path.append(os.path.join(os.getcwd(), 'bin'))

try:
    import dm_ai_module
except ImportError:
    pass

class TestFuzzing:
    @classmethod
    def setup_class(cls):
        # Load real cards if available
        try:
            cls.card_db = dm_ai_module.CsvLoader.load_cards("data/cards.csv")
            print(f"Loaded {len(cls.card_db)} cards.")
        except Exception as e:
            print(f"Failed to load cards: {e}")
            cls.card_db = {}

    def test_random_actions_fuzzing(self):
        """
        Fuzzing test: Runs multiple games with completely random actions
        to check for crashes or assertion failures in the engine.
        """
        if not self.card_db:
            pytest.skip("No cards loaded")

        # Number of games to simulate
        # For CI/interactive run, keep it relatively small. 
        # User can increase this for long-running stress tests.
        ITERATIONS = 100 
        MAX_STEPS = 2000 # Max steps per game to prevent infinite loops

        print(f"\nStarting Fuzzing Test: {ITERATIONS} games...")
        start_time = time.time()

        crashes = 0
        
        # Get all available card IDs for random deck generation
        all_card_ids = list(self.card_db.keys())
        if not all_card_ids:
            pytest.skip("No cards in database")

        for i in range(ITERATIONS):
            seed = 1000 + i
            # Initialize game with a seed
            gi = dm_ai_module.GameInstance(seed, self.card_db)
            
            # Setup random scenario
            config = dm_ai_module.ScenarioConfig()
            
            # Generate random decks (40 cards)
            deck1 = [random.choice(all_card_ids) for _ in range(40)]
            deck2 = [random.choice(all_card_ids) for _ in range(40)]
            
            # ScenarioConfig does not support deck setup yet, so we manually set them
            # config.my_deck = deck1
            # config.enemy_deck = deck2
            
            # Initial shields (5 cards)
            config.my_shields = [random.choice(all_card_ids) for _ in range(5)]
            # Enemy shield count is int
            config.enemy_shield_count = 5
            
            # Initial hand (5 cards)
            config.my_hand_cards = [random.choice(all_card_ids) for _ in range(5)]
            # Enemy hand is not directly configurable via ScenarioConfig yet in exposed bindings?
            # Actually ScenarioConfig has fields but maybe not all exposed or used.
            # Let's stick to what we used in the working stress script.
            
            gi.reset_with_scenario(config)
            
            # Manually set decks
            gi.state.set_deck(0, deck1)
            gi.state.set_deck(1, deck2)

            state = gi.state
            step_count = 0
            
            try:
                while state.winner == -1 and step_count < MAX_STEPS:
                    # Generate legal actions
                    actions = dm_ai_module.ActionGenerator.generate_legal_actions(state, self.card_db)
                    
                    if not actions:
                        # This should theoretically not happen as PASS is usually available
                        # But if it does, it's a stalemate or bug
                        # print(f"Game {i}: No legal actions at step {step_count}")
                        break
                    
                    # Pick a completely random action
                    action = random.choice(actions)
                    
                    # Execute action
                    dm_ai_module.EffectResolver.resolve_action(state, action, self.card_db)
                    
                    step_count += 1
                
                # Game finished or max steps reached
                # print(f"Game {i} finished in {step_count} steps. Winner: {state.winner}")

            except Exception as e:
                print(f"CRASH DETECTED in Game {i} (Seed {seed}) at step {step_count}!")
                print(f"Error: {e}")
                crashes += 1
                # We don't stop immediately to see if other seeds crash too, 
                # but in a real CI we might want to fail fast.
                # For this script, we count crashes.
        
        end_time = time.time()
        duration = end_time - start_time
        print(f"\nFuzzing Test Completed in {duration:.2f}s")
        print(f"Total Games: {ITERATIONS}")
        print(f"Crashes: {crashes}")

        assert crashes == 0, f"Fuzzing test failed with {crashes} crashes."

if __name__ == "__main__":
    # Allow running this script directly
    t = TestFuzzing()
    t.setup_class()
    t.test_random_actions_fuzzing()
