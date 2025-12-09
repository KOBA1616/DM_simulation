
import sys
import os

# Add bin to path to import dm_ai_module
sys.path.append(os.path.abspath("bin"))

try:
    import dm_ai_module
except ImportError:
    print("Could not import dm_ai_module. Please build the project first.")
    sys.exit(1)

def verify_beam_search():
    print("Verifying Beam Search Evaluator...")

    # 1. Setup Card DB with a Key Card
    def_key = dm_ai_module.CardDefinition()
    def_key.id = 1000
    def_key.is_key_card = True
    def_key.ai_importance_score = 100
    def_key.cost = 5
    def_key.civilizations = [dm_ai_module.Civilization.FIRE]

    def_dummy = dm_ai_module.CardDefinition()
    def_dummy.id = 999
    def_dummy.cost = 1
    def_dummy.civilizations = [dm_ai_module.Civilization.FIRE]

    card_db = {1000: def_key, 999: def_dummy}

    # 2. Initialize Evaluator
    evaluator = dm_ai_module.BeamSearchEvaluator(card_db, 7, 2)

    # 3. Create State
    state = dm_ai_module.GameState(42)
    state.turn_number = 1
    state.active_player_id = 0

    # Populate decks to prevent immediate game over (Draw/Loss)
    deck_cards = [999] * 20
    state.set_deck(0, deck_cards)
    state.set_deck(1, deck_cards)

    # Add Key Card to Opponent's Hand (Player 1)
    state.add_card_to_hand(1, 1000, 101) # Opponent has Key Card

    # Evaluate
    # Give Player 0 a dummy card to play/charge so simulation can proceed
    state.add_card_to_hand(0, 999, 201)
    state.add_card_to_mana(0, 999, 202) # To pay cost

    # Run Eval
    print("Running evaluation...")
    policy, value = evaluator.evaluate(state)

    print(f"Value: {value}")

    # We expect a negative value because of Opponent Danger (Key Card in hand = +Danger -> -Score)
    # Danger = 100. Resource diff small.
    if value < -50:
        print("Success: High negative score detected due to Opponent Danger.")
    else:
        print(f"Failure: Score {value} is not negative enough (Expected around -99.5).")

    print("Beam Search verification script finished.")

if __name__ == "__main__":
    verify_beam_search()
