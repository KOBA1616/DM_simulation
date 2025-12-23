
import sys
import os
import pytest

# Add bin to path to import dm_ai_module
sys.path.append(os.path.join(os.getcwd(), 'bin'))

try:
    import dm_ai_module
except ImportError:
    pass

@pytest.mark.skipif('dm_ai_module' not in sys.modules, reason="requires dm_ai_module C++ extension")
def test_beam_search_logic():
    """
    Verifies the Beam Search Evaluator logic, specifically Opponent Danger detection.
    Migrated from legacy_tests/verify_beam_search.py.
    """

    # 1. Setup Card DB with a Key Card
    def_key = dm_ai_module.CardDefinition()
    def_key.id = 1000
    def_key.is_key_card = True
    def_key.ai_importance_score = 100
    def_key.cost = 5
    def_key.civilizations = [dm_ai_module.Civilization.FIRE]
    def_key.type = dm_ai_module.CardType.CREATURE
    def_key.races = ["Dragon"]

    def_dummy = dm_ai_module.CardDefinition()
    def_dummy.id = 999
    def_dummy.cost = 1
    def_dummy.civilizations = [dm_ai_module.Civilization.FIRE]
    def_dummy.type = dm_ai_module.CardType.CREATURE
    def_dummy.races = ["Human"]

    card_db = {1000: def_key, 999: def_dummy}

    # 2. Initialize Evaluator
    if not hasattr(dm_ai_module, 'BeamSearchEvaluator'):
        pytest.skip("BeamSearchEvaluator not exposed in dm_ai_module")

    evaluator = dm_ai_module.BeamSearchEvaluator(card_db, 7, 2)

    # 3. Create State
    state = dm_ai_module.GameState(42)
    state.turn_number = 1
    state.active_player_id = 0

    # Populate decks
    for i in range(20):
        state.add_card_to_deck(0, 999, 10000 + i)
        state.add_card_to_deck(1, 999, 20000 + i)

    # Add Key Card to Opponent's Hand (Player 1)
    state.add_card_to_hand(1, 1000, 101) # Opponent has Key Card

    # Give Player 0 a dummy card to play/charge so simulation can proceed
    state.add_card_to_hand(0, 999, 201)
    state.add_card_to_mana(0, 999, 202) # To pay cost

    # Run Eval
    policy, value = evaluator.evaluate(state)

    # We expect a negative value because of Opponent Danger
    assert value < -50, f"Score {value} is not negative enough (Expected < -50 due to Opponent Danger)"
