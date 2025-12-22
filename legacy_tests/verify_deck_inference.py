
import sys
import os
import json
import pytest

# Add bin/ to path to import dm_ai_module
sys.path.append(os.path.join(os.path.dirname(__file__), '../../bin'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../../build'))

try:
    import dm_ai_module
except ImportError:
    pytest.skip("dm_ai_module not found", allow_module_level=True)

def test_deck_inference_logic():
    # Setup DeckInference
    inference = dm_ai_module.DeckInference()

    # Create a temporary meta_decks.json for testing
    meta_decks_data = {
        "decks": [
            {
                "name": "Deck A",
                "cards": [1, 1, 1, 1, 2, 2, 2, 2] # 4x ID 1, 4x ID 2
            },
            {
                "name": "Deck B",
                "cards": [1, 1, 1, 1, 3, 3, 3, 3] # 4x ID 1, 4x ID 3
            }
        ]
    }

    with open("temp_meta_decks.json", "w") as f:
        json.dump(meta_decks_data, f)

    inference.load_decks("temp_meta_decks.json")

    # Initialize GameState
    # Assume we need a card_db. We can just use dummy values since inference only cares about IDs.
    # GameState constructor: GameState(num_cards)
    game_state = dm_ai_module.GameState(100)

    # Setup Opponent (Player 1)
    # Observer is Player 0
    opponent_id = 1

    # Scenario 1: No visible cards. Probabilities should be equal (or uniform).
    probs = inference.infer_probabilities(game_state, 0)
    print("Scenario 1 Probs:", probs)
    assert abs(probs["Deck A"] - 0.5) < 0.01
    assert abs(probs["Deck B"] - 0.5) < 0.01

    # Scenario 2: Opponent has ID 2 in Mana. Deck B (has only 1 and 3) should be impossible.
    # Add ID 2 to opponent mana.
    # We need to manually add cards. GameState has helper `add_card_to_mana(player_id, card_id, instance_id)`
    game_state.add_card_to_mana(opponent_id, 2, 100)

    probs = inference.infer_probabilities(game_state, 0)
    print("Scenario 2 Probs:", probs)
    assert probs["Deck A"] > 0.99
    assert probs["Deck B"] < 0.01

    # Sample hidden cards
    # Should contain remaining cards from Deck A (4x 1, 3x 2)
    hidden_pool = inference.sample_hidden_cards(game_state, 0, 42)
    print("Hidden Pool:", hidden_pool)

    count_1 = hidden_pool.count(1)
    count_2 = hidden_pool.count(2)
    count_3 = hidden_pool.count(3)

    assert count_1 == 4
    assert count_2 == 3
    assert count_3 == 0

    # Cleanup
    os.remove("temp_meta_decks.json")

def test_deck_inference_incompatible():
    inference = dm_ai_module.DeckInference()
    meta_decks_data = {
        "decks": [
            { "name": "Deck A", "cards": [1, 1] }
        ]
    }
    with open("temp_meta_decks_2.json", "w") as f:
        json.dump(meta_decks_data, f)
    inference.load_decks("temp_meta_decks_2.json")

    game_state = dm_ai_module.GameState(100)
    opponent_id = 1

    # Opponent shows ID 2 (not in Deck A)
    game_state.add_card_to_mana(opponent_id, 2, 100)

    probs = inference.infer_probabilities(game_state, 0)
    print("Incompatible Probs:", probs)
    # Should fallback to uniform (1.0 for Deck A) or handle gracefully
    assert probs["Deck A"] > 0.0

    os.remove("temp_meta_decks_2.json")

if __name__ == "__main__":
    test_deck_inference_logic()
    test_deck_inference_incompatible()
