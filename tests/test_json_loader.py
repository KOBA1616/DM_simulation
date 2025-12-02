import sys
import os
import pytest
import json

# Add the bin directory to sys.path
sys.path.append(os.path.join(os.getcwd(), 'bin'))

try:
    import dm_ai_module
except ImportError:
    pass

class TestJsonLoader:
    def test_load_json_cards(self):
        """Test loading cards from JSON."""
        filepath = "data/test_cards.json"

        if not os.path.exists(filepath):
            pytest.skip("Test data not found")

        # Load cards using the new module
        cards = dm_ai_module.JsonLoader.load_cards(filepath)

        assert len(cards) >= 1

        card = cards[9999]
        assert card.name == "Test Json Creature"
        assert card.civilization == dm_ai_module.Civilization.FIRE
        assert card.cost == 2
        assert card.power == 2000
        assert card.type == dm_ai_module.CardType.CREATURE

        # Check keywords inferred from effects
        # S_TRIGGER should be set
        assert card.keywords.shield_trigger == True

        # We did not implement SPEED_ATTACKER inference in C++ yet,
        # but let's check if the basic properties are correct.
        # If we update the C++ loader to parse PASSIVE_CONST actions for keywords, we can test it here.

        # Test that these cards can be used in GameInstance
        card_db = { 9999: card }
        gi = dm_ai_module.GameInstance(42, card_db)

        # Verify deck setup works with this card
        s = gi.state
        s.set_deck(0, [9999, 9999, 9999, 9999])
        assert len(s.players[0].deck) == 4
        assert s.players[0].deck[0].card_id == 9999
