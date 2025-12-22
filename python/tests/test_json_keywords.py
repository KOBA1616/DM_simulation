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

class TestJsonKeywords:
    def test_load_json_keywords(self):
        """Test loading cards from JSON with various keywords."""
        filepath = "data/test_keywords.json"

        if not os.path.exists(filepath):
            pytest.skip("Test data not found")

        # Load cards using the new module
        cards = dm_ai_module.JsonLoader.load_cards(filepath)

        assert 8888 in cards
        card = cards[8888]
        assert card.name == "Test Keyword Creature"

        # Test Triggers
        assert card.keywords.cip == True, "ON_PLAY should set cip"
        assert card.keywords.at_attack == True, "ON_ATTACK should set at_attack"
        assert card.keywords.destruction == True, "ON_DESTROY should set destruction"

        # Test Passive Keywords
        assert card.keywords.blocker == True, "BLOCKER keyword missing"
        assert card.keywords.speed_attacker == True, "SPEED_ATTACKER keyword missing"
        assert card.keywords.slayer == True, "SLAYER keyword missing"
        assert card.keywords.double_breaker == True, "DOUBLE_BREAKER keyword missing"
        assert card.keywords.triple_breaker == True, "TRIPLE_BREAKER keyword missing"

        # Test Power Attacker
        assert card.keywords.power_attacker == True, "POWER_ATTACKER keyword missing"
        assert card.power_attacker_bonus == 4000, "POWER_ATTACKER bonus incorrect"
