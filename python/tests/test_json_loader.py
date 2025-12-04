import pytest
import os
import sys

# Add bin directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../bin'))

import dm_ai_module
from dm_ai_module import GameState, CardDefinition, CardType, Civilization, GameResult, ActionType, ActionGenerator

class TestJsonLoader:
    def test_load_json_cards(self):
        """Test loading cards from JSON."""
        filepath = "data/test_cards.json"

        # Create dummy test data if it doesn't exist
        # Note: The JSON structure must match what src/core/card_json_types.hpp expects.
        if not os.path.exists(filepath):
            os.makedirs("data", exist_ok=True)
            with open(filepath, "w") as f:
                f.write("""
                [
                    {
                        "id": 1001,
                        "name": "Bronze-Arm Tribe",
                        "type": "CREATURE",
                        "civilization": "NATURE",
                        "races": ["Beast Folk"],
                        "cost": 3,
                        "power": 1000,
                        "effects": []
                    },
                    {
                        "id": 1002,
                        "name": "Aqua Hulcus",
                        "type": "CREATURE",
                        "civilization": "WATER",
                        "races": ["Liquid People"],
                        "cost": 3,
                        "power": 2000,
                        "effects": []
                    }
                ]
                """)

        # Load cards using the new module
        cards = dm_ai_module.JsonLoader.load_cards(filepath)

        assert len(cards) > 0

        # Verify first card content (Bronze-Arm Tribe)
        assert 1001 in cards
        bat = cards[1001]

        assert bat.name == "Bronze-Arm Tribe"
        assert bat.cost == 3
        assert bat.power == 1000

        # Checking civilization
        # In C++, Civilization is likely an enum. Pybind11 exposes enums as objects that compare equal to their int value,
        # but strict type checking (==) might fail if one is 'Civilization.NATURE' and other is '16'.
        # However, `bat.civilization` returned <Civilization.NATURE: 16> and we compared it to int(Civilization.NATURE) which is 16.
        # The error "assert <Civilization.NATURE: 16> == 16" suggests they are NOT equal.
        # It's better to compare against the Enum member itself.
        assert bat.civilization == Civilization.NATURE
