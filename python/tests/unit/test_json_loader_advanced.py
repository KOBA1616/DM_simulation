
import sys
import os
import unittest
import pytest
import json

# Add bin directory to path if not present (assuming run from root with PYTHONPATH set, but adding logic just in case)
bin_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../bin'))
if bin_path not in sys.path:
    sys.path.append(bin_path)

try:
    import dm_ai_module
    from dm_ai_module import Civilization, TriggerType
except ImportError:
    pass # Will skip in test setup if not found

class TestJsonLoaderAdvanced(unittest.TestCase):
    def setUp(self):
        if 'dm_ai_module' not in sys.modules:
            self.skipTest("dm_ai_module not found")

    def test_advanced_features(self):
        filepath = "data/test_advanced_cards.json"

        card_data = [
            {
                "id": 2001,
                "name": "Evolution Totem",
                "type": "CREATURE",
                "civilizations": ["NATURE"],
                "cost": 5,
                "power": 5000,
                "evolution_condition": {
                    "races": ["Beast Folk"],
                    "civilizations": ["NATURE"]
                },
                "keywords": {
                    "evolution": True
                }
            },
            {
                "id": 2002,
                "name": "Friend Burst Creature",
                "type": "CREATURE",
                "civilizations": ["FIRE"],
                "cost": 3,
                "power": 3000,
                "keywords": {
                    "friend_burst": True
                }
            }
        ]

        if not os.path.exists("data"):
            os.makedirs("data", exist_ok=True)

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(card_data, f, ensure_ascii=False)

        try:
            cards = dm_ai_module.JsonLoader.load_cards(filepath)

            # Test Evolution Condition
            self.assertIn(2001, cards)
            evo = cards[2001]
            self.assertTrue(evo.keywords.evolution)
            # Check evolution_condition presence
            self.assertIsNotNone(evo.evolution_condition)
            self.assertIn("Beast Folk", evo.evolution_condition.races)
            self.assertIn(Civilization.NATURE, evo.evolution_condition.civilizations)

            # Test Friend Burst
            self.assertIn(2002, cards)
            fb = cards[2002]
            self.assertTrue(fb.keywords.friend_burst)
            # Check if effect was generated
            has_fb_effect = False
            for eff in fb.effects:
                if eff.trigger == TriggerType.ON_PLAY:
                    has_fb_effect = True
                    break

            self.assertTrue(has_fb_effect, "Friend Burst effect should be generated")

        finally:
            if os.path.exists(filepath):
                os.remove(filepath)

if __name__ == '__main__':
    unittest.main()
