import unittest
import json
import os
import sys
from types import ModuleType

# Add bin/ to sys.path to import dm_ai_module
sys.path.append(os.path.join(os.path.dirname(__file__), '../../bin'))
dm_ai_module: ModuleType | None
try:
    import dm_ai_module as _dm_ai_module  # type: ignore
    dm_ai_module = _dm_ai_module
except ImportError:
    print("Warning: dm_ai_module not found. Tests relying on C++ extension will fail.")
    dm_ai_module = None

class TestNewCardFeatures(unittest.TestCase):
    def setUp(self):
        if not dm_ai_module:
            self.skipTest("dm_ai_module not available")

        # Create a temporary JSON file with new features
        self.test_json_path = "test_cards_features.json"
        self.card_data = [
            {
                "id": 9001,
                "name": "Tamaseed Test",
                "civilization": "FIRE",
                "type": "TAMASEED",
                "cost": 3,
                "power": 0,
                "races": ["Dragon"],
                "effects": [
                    {
                        "trigger": "ON_PLAY",
                        "condition": {"type": "NONE"},
                        "actions": [
                            {
                                "type": "DISCARD",
                                "scope": "TARGET_SELECT",
                                "value1": 1,
                                "target_player": "OPPONENT",
                                "optional": True
                            }
                        ]
                    }
                ]
            },
            {
                "id": 9002,
                "name": "Reanimate Test",
                "civilization": "DARKNESS",
                "type": "SPELL",
                "cost": 5,
                "power": 0,
                "races": [],
                "effects": [
                    {
                        "trigger": "ON_PLAY",
                        "condition": {"type": "NONE"},
                        "actions": [
                            {
                                "type": "PLAY_FROM_ZONE",
                                "scope": "TARGET_SELECT",
                                "source_zone": "GRAVEYARD",
                                "filter": {
                                    "types": ["CREATURE"],
                                    "max_cost": 4,
                                    "is_evolution": False
                                }
                            }
                        ]
                    }
                ]
            }
        ]

        with open(self.test_json_path, 'w') as f:
            json.dump(self.card_data, f)

    def tearDown(self):
        if os.path.exists(self.test_json_path):
            os.remove(self.test_json_path)

    def test_load_cards(self):
        if not dm_ai_module:
            self.skipTest("dm_ai_module not available")

        try:
            assert dm_ai_module is not None
            cards = dm_ai_module.JsonLoader.load_cards(self.test_json_path)
            self.assertIn(9001, cards)
            self.assertIn(9002, cards)

            # Check properties if exposed, but currently only map[int, CardDefinition] returned
            # which is opaque in Python unless bindings expose fields.
            # Assuming succesful return means parsing passed.
            print("New features JSON loaded successfully.")

        except Exception as e:
            self.fail(f"JsonLoader failed to load new features: {e}")

if __name__ == '__main__':
    unittest.main()
