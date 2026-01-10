
import unittest
import json
import os
from dm_toolkit.validator.card_validator import CardValidator

class TestCardValidator(unittest.TestCase):
    def setUp(self):
        self.validator = CardValidator()

    def test_valid_card(self):
        card = {
            "id": 1,
            "name": "Test Card",
            "type": "CREATURE",
            "civilizations": ["FIRE"],
            "effects": [
                {
                    "trigger": "ON_PLAY",
                    "commands": [
                        {
                            "type": "DRAW_CARD",
                            "amount": 1,
                            "from_zone": "DECK",
                            "to_zone": "HAND",
                            "output_value_key": "var_drawn"
                        }
                    ]
                }
            ]
        }
        result = self.validator.validate_card(card)
        self.assertTrue(result.valid, f"Validation failed: {result.errors}")

    def test_missing_fields(self):
        card = {
            "name": "Invalid Card"
        }
        result = self.validator.validate_card(card)
        self.assertFalse(result.valid)
        self.assertIn("Missing 'id'", result.errors)
        self.assertIn("Missing 'type'", result.errors)

    def test_invalid_civilization(self):
        card = {
            "id": 2,
            "name": "Test Card 2",
            "type": "CREATURE",
            "civilizations": ["INVALID_CIV"],
            "effects": []
        }
        result = self.validator.validate_card(card)
        self.assertFalse(result.valid)
        self.assertTrue(any("Invalid civilization" in e for e in result.errors))

    def test_undefined_variable(self):
        card = {
            "id": 3,
            "name": "Var Test Card",
            "type": "SPELL",
            "civilizations": ["WATER"],
            "effects": [
                {
                    "trigger": "ON_PLAY",
                    "commands": [
                        {
                            "type": "DESTROY",
                            "input_value_key": "var_undefined",
                            "from_zone": "BATTLE_ZONE",
                            "to_zone": "GRAVEYARD"
                        }
                    ]
                }
            ]
        }
        result = self.validator.validate_card(card)
        self.assertFalse(result.valid)
        self.assertTrue(any("Reference to undefined variable" in e for e in result.errors))

    def test_defined_variable(self):
        card = {
            "id": 4,
            "name": "Var Test Card 2",
            "type": "SPELL",
            "civilizations": ["WATER"],
            "effects": [
                {
                    "trigger": "ON_PLAY",
                    "commands": [
                        {
                            "type": "QUERY",
                            "output_value_key": "var_query_result",
                            "from_zone": "NONE",
                            "to_zone": "NONE"
                        },
                        {
                            "type": "DESTROY",
                            "input_value_key": "var_query_result",
                            "from_zone": "BATTLE_ZONE",
                            "to_zone": "GRAVEYARD"
                        }
                    ]
                }
            ]
        }
        result = self.validator.validate_card(card)
        self.assertTrue(result.valid, f"Validation failed: {result.errors}")

    def test_special_variable(self):
         card = {
            "id": 5,
            "name": "Special Var Card",
            "type": "SPELL",
            "civilizations": ["WATER"],
            "effects": [
                {
                    "trigger": "ON_PLAY",
                    "commands": [
                        {
                            "type": "DESTROY",
                            "input_value_key": "target",
                            "from_zone": "BATTLE_ZONE",
                            "to_zone": "GRAVEYARD"
                        }
                    ]
                }
            ]
        }
         result = self.validator.validate_card(card)
         self.assertTrue(result.valid, f"Validation failed: {result.errors}")

    def test_nested_commands(self):
         card = {
            "id": 6,
            "name": "Nested Card",
            "type": "SPELL",
            "civilizations": ["WATER"],
            "effects": [
                {
                    "trigger": "ON_PLAY",
                    "commands": [
                         {
                            "type": "QUERY",
                            "output_value_key": "var_outer",
                            "from_zone": "NONE",
                            "to_zone": "NONE",
                            "if_true": [
                                {
                                    "type": "DESTROY",
                                    "input_value_key": "var_outer",
                                    "from_zone": "BATTLE_ZONE",
                                    "to_zone": "GRAVEYARD"
                                }
                            ]
                        }
                    ]
                }
            ]
        }
         result = self.validator.validate_card(card)
         self.assertTrue(result.valid, f"Validation failed: {result.errors}")

if __name__ == '__main__':
    unittest.main()
