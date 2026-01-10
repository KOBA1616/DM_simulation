
import pytest
import json
import os
from dm_toolkit.validator.card_validator import CardValidator, ValidationResult

class TestCardValidator:
    @pytest.fixture
    def validator(self):
        return CardValidator()

    def test_valid_card(self, validator):
        card_data = {
            "id": 1,
            "name": "Test Card",
            "civilizations": ["FIRE"],
            "type": "CREATURE",
            "cost": 3,
            "triggers": [
                {
                    "trigger": "ON_PLAY",
                    "commands": [
                        {
                            "type": "DRAW_CARD",
                            "from_zone": "DECK",
                            "to_zone": "HAND",
                            "output_value_key": "drawn_card"
                        }
                    ]
                }
            ]
        }
        result = validator.validate_card(card_data)
        assert result.is_valid
        assert len(result.errors) == 0

    def test_invalid_command_type(self, validator):
        card_data = {
            "id": 1,
            "name": "Test Card",
            "civilizations": ["FIRE"],
            "type": "CREATURE",
            "cost": 3,
            "triggers": [
                {
                    "trigger": "ON_PLAY",
                    "commands": [
                        {
                            "type": "INVALID_COMMAND",
                            "from_zone": "DECK",
                            "to_zone": "HAND"
                        }
                    ]
                }
            ]
        }
        result = validator.validate_card(card_data)
        assert not result.is_valid
        assert any("Unknown command type 'INVALID_COMMAND'" in err for err in result.errors)

    def test_invalid_zone(self, validator):
        card_data = {
            "id": 1,
            "name": "Test Card",
            "civilizations": ["FIRE"],
            "type": "CREATURE",
            "cost": 3,
            "triggers": [
                {
                    "trigger": "ON_PLAY",
                    "commands": [
                        {
                            "type": "TRANSITION",
                            "from_zone": "INVALID_ZONE",
                            "to_zone": "HAND"
                        }
                    ]
                }
            ]
        }
        result = validator.validate_card(card_data)
        assert not result.is_valid
        assert any("Unknown from_zone 'INVALID_ZONE'" in err for err in result.errors)

    def test_variable_reference_error(self, validator):
        card_data = {
            "id": 1,
            "name": "Test Card",
            "civilizations": ["FIRE"],
            "type": "CREATURE",
            "cost": 3,
            "triggers": [
                {
                    "trigger": "ON_PLAY",
                    "commands": [
                        {
                            "type": "TRANSITION",
                            "input_value_key": "undefined_var"
                        }
                    ]
                }
            ]
        }
        result = validator.validate_card(card_data)
        assert not result.is_valid
        assert any("Reference to undefined variable 'undefined_var'" in err for err in result.errors)

    def test_variable_reference_success(self, validator):
        card_data = {
            "id": 1,
            "name": "Test Card",
            "civilizations": ["FIRE"],
            "type": "CREATURE",
            "cost": 3,
            "triggers": [
                {
                    "trigger": "ON_PLAY",
                    "commands": [
                        {
                            "type": "QUERY",
                            "output_value_key": "var_1"
                        },
                        {
                            "type": "TRANSITION",
                            "input_value_key": "var_1"
                        }
                    ]
                }
            ]
        }
        result = validator.validate_card(card_data)
        assert result.is_valid

    def test_missing_required_fields(self, validator):
        card_data = {
            "id": 1
            # Missing name, civilizations, type, cost
        }
        result = validator.validate_card(card_data)
        assert not result.is_valid
        assert any("Missing required field: name" in err for err in result.errors)

    def test_nested_command_validation(self, validator):
        card_data = {
            "id": 1,
            "name": "Nested Test Card",
            "civilizations": ["LIGHT"],
            "type": "SPELL",
            "cost": 5,
            "triggers": [
                {
                    "trigger": "ON_CAST_SPELL",
                    "commands": [
                        {
                            "type": "IF",
                            "if_true": [
                                {
                                    "type": "INVALID_NESTED_COMMAND",
                                }
                            ]
                        }
                    ]
                }
            ]
        }
        result = validator.validate_card(card_data)
        assert not result.is_valid
        assert any("Unknown command type 'INVALID_NESTED_COMMAND'" in err for err in result.errors)

    def test_variable_scope_in_nested_block(self, validator):
        card_data = {
            "id": 1,
            "name": "Scope Test Card",
            "civilizations": ["DARKNESS"],
            "type": "CREATURE",
            "cost": 4,
            "triggers": [
                {
                    "trigger": "ON_PLAY",
                    "commands": [
                        {
                            "type": "QUERY",
                            "output_value_key": "outer_var"
                        },
                        {
                            "type": "IF",
                            "if_true": [
                                {
                                    "type": "TRANSITION",
                                    "input_value_key": "outer_var"  # Should be valid
                                },
                                {
                                    "type": "QUERY",
                                    "output_value_key": "inner_var"
                                }
                            ]
                        },
                        {
                            "type": "TRANSITION",
                            "input_value_key": "inner_var" # Should be invalid if strictly scoped
                        }
                    ]
                }
            ]
        }
        result = validator.validate_card(card_data)
        # Based on current implementation (block scope), inner_var should NOT be available outside
        assert not result.is_valid
        assert any("Reference to undefined variable 'inner_var'" in err for err in result.errors)

    def test_triggers_and_effects_both_checked(self, validator):
        card_data = {
            "id": 1,
            "name": "Both Lists Card",
            "civilizations": ["NATURE"],
            "type": "CREATURE",
            "cost": 2,
            "triggers": [
                {
                    "commands": [{"type": "INVALID_TRIGGER_CMD"}]
                }
            ],
            "effects": [
                {
                    "commands": [{"type": "INVALID_EFFECT_CMD"}]
                }
            ]
        }
        result = validator.validate_card(card_data)
        assert not result.is_valid
        errors_str = str(result.errors)
        assert "Unknown command type 'INVALID_TRIGGER_CMD'" in errors_str
        assert "Unknown command type 'INVALID_EFFECT_CMD'" in errors_str
