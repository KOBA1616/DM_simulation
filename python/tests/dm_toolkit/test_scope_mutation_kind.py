# -*- coding: utf-8 -*-
"""
Test scope unification and mutation_kind migration.
"""

import pytest
from dm_toolkit.consts import TargetScope
from dm_toolkit.gui.editor.data_migration import (
    migrate_modifier_keyword_field,
    normalize_modifier_scope,
    migrate_card_data,
    verify_migration
)
from dm_toolkit.gui.editor.validators_shared import ModifierValidator


class TestTargetScope:
    """Test TargetScope enum functionality."""
    
    def test_scope_constants(self):
        """Test that TargetScope defines correct constants."""
        assert TargetScope.SELF == "SELF"
        assert TargetScope.OPPONENT == "OPPONENT"
        assert TargetScope.ALL == "ALL"
    
    def test_legacy_aliases(self):
        """Test backward compatibility aliases."""
        assert TargetScope.PLAYER_SELF == TargetScope.SELF
        assert TargetScope.PLAYER_OPPONENT == TargetScope.OPPONENT
    
    def test_normalize_legacy_values(self):
        """Test normalize() converts legacy values."""
        assert TargetScope.normalize("PLAYER_SELF") == "SELF"
        assert TargetScope.normalize("PLAYER_OPPONENT") == "OPPONENT"
        assert TargetScope.normalize("SELF") == "SELF"
        assert TargetScope.normalize("ALL") == "ALL"
    
    def test_all_values(self):
        """Test all_values() returns correct list."""
        values = TargetScope.all_values()
        assert "SELF" in values
        assert "OPPONENT" in values
        assert "ALL" in values
        assert len(values) == 3


class TestMutationKindMigration:
    """Test mutation_kind field migration for keyword abilities."""
    
    def test_migrate_grant_keyword(self):
        """Test GRANT_KEYWORD str_val → mutation_kind migration."""
        modifier = {
            "type": "GRANT_KEYWORD",
            "str_val": "speed_attacker",
            "scope": "SELF"
        }
        
        result = migrate_modifier_keyword_field(modifier)
        
        assert result is True
        assert modifier["mutation_kind"] == "speed_attacker"
        assert modifier["str_val"] == "speed_attacker"  # Preserved for compatibility
    
    def test_migrate_set_keyword(self):
        """Test SET_KEYWORD str_val → mutation_kind migration."""
        modifier = {
            "type": "SET_KEYWORD",
            "str_val": "blocker",
            "scope": "OPPONENT"
        }
        
        result = migrate_modifier_keyword_field(modifier)
        
        assert result is True
        assert modifier["mutation_kind"] == "blocker"
    
    def test_no_migration_if_mutation_kind_exists(self):
        """Test migration skipped if mutation_kind already present."""
        modifier = {
            "type": "GRANT_KEYWORD",
            "mutation_kind": "double_breaker",
            "str_val": "old_value",
            "scope": "SELF"
        }
        
        result = migrate_modifier_keyword_field(modifier)
        
        assert result is False
        assert modifier["mutation_kind"] == "double_breaker"  # Unchanged
    
    def test_no_migration_for_non_keyword_types(self):
        """Test migration skipped for COST/POWER modifiers."""
        modifier = {
            "type": "POWER_MODIFIER",
            "value": 2000,
            "scope": "SELF"
        }
        
        result = migrate_modifier_keyword_field(modifier)
        
        assert result is False
        assert "mutation_kind" not in modifier


class TestScopeNormalization:
    """Test scope field normalization."""
    
    def test_normalize_player_self(self):
        """Test PLAYER_SELF → SELF normalization."""
        modifier = {
            "type": "POWER_MODIFIER",
            "value": 1000,
            "scope": "PLAYER_SELF"
        }
        
        result = normalize_modifier_scope(modifier)
        
        assert result is True
        assert modifier["scope"] == "SELF"
    
    def test_normalize_player_opponent(self):
        """Test PLAYER_OPPONENT → OPPONENT normalization."""
        modifier = {
            "type": "COST_MODIFIER",
            "value": -1,
            "scope": "PLAYER_OPPONENT"
        }
        
        result = normalize_modifier_scope(modifier)
        
        assert result is True
        assert modifier["scope"] == "OPPONENT"
    
    def test_no_normalization_if_already_correct(self):
        """Test no change if scope already normalized."""
        modifier = {
            "type": "GRANT_KEYWORD",
            "mutation_kind": "blocker",
            "scope": "SELF"
        }
        
        result = normalize_modifier_scope(modifier)
        
        assert result is False
        assert modifier["scope"] == "SELF"


class TestCardDataMigration:
    """Test full card data migration."""
    
    def test_migrate_multiple_static_abilities(self):
        """Test migrating card with multiple static abilities."""
        card_data = {
            "name": "Test Card",
            "static_abilities": [
                {
                    "type": "GRANT_KEYWORD",
                    "str_val": "speed_attacker",
                    "scope": "PLAYER_SELF"
                },
                {
                    "type": "POWER_MODIFIER",
                    "value": 2000,
                    "scope": "OPPONENT"  # Already normalized
                },
                {
                    "type": "SET_KEYWORD",
                    "str_val": "blocker",
                    "scope": "ALL"
                }
            ]
        }
        
        migrated_count = migrate_card_data(card_data)
        
        # First ability: keyword migration + scope normalization = 2
        # Second ability: no changes = 0
        # Third ability: keyword migration = 1
        assert migrated_count == 3
        
        # Verify first ability
        ability1 = card_data["static_abilities"][0]
        assert ability1["mutation_kind"] == "speed_attacker"
        assert ability1["scope"] == "SELF"
        
        # Verify second ability unchanged
        ability2 = card_data["static_abilities"][1]
        assert ability2["scope"] == "OPPONENT"
        assert "mutation_kind" not in ability2
        
        # Verify third ability
        ability3 = card_data["static_abilities"][2]
        assert ability3["mutation_kind"] == "blocker"


class TestModifierValidation:
    """Test ModifierValidator with mutation_kind."""
    
    def test_validate_grant_keyword_with_mutation_kind(self):
        """Test validation passes with mutation_kind."""
        modifier = {
            "type": "GRANT_KEYWORD",
            "mutation_kind": "speed_attacker",
            "scope": "SELF",
            "condition": {"type": "NONE"},
            "filter": {}
        }
        
        errors = ModifierValidator.validate(modifier)
        
        assert len(errors) == 0
    
    def test_validate_grant_keyword_with_str_val_legacy(self):
        """Test validation passes with legacy str_val (backward compat)."""
        modifier = {
            "type": "GRANT_KEYWORD",
            "str_val": "blocker",
            "scope": "SELF",
            "condition": {"type": "NONE"},
            "filter": {}
        }
        
        errors = ModifierValidator.validate(modifier)
        
        assert len(errors) == 0  # Should pass for backward compatibility
    
    def test_validate_grant_keyword_missing_both_fields(self):
        """Test validation fails when both mutation_kind and str_val are missing."""
        modifier = {
            "type": "GRANT_KEYWORD",
            "scope": "SELF",
            "condition": {"type": "NONE"},
            "filter": {}
        }
        
        errors = ModifierValidator.validate(modifier)
        
        assert len(errors) > 0
        assert any("mutation_kind" in err or "str_val" in err for err in errors)
    
    def test_validate_scope_normalization(self):
        """Test validator accepts normalized scope values."""
        modifier = {
            "type": "POWER_MODIFIER",
            "value": 1000,
            "scope": "SELF",
            "condition": {"type": "NONE"},
            "filter": {}
        }
        
        errors = ModifierValidator.validate(modifier)
        
        assert len(errors) == 0
    
    def test_validate_legacy_scope_normalized(self):
        """Test validator normalizes and accepts legacy scope."""
        modifier = {
            "type": "POWER_MODIFIER",
            "value": 1000,
            "scope": "PLAYER_SELF",  # Legacy format
            "condition": {"type": "NONE"},
            "filter": {}
        }
        
        errors = ModifierValidator.validate(modifier)
        
        # Validator should normalize internally and pass
        assert len(errors) == 0


class TestMigrationVerification:
    """Test migration verification utilities."""
    
    def test_verify_clean_modifier(self):
        """Test verification passes for fully migrated modifier."""
        modifier = {
            "type": "GRANT_KEYWORD",
            "mutation_kind": "speed_attacker",
            "scope": "SELF"
        }
        
        warnings = verify_migration(modifier)
        
        assert len(warnings) == 0
    
    def test_verify_warns_on_str_val_only(self):
        """Test verification warns when using str_val without mutation_kind."""
        modifier = {
            "type": "SET_KEYWORD",
            "str_val": "blocker",
            "scope": "SELF"
        }
        
        warnings = verify_migration(modifier)
        
        assert len(warnings) > 0
        assert any("mutation_kind" in w for w in warnings)
    
    def test_verify_warns_on_legacy_scope(self):
        """Test verification warns when scope not normalized."""
        modifier = {
            "type": "POWER_MODIFIER",
            "value": 2000,
            "scope": "PLAYER_SELF"
        }
        
        warnings = verify_migration(modifier)
        
        assert len(warnings) > 0
        assert any("normalized" in w.lower() for w in warnings)
