# -*- coding: utf-8 -*-
"""
Integration tests for Static/Trigger ability unification.
Validates that the unified infrastructure works end-to-end.
"""

import pytest


class TestUnifiedInfrastructure:
    """Tests for unified Static/Trigger infrastructure."""
    
    def test_text_resources_available(self):
        """CardTextResources provides all required mappings."""
        from dm_toolkit.gui.editor.text_resources import CardTextResources
        
        # Check condition texts
        assert CardTextResources.get_condition_text("DURING_YOUR_TURN") == "自分のターン中、"
        assert CardTextResources.get_condition_text("NONE") == ""
        
        # Check scope texts
        assert CardTextResources.get_scope_text("SELF") == "自分の"
        assert CardTextResources.get_scope_text("OPPONENT") == "相手の"
        assert CardTextResources.get_scope_text("ALL") == ""
        
        # Check trigger texts
        assert "クリーチャー" in CardTextResources.get_trigger_text("ON_PLAY", is_spell=False)
        assert "呪文" in CardTextResources.get_trigger_text("ON_PLAY", is_spell=True)
        
        # Check keyword texts
        assert CardTextResources.get_keyword_text("BLOCKER") == "ブロッカー"
        assert CardTextResources.get_keyword_text("blocker") == "ブロッカー"
    
    @pytest.mark.skip(reason="Requires PyQt6 GUI environment")
    def test_unified_filter_handler_static(self):
        """UnifiedFilterHandler creates STATIC-configured widgets."""
        from dm_toolkit.gui.editor.unified_filter_handler import UnifiedFilterHandler
        
        # Create static filter widget
        widget = UnifiedFilterHandler.create_filter_widget("STATIC")
        assert widget is not None
        
        # Apply scope override
        filter_def = {"min_cost": 3, "max_cost": 5}
        result = UnifiedFilterHandler.apply_scope_override(filter_def, "SELF")
        assert result["owner"] == "SELF"
        assert result["min_cost"] == 3
    
    @pytest.mark.skip(reason="Requires PyQt6 GUI environment")
    def test_unified_filter_handler_trigger(self):
        """UnifiedFilterHandler creates TRIGGER-configured widgets."""
        from dm_toolkit.gui.editor.unified_filter_handler import UnifiedFilterHandler
        
        # Create trigger filter widget
        widget = UnifiedFilterHandler.create_filter_widget("TRIGGER")
        assert widget is not None
        
        # Validate context
        errors = UnifiedFilterHandler.validate_filter_for_context(
            {"is_tapped": 1},
            "TRIGGER"
        )
        assert len(errors) > 0  # Should warn about is_tapped
    
    @pytest.mark.skip(reason="Requires PyQt6 GUI environment")
    def test_keyword_selector_widget(self):
        """KeywordSelectorWidget provides unified keyword selection."""
        from dm_toolkit.gui.editor.forms.parts.keyword_selector import KeywordSelectorWidget
        
        # Create widget
        widget = KeywordSelectorWidget(allow_settable=True)
        assert widget.count() > 0  # Should have keywords
        
        # Test set/get
        widget.set_keyword("BLOCKER")
        assert widget.get_keyword() == "BLOCKER"
    
    def test_validators_integration(self):
        """Validators work together for complete validation."""
        from dm_toolkit.gui.editor.validators_shared import (
            ModifierValidator,
            TriggerEffectValidator,
            AbilityContextValidator
        )
        
        # Valid modifier
        modifier = {
            "type": "COST_MODIFIER",
            "value": -2,
            "scope": "ALL",
            "condition": {"type": "NONE"},
            "filter": {}
        }
        errors = ModifierValidator.validate(modifier)
        assert len(errors) == 0
        
        # Valid trigger
        trigger = {
            "trigger": "ON_PLAY",
            "condition": {"type": "NONE"},
            "commands": [{"type": "TRANSITION"}]  # Add non-empty command
        }
        errors = TriggerEffectValidator.validate(trigger)
        assert len(errors) == 0
        
        # Context validator correctly identifies types
        ability_type, _ = AbilityContextValidator.validate_effect_or_modifier(modifier)
        assert ability_type == "STATIC"
        
        ability_type, _ = AbilityContextValidator.validate_effect_or_modifier(trigger)
        assert ability_type == "TRIGGER"
    
    def test_text_generator_uses_resources(self):
        """TextGenerator uses CardTextResources."""
        from dm_toolkit.gui.editor.text_generator import CardTextGenerator
        from dm_toolkit.gui.editor.text_resources import CardTextResources
        
        # Test trigger_to_japanese uses resources
        result = CardTextGenerator.trigger_to_japanese("ON_PLAY", is_spell=False)
        expected = CardTextResources.get_trigger_text("ON_PLAY", is_spell=False)
        assert result == expected
        
        # Test _format_modifier uses resources
        modifier = {
            "type": "GRANT_KEYWORD",
            "str_val": "BLOCKER",
            "scope": "SELF",
            "condition": {"type": "NONE"},
            "filter": {}
        }
        result = CardTextGenerator._format_modifier(modifier)
        assert "ブロッカー" in result
        assert "自分の" in result
    
    @pytest.mark.skip(reason="Requires PyQt6 GUI environment")
    def test_data_manager_validation(self):
        """DataManager uses validators for normalization."""
        from dm_toolkit.gui.editor.data_manager import CardDataManager
        from PyQt6.QtGui import QStandardItemModel
        
        model = QStandardItemModel()
        manager = CardDataManager(model)
        
        # Create a card with static ability
        card = {
            "name": "Test Card",
            "type": "CREATURE",
            "static_abilities": [
                {
                    "type": "COST_MODIFIER",
                    "value": -1,
                    "scope": "SELF",
                    "condition": {"type": "NONE"},
                    "filter": {}
                }
            ],
            "triggers": [
                {
                    "trigger": "ON_PLAY",
                    "condition": {"type": "NONE"},
                    "commands": []
                }
            ]
        }
        
        # Normalize should validate
        warnings = manager._normalize_card_for_engine(card)
        # Should have minimal warnings (empty commands list)
        assert isinstance(warnings, list)


class TestBackwardCompatibility:
    """Tests for backward compatibility with existing code."""
    
    def test_keyword_translation_still_available(self):
        """KEYWORD_TRANSLATION still accessible for legacy code."""
        from dm_toolkit.gui.editor.text_generator import CardTextGenerator
        
        # Legacy attribute should still exist
        assert hasattr(CardTextGenerator, 'KEYWORD_TRANSLATION')
        assert CardTextGenerator.KEYWORD_TRANSLATION.get('blocker') == "ブロッカー"
    
    def test_grantable_keywords_defined(self):
        """GRANTABLE_KEYWORDS and SETTABLE_KEYWORDS defined."""
        from dm_toolkit.consts import GRANTABLE_KEYWORDS, SETTABLE_KEYWORDS
        
        assert len(GRANTABLE_KEYWORDS) > 0
        assert len(SETTABLE_KEYWORDS) > 0
        assert "blocker" in GRANTABLE_KEYWORDS or "BLOCKER" in GRANTABLE_KEYWORDS


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
