# -*- coding: utf-8 -*-
"""
Unified Filter Handler for Static Abilities and Trigger Effects.
Provides common widget creation and filter text generation.
"""

from typing import Dict, Any, Optional
from dm_toolkit.gui.editor.forms.parts.filter_widget import FilterEditorWidget
from dm_toolkit.gui.editor.text_generator import CardTextGenerator


class UnifiedFilterHandler:
    """
    Handles common filter widget creation and management for both
    static abilities and trigger effects with context-aware configuration.
    """
    
    @staticmethod
    def create_filter_widget(ability_type: str, parent=None) -> FilterEditorWidget:
        """
        Create a FilterEditorWidget configured for the specified ability type.
        
        Args:
            ability_type: "STATIC" or "TRIGGER"
            parent: Parent widget (optional)
        
        Returns:
            Configured FilterEditorWidget
        """
        widget = FilterEditorWidget(parent)
        
        if ability_type == "STATIC":
            # Static abilities: emphasize owner/scope filters and numeric properties
            widget.set_visible_sections({
                'owner': True,         # Scope (SELF/OPPONENT/ALL)
                'basic': True,         # Type, civilization
                'stats': True,         # Cost, power ranges
                'flags': True,         # Tapped, blocker, evolution flags
                'keywords': False,     # Keywords (less relevant for static)
                'selection': False     # Selection (not applicable)
            })
        
        elif ability_type == "TRIGGER":
            # Trigger effects: exclude flags that change dynamically
            widget.set_visible_sections({
                'owner': True,         # Target specification
                'basic': True,         # Type, civilization
                'stats': True,         # Cost, power
                'flags': False,        # Tapped, blocker (change during game, unreliable)
                'keywords': False,
                'selection': False
            })
        
        else:
            # Default: show all available sections
            widget.set_visible_sections({
                'owner': True,
                'basic': True,
                'stats': True,
                'flags': True,
                'keywords': False,
                'selection': False
            })
        
        return widget
    
    @staticmethod
    def apply_scope_override(filter_def: Dict[str, Any], scope: str) -> Dict[str, Any]:
        """
        Apply scope/owner override to filter definition.
        Static abilities can override the target owner based on scope setting.
        
        Args:
            filter_def: Original filter definition
            scope: "SELF", "OPPONENT", or "ALL"
        
        Returns:
            Filter definition with scope applied (copies original)
        """
        result = filter_def.copy() if filter_def else {}
        
        # Apply scope as owner override for static abilities only
        if scope == "SELF":
            result['owner'] = "SELF"
        elif scope == "OPPONENT":
            result['owner'] = "OPPONENT"
        elif scope == "ALL":
            # Remove owner constraint for ALL scope
            result.pop('owner', None)
        
        return result
    
    @staticmethod
    def format_filter_text(
        filter_def: Dict[str, Any],
        is_static: bool = False,
        scope: str = "ALL"
    ) -> str:
        """
        Generate Japanese text for filter specification.
        
        Args:
            filter_def: Filter definition dict
            is_static: True for static abilities (affects phrasing)
            scope: Scope modifier ("SELF", "OPPONENT", "ALL")
        
        Returns:
            Japanese filter description
        """
        if not filter_def:
            return "\x91Ώ\xdb" if not is_static else ""
        
        # Apply scope override to filter for display
        effective_filter = UnifiedFilterHandler.apply_scope_override(filter_def, scope)
        
        # Delegate to TextGenerator's existing filter formatting
        return CardTextGenerator._format_modifier_target(effective_filter)
    
    @staticmethod
    def validate_filter_for_context(filter_def: Dict[str, Any], ability_type: str) -> list:
        """
        Validate filter definition for the given ability context.
        
        Args:
            filter_def: Filter definition to validate
            ability_type: "STATIC" or "TRIGGER"
        
        Returns:
            List of validation error messages (empty if valid)
        """
        from dm_toolkit.gui.editor.validators_shared import FilterValidator
        
        errors = FilterValidator.validate(filter_def)
        
        # Context-specific validation
        if ability_type == "TRIGGER":
            # Trigger effects should not use dynamic flags
            if filter_def.get('is_tapped') is not None:
                errors.append(
                    "Trigger effects should not specify 'is_tapped' "
                    "(changes during gameplay)"
                )
        
        return errors
