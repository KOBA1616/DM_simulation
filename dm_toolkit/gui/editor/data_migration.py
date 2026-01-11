# -*- coding: utf-8 -*-
"""
Data migration utilities for card data format updates.

Handles:
- str_val → mutation_kind migration for GRANT_KEYWORD/SET_KEYWORD
- PLAYER_SELF/PLAYER_OPPONENT → SELF/OPPONENT scope normalization
"""

from typing import Dict, Any, List
from dm_toolkit.consts import TargetScope


def migrate_modifier_keyword_field(modifier: Dict[str, Any]) -> bool:
    """
    Migrate legacy str_val to mutation_kind for keyword modifiers.
    
    Args:
        modifier: Modifier dictionary to migrate
    
    Returns:
        True if migration was performed, False if no changes needed
    """
    mtype = modifier.get('type', '')
    
    # Only migrate keyword types
    if mtype not in ('GRANT_KEYWORD', 'SET_KEYWORD'):
        return False
    
    # Check if mutation_kind already exists
    if modifier.get('mutation_kind'):
        return False  # Already migrated
    
    # Migrate str_val to mutation_kind
    str_val = modifier.get('str_val', '')
    if str_val:
        modifier['mutation_kind'] = str_val
        print(f"[Migration] Migrated keyword: {mtype} str_val='{str_val}' → mutation_kind")
        return True
    
    return False


def normalize_modifier_scope(modifier: Dict[str, Any]) -> bool:
    """
    Normalize scope field to use TargetScope constants.
    
    Args:
        modifier: Modifier dictionary to normalize
    
    Returns:
        True if normalization was performed, False if no changes needed
    """
    scope = modifier.get('scope')
    if not scope:
        return False
    
    normalized = TargetScope.normalize(scope)
    if normalized != scope:
        modifier['scope'] = normalized
        print(f"[Migration] Normalized scope: '{scope}' → '{normalized}'")
        return True
    
    return False


def migrate_card_data(card_data: Dict[str, Any]) -> int:
    """
    Migrate all static abilities in a card to use mutation_kind and normalized scope.
    
    Args:
        card_data: Complete card data dictionary
    
    Returns:
        Number of modifiers migrated
    """
    migrated = 0
    
    static_abilities = card_data.get('static_abilities', [])
    for modifier in static_abilities:
        # Migrate keyword field
        if migrate_modifier_keyword_field(modifier):
            migrated += 1
        
        # Normalize scope
        if normalize_modifier_scope(modifier):
            migrated += 1
    
    return migrated


def batch_migrate_cards(cards: List[Dict[str, Any]]) -> Dict[str, int]:
    """
    Migrate multiple cards.
    
    Args:
        cards: List of card data dictionaries
    
    Returns:
        Migration statistics
    """
    stats = {
        'total_cards': len(cards),
        'cards_migrated': 0,
        'modifiers_migrated': 0
    }
    
    for card in cards:
        before_count = stats['modifiers_migrated']
        migrated_count = migrate_card_data(card)
        stats['modifiers_migrated'] += migrated_count
        
        if migrated_count > 0:
            stats['cards_migrated'] += 1
    
    return stats


def verify_migration(modifier: Dict[str, Any]) -> List[str]:
    """
    Verify that a modifier has been properly migrated.
    
    Args:
        modifier: Modifier to verify
    
    Returns:
        List of warnings (empty if clean)
    """
    warnings = []
    mtype = modifier.get('type', '')
    
    # Check keyword types have mutation_kind
    if mtype in ('GRANT_KEYWORD', 'SET_KEYWORD'):
        if not modifier.get('mutation_kind'):
            if modifier.get('str_val'):
                warnings.append(f"Keyword type {mtype} still using str_val, should use mutation_kind")
            else:
                warnings.append(f"Keyword type {mtype} missing both mutation_kind and str_val")
    
    # Check scope is normalized
    scope = modifier.get('scope', TargetScope.ALL)
    normalized = TargetScope.normalize(scope)
    if scope != normalized:
        warnings.append(f"Scope '{scope}' should be normalized to '{normalized}'")
    
    return warnings
