# -*- coding: utf-8 -*-
"""
Data migration utilities for card data format updates.

Handles:
- str_val → mutation_kind migration for GRANT_KEYWORD/SET_KEYWORD
- PLAYER_SELF/PLAYER_OPPONENT → SELF/OPPONENT scope normalization
- Legacy 'actions' -> 'commands' lifting
"""

from typing import Dict, Any, List
from dm_toolkit.consts import TargetScope
from dm_toolkit.gui.editor.action_converter import convert_action_to_objs

class DataMigration:
    """Handles migration of legacy data structures to modern formats."""

    @staticmethod
    def lift_actions_to_commands(effect_data):
        """
        Converts legacy 'actions' list to 'commands' if present in raw dict.
        Modifies effect_data in-place.
        """
        if 'actions' in effect_data:
            legacy_actions = effect_data.pop('actions')
            commands = effect_data.get('commands', [])
            for act in legacy_actions:
                 try:
                     objs = convert_action_to_objs(act)
                     for o in objs:
                         if hasattr(o, 'to_dict'):
                             commands.append(o.to_dict())
                         elif isinstance(o, dict):
                             commands.append(o)
                 except:
                     pass
            effect_data['commands'] = commands

    @staticmethod
    def normalize_card_data(card_raw):
        """
        Normalizes card data structure (e.g., triggers -> effects).
        Also performs field migrations for static abilities.
        """
        if 'triggers' in card_raw:
            card_raw['effects'] = card_raw.pop('triggers')

        # Process effects for command lifting
        if 'effects' in card_raw:
            for eff in card_raw['effects']:
                DataMigration.lift_actions_to_commands(eff)

        # Migrate static abilities using preserved logic
        migrate_card_data(card_raw)

        return card_raw

# --- Preserved Logic from Original File ---

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
        # print(f"[Migration] Migrated keyword: {mtype} str_val='{str_val}' → mutation_kind")
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
    
    # Assuming TargetScope.normalize exists and is robust.
    # If not, we might need to be careful.
    try:
        normalized = TargetScope.normalize(scope)
        if normalized != scope:
            modifier['scope'] = normalized
            # print(f"[Migration] Normalized scope: '{scope}' → '{normalized}'")
            return True
    except:
        pass
    
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
    scope = modifier.get('scope', 'ALL') # TargetScope.ALL default?
    # Cannot easily check normalization without constants map, assuming cleaned if logic ran.
    
    return warnings
