# -*- coding: utf-8 -*-
"""
Verification script for Unified Execution Consistency
Verifies that action_to_command and unified_execution enforce:
1. Bidirectional field normalization (legacy <-> canonical)
2. Zone name normalization
3. Legacy type preservation
4. Consistent behavior across various input formats
"""
import pytest
from dm_toolkit.unified_execution import ensure_executable_command

def test_numeric_normalization_legacy_to_canonical():
    """Verify value1 -> amount normalization."""
    action = {"type": "DRAW_CARD", "value1": 3}
    cmd = ensure_executable_command(action)
    assert cmd['amount'] == 3
    assert cmd['value1'] == 3
    assert cmd['type'] == "DRAW_CARD"

def test_numeric_normalization_canonical_to_legacy():
    """Verify amount -> value1 normalization."""
    action = {"type": "DRAW_CARD", "amount": 3}
    cmd = ensure_executable_command(action)
    assert cmd['amount'] == 3
    assert cmd['value1'] == 3

def test_numeric_conflict_resolution():
    """Verify that canonical amount takes precedence over value1 if different."""
    # Note: map_action usually prioritizes 'amount' if present in input for standard moves.
    # However, for specific moves, logic might vary.
    # DRAW_CARD uses _transfer_common_move_fields which checks amount first.
    action = {"type": "DRAW_CARD", "amount": 5, "value1": 2}
    cmd = ensure_executable_command(action)
    assert cmd['amount'] == 5
    assert cmd['value1'] == 5  # normalized to match amount

def test_string_normalization_legacy_to_canonical():
    """Verify str_val -> str_param normalization."""
    action = {"type": "MUTATE", "str_val": "TEST_KEYWORD", "target_group": "NONE"}
    cmd = ensure_executable_command(action)
    assert cmd['str_param'] == "TEST_KEYWORD"
    assert cmd['str_val'] == "TEST_KEYWORD"

def test_string_normalization_canonical_to_legacy():
    """Verify str_param -> str_val normalization."""
    action = {"type": "MUTATE", "str_param": "TEST_KEYWORD", "target_group": "NONE"}
    cmd = ensure_executable_command(action)
    assert cmd['str_param'] == "TEST_KEYWORD"
    assert cmd['str_val'] == "TEST_KEYWORD"

def test_zone_normalization():
    """Verify MANA_ZONE -> MANA normalization, etc."""
    action = {
        "type": "MOVE_CARD",
        "from_zone": "MANA_ZONE",
        "to_zone": "HAND",
        "source_instance_id": 1
    }
    cmd = ensure_executable_command(action)
    # Generic MOVE_CARD maps to RETURN_TO_HAND if to=HAND and from=MANA
    # But unified logic maps MANA_ZONE -> MANA first.
    # _handle_move_card maps to RETURN_TO_HAND if dest=HAND and src in [MANA, ...]

    assert cmd['from_zone'] == "MANA"
    assert cmd['to_zone'] == "HAND"
    # Should be mapped to specific type
    assert cmd['type'] == "RETURN_TO_HAND"

def test_shield_zone_normalization():
    """Verify SHIELD_ZONE -> SHIELD."""
    action = {
        "type": "MOVE_CARD",
        "from_zone": "SHIELD_ZONE",
        "to_zone": "HAND",
        "source_instance_id": 1
    }
    cmd = ensure_executable_command(action)
    assert cmd['from_zone'] == "SHIELD"
    assert cmd['type'] == "RETURN_TO_HAND"

def test_legacy_warning_for_unknown_type():
    """Verify that unknown types trigger a legacy warning."""
    action = {"type": "UNKNOWN_SUPER_ACTION", "value1": 1}
    cmd = ensure_executable_command(action)
    assert cmd['type'] == "NONE"
    assert cmd['legacy_warning'] is True
    assert cmd['legacy_original_type'] == "UNKNOWN_SUPER_ACTION"

def test_draw_card_preservation():
    """Verify DRAW_CARD type is preserved."""
    action = {"type": "DRAW_CARD", "value1": 1}
    cmd = ensure_executable_command(action)
    assert cmd['type'] == "DRAW_CARD"
    assert cmd['amount'] == 1

def test_shield_burn_mapping():
    """Verify SHIELD_BURN maps correctly and preserves legacy info."""
    action = {"type": "SHIELD_BURN", "value1": 1}
    cmd = ensure_executable_command(action)
    assert cmd['type'] == "SHIELD_BURN"
    assert cmd['amount'] == 1
    # Check alias info
    assert "SHIELD_BURN" in cmd.get('aliases', []) or cmd.get('legacy_original_type') == "SHIELD_BURN"

def test_search_deck_mapping():
    """Verify SEARCH_DECK maps to SEARCH_DECK with unified_type."""
    action = {"type": "SEARCH_DECK", "value1": 1, "filter": {"zones": ["DECK"]}}
    cmd = ensure_executable_command(action)
    assert cmd['type'] == "SEARCH_DECK"
    assert cmd['unified_type'] == "SEARCH"
    assert cmd['search_type'] == "SEARCH_DECK"
    assert cmd['amount'] == 1

def test_replace_card_move_fallback():
    """Verify REPLACE_CARD_MOVE creation."""
    action = {
        "type": "REPLACE_CARD_MOVE",
        "source_zone": "BATTLE",
        "destination_zone": "DECK_BOTTOM",
        "source_instance_id": 10
    }
    cmd = ensure_executable_command(action)
    assert cmd['type'] == "REPLACE_CARD_MOVE"
    # Unified mapping logic sets original_to_zone if provided, but here we don't have original intent other than explicit params?
    # Actually _handle_replace_card_move uses dest from input.
    assert cmd['to_zone'] == "DECK_BOTTOM"
    assert cmd['current_zone'] == "BATTLE"
    assert cmd['instance_id'] == 10

def test_input_object_support():
    """Verify ensure_executable_command works with objects having __dict__."""
    class ActionObj:
        def __init__(self):
            self.type = "DRAW_CARD"
            self.value1 = 2

    obj = ActionObj()
    cmd = ensure_executable_command(obj)
    assert cmd['type'] == "DRAW_CARD"
    assert cmd['amount'] == 2
    assert cmd['value1'] == 2
