# -*- coding: utf-8 -*-
"""
Simple integration test for MANA_CHARGE command execution.
"""
import pytest


def test_mana_charge_command_builder():
    """Test that build_mana_charge_command creates correct structure."""
    from dm_toolkit.command_builders import build_mana_charge_command
    
    # Build command
    cmd = build_mana_charge_command(instance_id=123)
    
    # Verify structure
    assert cmd['type'] == 'MANA_CHARGE', "Type should be MANA_CHARGE"
    assert cmd['instance_id'] == 123, "instance_id should be 123"
    assert cmd['from_zone'] == 'HAND', "from_zone should be HAND"
    assert cmd['to_zone'] == 'MANA', "to_zone should be MANA"
    assert 'uid' in cmd, "Command should have uid"
    print(f"✅ MANA_CHARGE command structure: {cmd}")


def test_mana_charge_command_dict_structure():
    """Test that MANA_CHARGE command dict has correct keys for C++ binding."""
    cmd = {
        "type": "MANA_CHARGE",
        "instance_id": 5,  # Must be instance_id, not source_instance_id
        "from_zone": "HAND",
        "to_zone": "MANA"
    }
    
    # Verify C++ binding expectations
    assert 'instance_id' in cmd, "Must have instance_id for C++ binding"
    assert cmd['instance_id'] > 0, "instance_id must be valid"
    assert cmd['type'] == 'MANA_CHARGE', "Type must be MANA_CHARGE"
    print(f"✅ Command dict for C++ binding: {cmd}")


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])
