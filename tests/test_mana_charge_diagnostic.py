# -*- coding: utf-8 -*-
"""
Diagnostic test for MANA_CHARGE command execution.
This test traces the entire command flow from generation to execution.
"""
import pytest
import dm_ai_module


def test_mana_charge_command_flow_diagnostic():
    """Trace the complete MANA_CHARGE command flow."""
    print("\n" + "="*80)
    print("DIAGNOSTIC: MANA_CHARGE Command Flow Test")
    print("="*80)
    
    # Setup
    game_instance = dm_ai_module.GameInstance(seed=42)
    card_db = dm_ai_module.JsonLoader.load_cards("data/cards.json")
    
    # Initialize game
    game_instance.state.active_player_id = 0
    game_instance.state.current_phase = dm_ai_module.Phase.MANA
    game_instance.state.turn_stats.mana_charged_by_player[0] = False
    
    # Add cards to hand
    from dm_toolkit.engine.compat import EngineCompat
    card_iid_1 = EngineCompat.add_card_to_zone(
        game_instance.state,
        player_id=0,
        card_id=1,
        zone="HAND",
        card_db=card_db
    )
    card_iid_2 = EngineCompat.add_card_to_zone(
        game_instance.state,
        player_id=0,
        card_id=2,
        zone="HAND",
        card_db=card_db
    )
    
    print(f"\n1. Initial State:")
    print(f"   - Player 0 hand size: {len(game_instance.state.players[0].hand)}")
    print(f"   - Player 0 mana size: {len(game_instance.state.players[0].mana_zone)}")
    print(f"   - Card instance IDs in hand: {[c.instance_id for c in game_instance.state.players[0].hand]}")
    print(f"   - Added card_iid_1: {card_iid_1}")
    print(f"   - Added card_iid_2: {card_iid_2}")
    
    # Generate commands using C++ command generator
    print(f"\n2. Generating commands via C++ generate_commands:")
    try:
        commands = dm_ai_module.generate_commands(game_instance.state, card_db)
        print(f"   - Generated {len(commands)} commands")
        
        mana_charge_cmds = [cmd for cmd in commands if cmd.get('type') == 'MANA_CHARGE']
        print(f"   - Found {len(mana_charge_cmds)} MANA_CHARGE commands")
        
        for i, cmd in enumerate(mana_charge_cmds):
            print(f"   - MANA_CHARGE command {i}: {cmd}")
            
    except Exception as e:
        print(f"   - ERROR generating commands: {e}")
        import traceback
        traceback.print_exc()
        mana_charge_cmds = []
    
    # Try Python command generation
    print(f"\n3. Generating commands via Python commands_v2:")
    try:
        from dm_toolkit import commands_v2
        py_commands = commands_v2.generate_legal_commands(game_instance.state, card_db)
        print(f"   - Generated {len(py_commands)} commands")
        
        py_mana_cmds = []
        for cmd in py_commands:
            if hasattr(cmd, 'to_dict'):
                cmd_dict = cmd.to_dict()
                if cmd_dict.get('type') == 'MANA_CHARGE':
                    py_mana_cmds.append(cmd_dict)
                    print(f"   - MANA_CHARGE command: {cmd_dict}")
        
        print(f"   - Found {len(py_mana_cmds)} MANA_CHARGE commands")
        
    except Exception as e:
        print(f"   - ERROR generating Python commands: {e}")
        import traceback
        traceback.print_exc()
        py_mana_cmds = []
    
    # Execute a MANA_CHARGE command if available
    if mana_charge_cmds:
        cmd_to_execute = mana_charge_cmds[0]
        print(f"\n4. Executing MANA_CHARGE command:")
        print(f"   - Command: {cmd_to_execute}")
        print(f"   - instance_id: {cmd_to_execute.get('instance_id', 'NOT FOUND')}")
        
        hand_before = len(game_instance.state.players[0].hand)
        mana_before = len(game_instance.state.players[0].mana_zone)
        
        try:
            game_instance.execute_command(cmd_to_execute)
            print(f"   - Command executed successfully")
        except Exception as e:
            print(f"   - ERROR executing command: {e}")
            import traceback
            traceback.print_exc()
        
        hand_after = len(game_instance.state.players[0].hand)
        mana_after = len(game_instance.state.players[0].mana_zone)
        
        print(f"\n5. Result:")
        print(f"   - Hand before: {hand_before}, after: {hand_after}")
        print(f"   - Mana before: {mana_before}, after: {mana_after}")
        print(f"   - Mana charged flag: {game_instance.state.turn_stats.mana_charged_by_player[0]}")
        
        if hand_after == hand_before - 1 and mana_after == mana_before + 1:
            print(f"   ✅ MANA_CHARGE SUCCESSFUL!")
        else:
            print(f"   ❌ MANA_CHARGE FAILED!")
            
    else:
        print(f"\n4. No MANA_CHARGE commands generated!")
        print(f"   ❌ TEST FAILED: Cannot execute if no commands are generated")
    
    # Check logs
    print(f"\n6. Checking logs:")
    import os
    if os.path.exists("logs/manacharge_trace.txt"):
        with open("logs/manacharge_trace.txt", "r") as f:
            log_content = f.read()
            print(f"   - manacharge_trace.txt content:")
            for line in log_content.split('\n')[-10:]:
                if line.strip():
                    print(f"     {line}")
    else:
        print(f"   - No manacharge_trace.txt found")
    
    if os.path.exists("logs/pipeline_trace.txt"):
        with open("logs/pipeline_trace.txt", "r") as f:
            log_content = f.read()
            print(f"   - pipeline_trace.txt content:")
            for line in log_content.split('\n')[-10:]:
                if line.strip():
                    print(f"     {line}")
    else:
        print(f"   - No pipeline_trace.txt found")
    
    print("\n" + "="*80)
    
    # Assert that we can generate and execute MANA_CHARGE
    assert len(mana_charge_cmds) > 0 or len(py_mana_cmds) > 0, "Should generate MANA_CHARGE commands"


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])
