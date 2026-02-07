#!/usr/bin/env python3
"""Test Phase 2: Player Mode Management Migration to C++

Verifies:
1. PlayerMode enum is accessible from Python
2. GameState.player_modes array works
3. GameState.is_human_player() helper works
4. GameSession.set_player_mode() updates C++ state
5. GameInstance.step() returns false for human players
"""

import sys
import os

# Ensure module can be imported
sys.path.insert(0, os.path.dirname(__file__))

try:
    import dm_ai_module
except ImportError as e:
    print(f"❌ Failed to import dm_ai_module: {e}")
    print("Build the module first:")
    print("  cmake --build build-msvc --config Release --target dm_ai_module")
    print("  Copy-Item -Force build-msvc\\Release\\dm_ai_module.*.pyd $env:VIRTUAL_ENV\\Lib\\site-packages\\")
    sys.exit(1)

def test_player_mode_enum():
    """Test PlayerMode enum accessibility."""
    print("=" * 60)
    print("TEST 1: PlayerMode enum")
    print("=" * 60)
    
    # Check enum exists and has correct values
    assert hasattr(dm_ai_module, 'PlayerMode'), "PlayerMode enum not found"
    assert hasattr(dm_ai_module.PlayerMode, 'AI'), "PlayerMode.AI not found"
    assert hasattr(dm_ai_module.PlayerMode, 'HUMAN'), "PlayerMode.HUMAN not found"
    
    print(f"✅ PlayerMode.AI = {dm_ai_module.PlayerMode.AI}")
    print(f"✅ PlayerMode.HUMAN = {dm_ai_module.PlayerMode.HUMAN}")
    print(f"✅ PlayerMode enum accessible from Python\n")

def test_gamestate_player_modes():
    """Test GameState.player_modes array."""
    print("=" * 60)
    print("TEST 2: GameState.player_modes array")
    print("=" * 60)
    
    # Create GameState
    card_db = dm_ai_module.create_card_database()
    gi = dm_ai_module.GameInstance(card_db)
    gs = gi.state
    
    # Check default values (both AI)
    print(f"Default player_modes[0]: {gs.player_modes[0]}")
    print(f"Default player_modes[1]: {gs.player_modes[1]}")
    assert gs.player_modes[0] == dm_ai_module.PlayerMode.AI, "Default P0 should be AI"
    assert gs.player_modes[1] == dm_ai_module.PlayerMode.AI, "Default P1 should be AI"
    print("✅ Default: both players are AI")
    
    # Modify player modes
    gs.player_modes[0] = dm_ai_module.PlayerMode.HUMAN
    print(f"After setting P0 to HUMAN: {gs.player_modes[0]}")
    assert gs.player_modes[0] == dm_ai_module.PlayerMode.HUMAN, "P0 should be HUMAN"
    print("✅ Can modify player_modes array\n")

def test_is_human_player():
    """Test GameState.is_human_player() helper."""
    print("=" * 60)
    print("TEST 3: GameState.is_human_player()")
    print("=" * 60)
    
    card_db = dm_ai_module.create_card_database()
    gi = dm_ai_module.GameInstance(card_db)
    gs = gi.state
    
    # Both AI initially
    assert not gs.is_human_player(0), "P0 should not be human initially"
    assert not gs.is_human_player(1), "P1 should not be human initially"
    print("✅ Default: is_human_player(0) = False")
    print("✅ Default: is_human_player(1) = False")
    
    # Set P0 to human
    gs.player_modes[0] = dm_ai_module.PlayerMode.HUMAN
    assert gs.is_human_player(0), "P0 should be human after setting"
    assert not gs.is_human_player(1), "P1 should still be AI"
    print("✅ After P0 → HUMAN: is_human_player(0) = True")
    print("✅ After P0 → HUMAN: is_human_player(1) = False\n")

def test_game_session_integration():
    """Test GameSession.set_player_mode() C++ integration."""
    print("=" * 60)
    print("TEST 4: GameSession.set_player_mode() integration")
    print("=" * 60)
    
    from dm_toolkit.gui.game_session import GameSession
    
    session = GameSession()
    session.initialize_game()
    
    # Check initial state
    assert not session.gs.is_human_player(0), "P0 should be AI initially"
    print("✅ Initial: P0 is AI")
    
    # Set P0 to Human
    session.set_player_mode(0, 'Human')
    assert session.gs.is_human_player(0), "P0 should be Human after set_player_mode"
    assert session.gs.player_modes[0] == dm_ai_module.PlayerMode.HUMAN
    print("✅ After set_player_mode(0, 'Human'): P0 is HUMAN")
    
    # Set P0 back to AI
    session.set_player_mode(0, 'AI')
    assert not session.gs.is_human_player(0), "P0 should be AI after reset"
    assert session.gs.player_modes[0] == dm_ai_module.PlayerMode.AI
    print("✅ After set_player_mode(0, 'AI'): P0 is AI\n")

def test_gameinstance_step_human_check():
    """Test GameInstance.step() returns false for human players."""
    print("=" * 60)
    print("TEST 5: GameInstance.step() human player check")
    print("=" * 60)
    
    card_db = dm_ai_module.create_card_database()
    gi = dm_ai_module.GameInstance(card_db)
    gs = gi.state
    
    # AI vs AI - step should progress
    print("Testing AI vs AI game...")
    result = gi.step()
    print(f"AI turn step() result: {result}")
    # Result depends on action availability, but should not fail
    print("✅ AI turn: step() executed without error")
    
    # Set active player to Human
    active_pid = gs.active_player
    gs.player_modes[active_pid] = dm_ai_module.PlayerMode.HUMAN
    print(f"\nSet P{active_pid} (active player) to HUMAN")
    
    # step() should return false immediately
    result = gi.step()
    assert result == False, "step() should return False for human player"
    print(f"✅ Human turn: step() returned False (expected)\n")

def main():
    """Run all Phase 2 tests."""
    print("\n" + "=" * 60)
    print("PHASE 2 TEST SUITE: Player Mode Management")
    print("=" * 60 + "\n")
    
    try:
        test_player_mode_enum()
        test_gamestate_player_modes()
        test_is_human_player()
        test_game_session_integration()
        test_gameinstance_step_human_check()
        
        print("=" * 60)
        print("✅ ALL PHASE 2 TESTS PASSED")
        print("=" * 60)
        print("\nPhase 2 successfully migrated player mode management to C++:")
        print("  ✓ PlayerMode enum (AI/HUMAN)")
        print("  ✓ GameState.player_modes array")
        print("  ✓ GameState.is_human_player() helper")
        print("  ✓ GameSession integration")
        print("  ✓ GameInstance human player check")
        print("\nReady for Phase 3: Event Notification System")
        
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ UNEXPECTED ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    main()
