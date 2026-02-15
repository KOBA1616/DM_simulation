import unittest
import sys
import os
import enum

# Ensure project root is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from dm_toolkit.engine.compat import EngineCompat

# Try to import dm_ai_module and Phase. If not available, use fallbacks.
try:
    from dm_toolkit import dm_ai_module
    from dm_toolkit.dm_ai_module import GameState, Phase, PlayerStub
except ImportError:
    # Minimal stub if not found
    class Phase(enum.IntEnum):
        START = 0
        DRAW = 1
        MANA = 2
        MAIN = 3
        ATTACK = 4
        END = 5

    class PlayerStub:
        def __init__(self):
            self.hand = []
            self.deck = []
            self.battle_zone = []
            self.mana_zone = []
            self.shield_zone = []
            self.graveyard = []

    class GameState:
        def __init__(self):
            self.players = [PlayerStub(), PlayerStub()]
            self.active_player_id = 0
            self.current_phase = 0
            self.turn_number = 1
            self.pending_query = None
            self.pending_effects = []
            self.command_history = []
            self.effect_buffer = []

class TestEngineCompatGetters(unittest.TestCase):
    def setUp(self):
        self.state = GameState()

    def test_get_current_phase(self):
        # 1. Test with Enum
        self.state.current_phase = Phase.MANA
        val = EngineCompat.get_current_phase(self.state)
        print(f"DEBUG: get_current_phase(Enum) -> {val}")
        self.assertEqual(val, Phase.MANA)

        # 2. Test with Int
        self.state.current_phase = 2
        val = EngineCompat.get_current_phase(self.state)
        print(f"DEBUG: get_current_phase(Int) -> {val}")
        # EngineCompat tries to convert int to Phase enum if available
        self.assertEqual(val, Phase.MANA)

        # 3. Test with String
        self.state.current_phase = "MANA"
        val = EngineCompat.get_current_phase(self.state)
        print(f"DEBUG: get_current_phase(Str) -> {val}")
        # EngineCompat logic explicitly preserves strings (see comments in compat.py)
        # So it returns "MANA", not Phase.MANA
        self.assertEqual(val, "MANA")

        # 4. Test with Alias 'phase'
        del self.state.current_phase
        self.state.phase = Phase.ATTACK
        val = EngineCompat.get_current_phase(self.state)
        print(f"DEBUG: get_current_phase(Alias) -> {val}")
        self.assertEqual(val, Phase.ATTACK)

        print("OK: get_current_phase")

    def test_get_turn_number(self):
        self.state.turn_number = 42
        val = EngineCompat.get_turn_number(self.state)
        print(f"DEBUG: get_turn_number -> {val}")
        self.assertEqual(val, 42)
        print("OK: get_turn_number")

    def test_get_active_player_id(self):
        # 1. Standard
        self.state.active_player_id = 1
        val = EngineCompat.get_active_player_id(self.state)
        print(f"DEBUG: get_active_player_id -> {val}")
        self.assertEqual(val, 1)

        # 2. Alias
        del self.state.active_player_id
        self.state.active_player = 0
        val = EngineCompat.get_active_player_id(self.state)
        print(f"DEBUG: get_active_player_id(Alias) -> {val}")
        self.assertEqual(val, 0)

        print("OK: get_active_player_id")

    def test_get_player(self):
        # 1. Standard list access
        p0 = self.state.players[0]
        val = EngineCompat.get_player(self.state, 0)
        print(f"DEBUG: get_player(0) -> {val}")
        self.assertIs(val, p0)

        # 2. Legacy fallback (simulate no players list, but player0/player1 attrs)
        del self.state.players
        p0_stub = PlayerStub()
        p1_stub = PlayerStub()
        self.state.player0 = p0_stub
        self.state.player1 = p1_stub

        val0 = EngineCompat.get_player(self.state, 0)
        print(f"DEBUG: get_player(0) Legacy -> {val0}")
        self.assertIs(val0, p0_stub)

        val1 = EngineCompat.get_player(self.state, 1)
        print(f"DEBUG: get_player(1) Legacy -> {val1}")
        self.assertIs(val1, p1_stub)

        # 3. Invalid index
        val_invalid = EngineCompat.get_player(self.state, 99)
        self.assertIsNone(val_invalid)

        print("OK: get_player")

    def test_get_pending_query(self):
        self.state.pending_query = "Choose 1 card"
        val = EngineCompat.get_pending_query(self.state)
        print(f"DEBUG: get_pending_query -> {val}")
        self.assertEqual(val, "Choose 1 card")
        print("OK: get_pending_query")

    def test_get_pending_effects_info(self):
        class MockCmd:
            type = "DRAW_CARD"
            source_instance_id = 101
            target_player = 0

        cmd = MockCmd()
        if hasattr(self.state, 'pending_effects'):
            self.state.pending_effects.append(cmd)

        # This calls EngineCompat which might use dm_ai_module or fallback
        info = EngineCompat.get_pending_effects_info(self.state)
        print(f"DEBUG: get_pending_effects_info -> {info}")

        # Depending on whether dm_ai_module stub or native is loaded, logic might slightly vary
        # But fallback logic expects (type, sid, pid, cmd)
        self.assertTrue(len(info) > 0)
        first = info[0]
        self.assertEqual(first[0], "DRAW_CARD")
        self.assertEqual(first[1], 101)
        self.assertEqual(first[2], 0)
        # 4th element is the command object itself

        print("OK: get_pending_effects_info")

    def test_get_command_history(self):
        self.state.command_history = ["CmdA", "CmdB"]
        val = EngineCompat.get_command_history(self.state)
        print(f"DEBUG: get_command_history -> {val}")
        self.assertEqual(val, ["CmdA", "CmdB"])
        print("OK: get_command_history")

    def test_dump_state_debug(self):
        # Setup some state
        self.state.active_player_id = 1
        self.state.current_phase = Phase.MAIN

        # Add some cards to verify counts
        if hasattr(self.state, 'players') and self.state.players:
            p1 = self.state.players[1]
            p1.hand.append("Card1")
            p1.mana_zone.append("Mana1")
        elif hasattr(self.state, 'player1'): # Legacy mode from previous test cleanup?
            # Revert to standard for this test
            self.state.players = [PlayerStub(), PlayerStub()]
            p1 = self.state.players[1]
            p1.hand.append("Card1")

        dump = EngineCompat.dump_state_debug(self.state)
        print(f"DEBUG: dump_state_debug -> {dump}")

        self.assertIn('native_present', dump)
        self.assertIn('active_player_id', dump)
        self.assertIn('current_phase', dump)
        self.assertIn('players', dump)

        self.assertEqual(dump['active_player_id'], 1)

        # Phase might be stringified.
        # Since Phase is IntEnum, str(Phase.MAIN) might be '3' or 'Phase.MAIN' depending on python version/impl
        # Phase.MAIN is 3.
        phase_str = str(dump['current_phase'])
        print(f"DEBUG: dump['current_phase'] -> {phase_str}")
        self.assertTrue('MAIN' in phase_str or '3' in phase_str)

        players_dump = dump['players']
        self.assertTrue(len(players_dump) >= 2)
        p1_dump = players_dump[1]
        self.assertTrue(p1_dump['hand_count'] >= 1)

        print("OK: dump_state_debug")

if __name__ == '__main__':
    unittest.main()
