
import sys
import os
import unittest

# Add path for dm_toolkit
sys.path.append(os.getcwd())

from dm_toolkit.unified_execution import ensure_executable_command
from dm_toolkit.action_to_command import map_action

class TestActionMigrationCoverage(unittest.TestCase):
    def test_legacy_primitives(self):
        """Verify that all EffectPrimitives map to a valid command."""
        primitives = [
            "DRAW_CARD", "ADD_MANA", "DESTROY", "RETURN_TO_HAND", "SEND_TO_MANA", "TAP", "UNTAP",
            "MODIFY_POWER", "BREAK_SHIELD", "LOOK_AND_ADD", "SUMMON_TOKEN", "SEARCH_DECK_BOTTOM",
            "MEKRAID", "DISCARD", "PLAY_FROM_ZONE", "COST_REFERENCE", "LOOK_TO_BUFFER",
            "SELECT_FROM_BUFFER", "PLAY_FROM_BUFFER", "MOVE_BUFFER_TO_ZONE", "REVOLUTION_CHANGE",
            "COUNT_CARDS", "GET_GAME_STAT", "APPLY_MODIFIER", "REVEAL_CARDS", "REGISTER_DELAYED_EFFECT",
            "RESET_INSTANCE", "SEARCH_DECK", "SHUFFLE_DECK", "ADD_SHIELD", "SEND_SHIELD_TO_GRAVE",
            "SEND_TO_DECK_BOTTOM", "MOVE_TO_UNDER_CARD", "SELECT_NUMBER", "FRIEND_BURST",
            "GRANT_KEYWORD", "MOVE_CARD", "CAST_SPELL", "PUT_CREATURE", "SELECT_OPTION",
            "RESOLVE_BATTLE"
        ]

        for p in primitives:
            with self.subTest(primitive=p):
                act = {"type": p, "value1": 10, "str_val": "TEST"}
                cmd = ensure_executable_command(act)
                self.assertNotEqual(cmd.get('type'), 'NONE', f"Primitive {p} mapped to NONE")
                self.assertFalse(cmd.get('legacy_warning'), f"Primitive {p} generated legacy warning: {cmd.get('str_param')}")

    def test_unified_execution_path(self):
        """Verify unified execution path handles dicts."""
        act = {"type": "DRAW_CARD", "value1": 1}
        cmd = ensure_executable_command(act)
        self.assertEqual(cmd['type'], 'DRAW_CARD')

    def test_missing_mapped_fields(self):
        """Verify specific fields are preserved for newly added mappings."""

        # RESET_INSTANCE
        cmd = ensure_executable_command({"type": "RESET_INSTANCE", "target_choice": "SELF"})
        self.assertEqual(cmd['type'], 'MUTATE')
        self.assertEqual(cmd['mutation_kind'], 'RESET_INSTANCE')

        # MOVE_TO_UNDER_CARD
        cmd = ensure_executable_command({"type": "MOVE_TO_UNDER_CARD", "from_zone": "HAND"})
        self.assertEqual(cmd['type'], 'TRANSITION')
        self.assertEqual(cmd['to_zone'], 'UNDER_CARD')

if __name__ == '__main__':
    unittest.main()
