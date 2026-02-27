import unittest
import dm_ai_module
from dm_toolkit.command_builders import (
    build_draw_command,
    build_transition_command,
    build_mana_charge_command,
    build_destroy_command,
    _build_native_command
)

class TestNativeCommandBuilders(unittest.TestCase):
    def test_draw_command_native(self):
        cmd = build_draw_command(amount=2, owner_id=1, native=True)
        self.assertIsInstance(cmd, dm_ai_module.CommandDef)

        d = cmd.to_dict()
        self.assertEqual(d['type'], dm_ai_module.CommandType.DRAW_CARD)
        self.assertEqual(d['amount'], 2)
        self.assertEqual(d['owner_id'], 1)
        self.assertEqual(d['from_zone'], "DECK")
        self.assertEqual(d['to_zone'], "HAND")

    def test_transition_command_native(self):
        cmd = build_transition_command(
            from_zone="BATTLE",
            to_zone="GRAVEYARD",
            source_instance_id=100,
            native=True
        )
        self.assertIsInstance(cmd, dm_ai_module.CommandDef)

        d = cmd.to_dict()
        self.assertEqual(d['type'], dm_ai_module.CommandType.TRANSITION)
        self.assertEqual(d['instance_id'], 100)
        self.assertEqual(d['from_zone'], "BATTLE")
        self.assertEqual(d['to_zone'], "GRAVEYARD")

    def test_complex_command_recursive(self):
        # Test if_true/if_false recursion and conditions
        condition = {
            "type": "some_condition",
            "value": 5,
            "filter": {"zones": ["BATTLE_ZONE"]}
        }

        # Using builder instead of dict literal
        true_cmd = build_draw_command(amount=1, native=True)

        cmd = _build_native_command(
            "IF",
            condition=condition,
            if_true=[true_cmd],
            target_group="PLAYER_SELF"
        )

        self.assertIsInstance(cmd, dm_ai_module.CommandDef)
        d = cmd.to_dict()

        self.assertEqual(d['type'], dm_ai_module.CommandType.IF)
        self.assertEqual(d['target_group'], dm_ai_module.TargetScope.PLAYER_SELF)

        # Check Condition
        self.assertIsNotNone(d['condition'])
        self.assertEqual(d['condition']['type'], "some_condition")
        self.assertEqual(d['condition']['value'], 5)
        self.assertEqual(d['condition']['filter']['zones'], ["BATTLE_ZONE"])

        # Check recursive command
        self.assertEqual(len(d['if_true']), 1)
        self.assertEqual(d['if_true'][0]['type'], dm_ai_module.CommandType.DRAW_CARD)
        self.assertEqual(d['if_true'][0]['amount'], 1)

    def test_filter_enum_conversion(self):
        # Test civilization enum conversion in filters
        cmd = build_destroy_command(
            target_filter={
                "civilizations": ["FIRE", "WATER"],
                "types": ["CREATURE"]
            },
            native=True
        )

        d = cmd.to_dict()
        tf = d['target_filter']

        self.assertTrue(len(tf['civilizations']) == 2)
        # Check if they match FIRE/WATER constants
        self.assertIn(dm_ai_module.Civilization.FIRE, tf['civilizations'])
        self.assertIn(dm_ai_module.Civilization.WATER, tf['civilizations'])

if __name__ == '__main__':
    unittest.main()
