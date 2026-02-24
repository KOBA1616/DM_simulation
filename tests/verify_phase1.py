import sys
import os
import unittest
try:
    import dm_ai_module
    HAS_NATIVE = True
except ImportError:
    HAS_NATIVE = False

from dm_toolkit.command_builders import (
    build_draw_command,
    build_mana_charge_command,
    build_destroy_command
)

class TestPhase1(unittest.TestCase):
    def setUp(self):
        if not HAS_NATIVE:
            self.skipTest("dm_ai_module not available")

    def test_native_builder_draw(self):
        cmd = build_draw_command(amount=2, native=True)
        self.assertIsInstance(cmd, dm_ai_module.CommandDef)

        d = cmd.to_dict()
        print(f"Draw Command Dict: {d}")

        # Check type (it might be Enum object or int depending on binding, usually Enum)
        self.assertEqual(d['type'], dm_ai_module.CommandType.DRAW_CARD)
        self.assertEqual(d['amount'], 2)
        self.assertEqual(d['from_zone'], "DECK")
        self.assertEqual(d['to_zone'], "HAND")

    def test_native_builder_mana(self):
        cmd = build_mana_charge_command(source_instance_id=10, native=True)
        d = cmd.to_dict()
        print(f"Mana Command Dict: {d}")

        self.assertEqual(d['type'], dm_ai_module.CommandType.MANA_CHARGE)
        self.assertEqual(d['instance_id'], 10)
        self.assertEqual(d['to_zone'], "MANA")

    def test_native_builder_destroy_with_filter(self):
        # build_destroy_command accepts target_filter as dict
        f_dict = {'races': ['Dragon'], 'min_cost': 5}
        cmd = build_destroy_command(target_filter=f_dict, native=True)
        d = cmd.to_dict()
        print(f"Destroy Command Dict: {d}")

        self.assertEqual(d['type'], dm_ai_module.CommandType.DESTROY)

        # Check filter
        tf = d['target_filter']
        self.assertIsInstance(tf, dict)
        self.assertEqual(tf['races'], ['Dragon'])
        self.assertEqual(tf['min_cost'], 5)

    def test_native_builder_filter_civilizations(self):
        # Test string conversion
        f_dict = {'civilizations': ['FIRE']}
        cmd = build_destroy_command(target_filter=f_dict, native=True)
        self.assertEqual(cmd.target_filter.civilizations[0], dm_ai_module.Civilization.FIRE)

        # Test enum input
        f_dict2 = {'civilizations': [dm_ai_module.Civilization.FIRE]}
        cmd2 = build_destroy_command(target_filter=f_dict2, native=True)
        self.assertEqual(cmd2.target_filter.civilizations[0], dm_ai_module.Civilization.FIRE)

if __name__ == '__main__':
    unittest.main()
