# -*- coding: utf-8 -*-
import unittest
from dm_toolkit.action_to_command import map_action

class TestActionToCommand(unittest.TestCase):

    def test_invalid_input(self):
        cmd = map_action(None)
        self.assertTrue(cmd.get('legacy_warning'))
        self.assertEqual(cmd['str_param'], "Invalid action shape")

    def test_basic_move_to_mana(self):
        act = {
            "type": "MOVE_CARD",
            "from_zone": "HAND",
            "to_zone": "MANA_ZONE",
            "value1": 1
        }
        cmd = map_action(act)
        self.assertEqual(cmd['type'], "TRANSITION")
        self.assertEqual(cmd['to_zone'], "MANA")
        self.assertEqual(cmd['from_zone'], "HAND")
        self.assertEqual(cmd['amount'], 1)

    def test_destroy_card(self):
        act = {
            "type": "DESTROY",
            "source_zone": "BATTLE_ZONE"
        }
        cmd = map_action(act)
        self.assertEqual(cmd['type'], "DESTROY")
        self.assertEqual(cmd['to_zone'], "GRAVEYARD")
        self.assertEqual(cmd['from_zone'], "BATTLE")

    def test_draw_card(self):
        act = {
            "type": "DRAW_CARD",
            "value1": 2
        }
        cmd = map_action(act)
        self.assertEqual(cmd['type'], "DRAW_CARD")
        self.assertEqual(cmd['from_zone'], "DECK")
        self.assertEqual(cmd['to_zone'], "HAND")
        self.assertEqual(cmd['amount'], 2)

    def test_tap(self):
        act = {
            "type": "TAP",
            "filter": {"zones": ["BATTLE_ZONE"]}
        }
        cmd = map_action(act)
        self.assertEqual(cmd['type'], "TAP")
        self.assertEqual(cmd['target_filter']['zones'], ["BATTLE_ZONE"])

    def test_modifiers(self):
        act = {
            "type": "APPLY_MODIFIER",
            "str_val": "POWER_MOD",
            "value1": 1000
        }
        cmd = map_action(act)
        self.assertEqual(cmd['type'], "MUTATE")
        self.assertEqual(cmd['str_param'], "POWER_MOD")
        # value1 is not automatically mapped to amount for generic MUTATE in the mapper logic unless specific cases
        # Wait, the mapper logic says:
        # else: cmd['type']="MUTATE"; cmd['str_param']=val; _transfer_targeting(act, cmd)
        # _transfer_targeting calls _transfer_common_move_fields only if explicitly called? No.
        # _transfer_targeting does NOT map amount.
        # But _finalize_command maps 'value1' to 'amount' if missing.
        self.assertEqual(cmd['amount'], 1000)

    def test_nested_options(self):
        act = {
            "type": "SELECT_OPTION",
            "options": [
                {"type": "DRAW_CARD", "value1": 1},
                {"type": "MANA_CHARGE", "value1": 1}
            ]
        }
        cmd = map_action(act)
        self.assertEqual(cmd['type'], "CHOICE")
        self.assertTrue('options' in cmd)
        self.assertEqual(len(cmd['options']), 2)

        opt1 = cmd['options'][0][0] # options is list of lists of commands
        self.assertEqual(opt1['type'], "DRAW_CARD")
        self.assertEqual(opt1['to_zone'], "HAND")

        opt2 = cmd['options'][1][0]
        self.assertEqual(opt2['type'], "MANA_CHARGE")
        self.assertEqual(opt2['to_zone'], "MANA")

    def test_attack_player(self):
        act = {
            "type": "ATTACK_PLAYER",
            "source_instance_id": 100,
            "target_player": 1
        }
        cmd = map_action(act)
        self.assertEqual(cmd['type'], "ATTACK_PLAYER")
        self.assertEqual(cmd['instance_id'], 100)
        self.assertEqual(cmd['target_player'], 1)

    def test_legacy_keyword_fallback(self):
        act = {
            "type": "NONE",
            "str_val": "SPEED_ATTACKER"
        }
        cmd = map_action(act)
        self.assertEqual(cmd['type'], "ADD_KEYWORD")
        self.assertEqual(cmd['mutation_kind'], "SPEED_ATTACKER")

    def test_strict_type_validation_with_mock(self):
        # Mock dm_ai_module.CommandType to test validation logic
        import dm_toolkit.action_to_command as module_under_test

        # Create a mock Enum class
        class MockCommandType:
            VALID_TYPE = 1
            DRAW_CARD = 2

        # Capture original
        original_command_type = module_under_test._CommandType

        # Inject Mock
        module_under_test.set_command_type_enum(MockCommandType)

        try:
            # Case 1: Valid type
            act_valid = {"type": "DRAW_CARD", "value1": 1}
            cmd_valid = map_action(act_valid)
            self.assertEqual(cmd_valid['type'], "DRAW_CARD")
            self.assertFalse(cmd_valid.get('legacy_warning'))

            # Case 2: Invalid type (simulated)
            act_invalid = {"type": "MOVE_CARD", "from_zone": "HAND", "to_zone": "MANA_ZONE"}
            # map_action converts MOVE_CARD to TRANSITION.
            # TRANSITION is NOT in MockCommandType.
            cmd_invalid = map_action(act_invalid)

            self.assertEqual(cmd_invalid['type'], "NONE")
            self.assertTrue(cmd_invalid['legacy_warning'])
            # add_aliases_to_command might set legacy_original_type to original action type (MOVE_CARD)
            # which is preferred over the intermediate mapped type (TRANSITION).
            self.assertEqual(cmd_invalid['legacy_original_type'], "MOVE_CARD")
            self.assertIn("Invalid CommandType: TRANSITION", cmd_invalid['str_param'])

        finally:
            # Restore original
            module_under_test.set_command_type_enum(original_command_type)

if __name__ == '__main__':
    unittest.main()
