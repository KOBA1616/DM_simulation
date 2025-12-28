
import unittest
from typing import Any, Dict
from dm_toolkit.action_to_command import ActionToCommand

class TestActionToCommand(unittest.TestCase):

    def test_move_card_mappings(self):
        # 1. Destroy
        act_destroy = {
            "type": "DESTROY",
            "source_zone": "BATTLE_ZONE",
            "filter": {"zones": ["BATTLE_ZONE"]}
        }
        cmd = ActionToCommand.map_action(act_destroy)
        self.assertEqual(cmd['type'], "DESTROY")
        self.assertEqual(cmd['to_zone'], "GRAVEYARD")
        self.assertEqual(cmd['from_zone'], "BATTLE_ZONE")

        # 2. Mana Charge
        act_mana = {
            "type": "MANA_CHARGE",
            "source_zone": "HAND",
            "filter": {"zones": ["HAND"]}
        }
        cmd = ActionToCommand.map_action(act_mana)
        self.assertEqual(cmd['type'], "MANA_CHARGE")
        self.assertEqual(cmd['to_zone'], "MANA_ZONE")
        self.assertEqual(cmd['from_zone'], "HAND")

        # 3. Discard
        act_discard = {
            "type": "DISCARD",
            "filter": {"zones": ["HAND"]}
        }
        cmd = ActionToCommand.map_action(act_discard)
        self.assertEqual(cmd['type'], "DISCARD")
        self.assertEqual(cmd['to_zone'], "GRAVEYARD")
        self.assertEqual(cmd['from_zone'], "HAND")

    def test_move_card_generic_mappings(self):
        # MOVE_CARD to Graveyard -> DESTROY/DISCARD
        act_move_grave = {
            "type": "MOVE_CARD",
            "from_zone": "BATTLE_ZONE",
            "to_zone": "GRAVEYARD"
        }
        cmd = ActionToCommand.map_action(act_move_grave)
        self.assertEqual(cmd['type'], "DESTROY")
        self.assertEqual(cmd['from_zone'], "BATTLE_ZONE")

        act_move_discard = {
            "type": "MOVE_CARD",
            "from_zone": "HAND",
            "to_zone": "GRAVEYARD"
        }
        cmd2 = ActionToCommand.map_action(act_move_discard)
        self.assertEqual(cmd2['type'], "DISCARD")

    def test_draw_card(self):
        act = {
            "type": "DRAW_CARD",
            "value1": 2,
            "scope": "PLAYER_SELF"
        }
        cmd = ActionToCommand.map_action(act)
        self.assertEqual(cmd['type'], "DRAW_CARD")
        self.assertEqual(cmd['amount'], 2)
        self.assertEqual(cmd['target_group'], "PLAYER_SELF")

    def test_engine_actions(self):
        act_attack = {
            "type": "ATTACK_PLAYER",
            "target_player": "OPPONENT",
            "source_instance_id": 10
        }
        cmd = ActionToCommand.map_action(act_attack)
        self.assertEqual(cmd['type'], "ATTACK_PLAYER")
        self.assertEqual(cmd['instance_id'], 10)

    def test_filter_transfer(self):
        act = {
            "type": "TAP",
            "filter": {
                "zones": ["BATTLE_ZONE"],
                "is_tapped": False
            }
        }
        cmd = ActionToCommand.map_action(act)
        self.assertEqual(cmd['type'], "TAP")
        self.assertIn('target_filter', cmd)
        self.assertEqual(cmd['target_filter']['zones'], ["BATTLE_ZONE"])
        self.assertEqual(cmd['target_filter']['is_tapped'], False)

    def test_choice_option_mapping(self):
        act = {
            "type": "SELECT_OPTION",
            "value1": 1,
            "options": [
                [{"type": "DRAW_CARD", "value1": 1}],
                [{"type": "ADD_SHIELD"}]
            ]
        }
        cmd = ActionToCommand.map_action(act)
        self.assertEqual(cmd['type'], "CHOICE")
        self.assertEqual(cmd['amount'], 1)
        self.assertIn('if_true', cmd)
        self.assertIn('if_false', cmd)
        self.assertEqual(cmd['if_true'][0]['type'], "DRAW_CARD")
        # ADD_SHIELD maps to TRANSITION currently
        self.assertEqual(cmd['if_false'][0]['type'], "TRANSITION")
        self.assertEqual(cmd['if_false'][0]['to_zone'], "SHIELD_ZONE")

    def test_play_from_zone_explicit(self):
        act = {
            "type": "PLAY_FROM_ZONE",
            "source_zone": "MANA_ZONE",
            "value1": 5
        }
        cmd = ActionToCommand.map_action(act)
        self.assertEqual(cmd['type'], "PLAY_FROM_ZONE")
        self.assertEqual(cmd['from_zone'], "MANA_ZONE")
        self.assertEqual(cmd['to_zone'], "BATTLE_ZONE")
        self.assertEqual(cmd['max_cost'], 5)

if __name__ == '__main__':
    unittest.main()
