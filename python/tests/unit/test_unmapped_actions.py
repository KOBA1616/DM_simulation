
import unittest
import sys
import os

# Ensure project root is in path
sys.path.append(os.getcwd())

from dm_toolkit.action_mapper import ActionToCommandMapper

class TestUnmappedActions(unittest.TestCase):
    def test_high_priority_mappings(self):
        """Verify High Priority Engine actions map to correct Command types."""

        # BLOCK
        act_block = {"type": "BLOCK", "blocker_id": 10, "attacker_id": 20}
        cmd_block = ActionToCommandMapper.map_action(act_block)
        self.assertEqual(cmd_block["type"], "FLOW")
        self.assertEqual(cmd_block["flow_type"], "BLOCK")
        self.assertEqual(cmd_block["instance_id"], 10)
        self.assertEqual(cmd_block["target_instance"], 20)

        # BREAK_SHIELD
        act_break = {"type": "BREAK_SHIELD", "creature_id": 5, "value1": 2}
        cmd_break = ActionToCommandMapper.map_action(act_break)
        self.assertEqual(cmd_break["type"], "BREAK_SHIELD")
        self.assertEqual(cmd_break["amount"], 2)
        self.assertEqual(cmd_break["instance_id"], 5)

        # RESOLVE_BATTLE
        act_res_bat = {"type": "RESOLVE_BATTLE", "winner_id": 99}
        cmd_res_bat = ActionToCommandMapper.map_action(act_res_bat)
        self.assertEqual(cmd_res_bat["type"], "RESOLVE_BATTLE")
        self.assertEqual(cmd_res_bat["winner_instance"], 99)

        # RESOLVE_EFFECT
        act_res_eff = {"type": "RESOLVE_EFFECT", "effect_id": 1234}
        cmd_res_eff = ActionToCommandMapper.map_action(act_res_eff)
        self.assertEqual(cmd_res_eff["type"], "RESOLVE_EFFECT")
        self.assertEqual(cmd_res_eff["effect_id"], 1234)

        # USE_SHIELD_TRIGGER
        act_st = {"type": "USE_SHIELD_TRIGGER", "card_id": 7}
        cmd_st = ActionToCommandMapper.map_action(act_st)
        self.assertEqual(cmd_st["type"], "USE_SHIELD_TRIGGER")
        self.assertEqual(cmd_st["instance_id"], 7)

        # RESOLVE_PLAY
        act_rp = {"type": "RESOLVE_PLAY", "card_id": 7}
        cmd_rp = ActionToCommandMapper.map_action(act_rp)
        self.assertEqual(cmd_rp["type"], "RESOLVE_PLAY")
        self.assertEqual(cmd_rp["instance_id"], 7)

    def test_medium_priority_mappings(self):
        """Verify Medium Priority Effect actions."""

        # DESTROY
        act_des = {"type": "DESTROY", "scope": "OPPONENT_BATTLE_ZONE"}
        cmd_des = ActionToCommandMapper.map_action(act_des)
        self.assertEqual(cmd_des["type"], "DESTROY")
        self.assertEqual(cmd_des["target_group"], "OPPONENT_BATTLE_ZONE")

        # MEKRAID
        act_mek = {"type": "MEKRAID", "value1": 5, "value2": 3}
        cmd_mek = ActionToCommandMapper.map_action(act_mek)
        self.assertEqual(cmd_mek["type"], "MEKRAID")
        self.assertEqual(cmd_mek["max_cost"], 5)
        self.assertEqual(cmd_mek["look_count"], 3)
        self.assertEqual(cmd_mek["play_for_free"], True)

    def test_legacy_warning(self):
        """Verify unmapped action produces legacy warning."""
        act_unknown = {"type": "UNKNOWN_ACTION", "foo": "bar"}
        cmd_unknown = ActionToCommandMapper.map_action(act_unknown)
        self.assertEqual(cmd_unknown["type"], "NONE")
        self.assertTrue(cmd_unknown["legacy_warning"])
        self.assertEqual(cmd_unknown["legacy_original_type"], "UNKNOWN_ACTION")

if __name__ == "__main__":
    unittest.main()
