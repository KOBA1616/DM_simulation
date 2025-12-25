# -*- coding: utf-8 -*-
import sys
import os
import unittest

# Ensure dm_toolkit is in path
sys.path.append(os.getcwd())

from dm_toolkit.gui.editor.action_converter import ActionConverter

class TestActionConverter(unittest.TestCase):
    def test_move_card_destroy(self):
        action = {
            "type": "DESTROY",
            "scope": "PLAYER_OPPONENT",
            "filter": {"count": 1, "card_type": "CREATURE"}
        }
        cmd = ActionConverter.convert(action)
        self.assertEqual(cmd['type'], "DESTROY")
        self.assertEqual(cmd['target_group'], "PLAYER_OPPONENT")
        self.assertEqual(cmd['amount'], 1)
        self.assertEqual(cmd['target_filter']['card_type'], "CREATURE")

    def test_move_card_generic_discard(self):
        action = {
            "type": "MOVE_CARD",
            "destination_zone": "GRAVEYARD",
            "source_zone": "HAND",
            "scope": "PLAYER_SELF",
            "filter": {"count": 2}
        }
        cmd = ActionConverter.convert(action)
        self.assertEqual(cmd['type'], "DISCARD")
        self.assertEqual(cmd['amount'], 2)

    def test_count_cards(self):
        action = {
            "type": "COUNT_CARDS",
            "scope": "PLAYER_SELF",
            "filter": {"zone": "GRAVEYARD"},
            "output_value_key": "grave_count"
        }
        cmd = ActionConverter.convert(action)
        self.assertEqual(cmd['type'], "QUERY")
        self.assertEqual(cmd['str_param'], "CARDS_MATCHING_FILTER")
        self.assertEqual(cmd['output_value_key'], "grave_count")

    def test_cost_reduction(self):
        action = {
            "type": "COST_REDUCTION",
            "value1": 2,
            "filter": {"civilization": "FIRE"}
        }
        cmd = ActionConverter.convert(action)
        self.assertEqual(cmd['type'], "MUTATE")
        self.assertEqual(cmd['mutation_kind'], "COST")
        self.assertEqual(cmd['amount'], 2)

    def test_grant_keyword(self):
        action = {
            "type": "GRANT_KEYWORD",
            "str_val": "speed_attacker",
            "scope": "TARGET_SELECT"
        }
        cmd = ActionConverter.convert(action)
        self.assertEqual(cmd['type'], "ADD_KEYWORD")
        self.assertEqual(cmd['mutation_kind'], "speed_attacker")

if __name__ == '__main__':
    unittest.main()
