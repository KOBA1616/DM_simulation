# -*- coding: utf-8 -*-
import unittest
from dm_toolkit.action_to_command import map_action


class TestActionToCommandCoverage(unittest.TestCase):
    def test_all_known_action_types_map(self):
        types = [
            "MOVE_CARD", "DESTROY", "DISCARD", "MANA_CHARGE", "RETURN_TO_HAND",
            "SEND_TO_MANA", "SEND_TO_DECK_BOTTOM", "ADD_SHIELD", "SHIELD_BURN",
            "ADD_MANA", "SEARCH_DECK_BOTTOM", "DRAW_CARD", "TAP", "UNTAP",
            "COUNT_CARDS", "MEASURE_COUNT", "GET_GAME_STAT",
            "APPLY_MODIFIER", "COST_REDUCTION", "GRANT_KEYWORD",
            "MUTATE", "POWER_MOD", "MODIFY_POWER",
            "SELECT_OPTION", "SELECT_NUMBER", "SELECT_TARGET",
            "SEARCH_DECK", "SHUFFLE_DECK", "REVEAL_CARDS", "LOOK_AND_ADD", "MEKRAID", "REVOLUTION_CHANGE",
            "PLAY_FROM_ZONE", "FRIEND_BURST", "REGISTER_DELAYED_EFFECT", "CAST_SPELL",
            "RESET_INSTANCE", "SEND_SHIELD_TO_GRAVE", "PUT_CREATURE", "COST_REFERENCE",
            "ATTACK_PLAYER", "ATTACK_CREATURE", "BLOCK", "BREAK_SHIELD",
            "RESOLVE_BATTLE", "RESOLVE_EFFECT", "USE_SHIELD_TRIGGER", "RESOLVE_PLAY",
            "LOOK_TO_BUFFER", "SELECT_FROM_BUFFER", "PLAY_FROM_BUFFER", "MOVE_BUFFER_TO_ZONE", "SUMMON_TOKEN"
        ]

        for t in types:
            with self.subTest(action_type=t):
                act = {"type": t}
                # Provide some minimal fields for types that often expect values
                if t in ("MOVE_CARD", "PLAY_FROM_ZONE"):
                    act.update({"from_zone": "HAND", "to_zone": "BATTLE_ZONE"})
                if t == "DRAW_CARD":
                    act.update({"value1": 1})
                if t == "SELECT_OPTION":
                    act.update({"options": [{"type": "DRAW_CARD", "value1": 1}]})

                cmd = map_action(act)
                self.assertIsInstance(cmd, dict)
                self.assertIn('uid', cmd)
                # Either a valid mapped type exists or a legacy warning is produced
                self.assertTrue(isinstance(cmd.get('type'), str))
                self.assertTrue(cmd.get('type') != 'NONE' or cmd.get('legacy_warning') is True)


if __name__ == '__main__':
    unittest.main()
