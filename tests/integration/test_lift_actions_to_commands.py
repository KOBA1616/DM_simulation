# -*- coding: utf-8 -*-
import unittest
from PyQt6.QtGui import QStandardItemModel
from dm_toolkit.gui.editor.data_manager import CardDataManager


class TestLiftActionsToCommands(unittest.TestCase):
    def setUp(self):
        self.model = QStandardItemModel()
        self.manager = CardDataManager(self.model)

    def test_lift_actions_removes_actions_and_adds_commands(self):
        # Prepare an effect with a representative set of legacy actions
        actions = [
            {"type": "MOVE_CARD", "from_zone": "HAND", "to_zone": "MANA_ZONE", "value1": 1},
            {"type": "DESTROY", "source_zone": "BATTLE_ZONE"},
            {"type": "DRAW_CARD", "value1": 2},
            {"type": "TAP", "filter": {"zones": ["BATTLE_ZONE"]}},
            {"type": "COUNT_CARDS", "filter": {"zone": "GRAVEYARD"}},
            {"type": "APPLY_MODIFIER", "str_val": "POWER_MOD", "value1": 5},
            {"type": "SELECT_OPTION", "options": [{"type": "DRAW_CARD", "value1": 1}]},
            {"type": "SEARCH_DECK", "value1": 3},
            {"type": "PLAY_FROM_ZONE", "from_zone": "HAND", "to_zone": "BATTLE_ZONE"},
            {"type": "RESET_INSTANCE"},
            {"type": "ATTACK_PLAYER", "source_instance_id": 10, "target_player": 1},
        ]

        effect = {"actions": actions.copy()}

        # Call internalizer to lift actions to commands
        self.manager._lift_actions_to_commands(effect)

        # After lifting, actions should be removed (unless legacy save forced)
        self.assertIn('commands', effect)
        self.assertNotIn('actions', effect)

        cmds = effect['commands']
        # There should be at least as many commands as actions
        self.assertGreaterEqual(len(cmds), len(actions))

        # Each command should be a dict with a 'type' and 'uid'
        for c in cmds:
            self.assertIsInstance(c, dict)
            self.assertIn('type', c)
            self.assertIn('uid', c)


if __name__ == '__main__':
    unittest.main()
