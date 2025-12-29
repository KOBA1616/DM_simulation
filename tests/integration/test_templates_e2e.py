# -*- coding: utf-8 -*-
import os
import json
import tempfile
import unittest
from dm_toolkit.gui.editor.data_manager import CardDataManager
from PyQt6.QtGui import QStandardItemModel


class TestEditorTemplatesE2E(unittest.TestCase):
    def test_templates_load_and_lift_actions(self):
        # Create a temporary editor_templates.json with an effect that contains actions
        tmp = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json')
        try:
            sample = {
                "actions": [
                    {
                        "uid": "tmpl1",
                        "actions": [
                            {"type": "MOVE_CARD", "from_zone": "HAND", "to_zone": "MANA_ZONE", "value1": 1},
                            {"type": "DRAW_CARD", "value1": 2}
                        ]
                    }
                ],
                "commands": []
            }
            json.dump(sample, tmp)
            tmp.close()

            # Point the CardDataManager to this file via environment variable
            os.environ['DM_EDITOR_TEMPLATES_PATH'] = tmp.name

            model = QStandardItemModel()
            manager = CardDataManager(model)

            # Ensure templates loaded
            self.assertIn('actions', manager.templates)

            # Lift actions for each template entry and verify conversion
            for eff in manager.templates.get('actions', []):
                manager._lift_actions_to_commands(eff)
                self.assertIn('commands', eff)
                self.assertNotIn('actions', eff)
                for c in eff['commands']:
                    self.assertIsInstance(c, dict)
                    self.assertIn('type', c)
                    self.assertIn('uid', c)

        finally:
            try:
                os.unlink(tmp.name)
            except Exception:
                pass
            if 'DM_EDITOR_TEMPLATES_PATH' in os.environ:
                del os.environ['DM_EDITOR_TEMPLATES_PATH']


if __name__ == '__main__':
    unittest.main()
