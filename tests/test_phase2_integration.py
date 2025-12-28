
import sys
import os
import unittest

# Add bin to path
sys.path.append(os.path.join(os.getcwd(), 'bin'))

try:
    import dm_ai_module
except ImportError:
    dm_ai_module = None

from dm_toolkit.commands_new import wrap_action
from dm_toolkit.engine.compat import EngineCompat

@unittest.skipIf(dm_ai_module is None, "dm_ai_module not found")
class TestPhase2Integration(unittest.TestCase):
    def setUp(self):
        self.state = dm_ai_module.GameState(40)
        self.p1_id = 0
        self.card_id = 1
        card_data = dm_ai_module.CardData(
             self.card_id, "TestCard", 1, "FIRE", 1000, "CREATURE", ["Human"], []
        )
        dm_ai_module.register_card_data(card_data)

        self.instance_id = 100
        # Add card to Battle Zone
        self.state.add_test_card_to_battle(self.p1_id, self.card_id, self.instance_id, False, True)

    def test_legacy_tap_action_flow(self):
        # 1. Create a legacy Action dict
        legacy_action = {
            "type": "TAP",
            "filter": {
                "zones": ["BATTLE_ZONE"],
                "owner": "SELF"
            },
            "scope": "SELF"
        }

        # 2. Wrap it
        cmd = wrap_action(legacy_action)
        self.assertIsNotNone(cmd, "Wrapped command should not be None")

        # 3. Execute via Compat
        # This calls EngineCompat.ExecuteCommand -> builds CommandDef -> dm_ai_module.CommandSystem.execute_command
        # We need to ensure ExecuteCommand can infer context.
        # Since 'cmd' is a wrapper around a dict, it doesn't have '_action' attribute pointing to an Action object
        # unless wrap_action was called with an Action object. Here it was called with a dict.
        # So source_id will default to -1.
        # However, for TAP "SELF" scope, CommandSystem uses players_to_check based on TargetScope.
        # TAP logic: resolve_targets -> MutateCommand.

        EngineCompat.ExecuteCommand(self.state, cmd)

        # 4. Verify State Change
        inst = self.state.get_card_instance(self.instance_id)
        self.assertTrue(inst.is_tapped, "Card should be tapped via legacy action flow")

if __name__ == '__main__':
    unittest.main()
