import unittest
import dm_ai_module
from dm_ai_module import GameState, ActionType, EffectActionType, Zone, Phase, GameCommand
from dm_ai_module import ActionDef, EffectDef, CardData, Civilization, CardType, FilterDef
import pytest
import sys

# Helper to verify command migration
class TestCommandMigrationModifiers(unittest.TestCase):
    def setUp(self):
        self.state = GameState(40)
        self.card_db = {
            1: dm_ai_module.CardDefinition()
        }
        # Initialize players
        self.state.players[0].battle_zone = []
        self.state.players[1].battle_zone = []

        # Setup active player
        self.state.active_player_id = 0

        # Ensure owner map covers instance 100
        # pid=0, cid=1, iid=100, tapped=False, sick=False
        self.state.add_test_card_to_battle(0, 1, 100, False, False)

    def test_apply_modifier_uses_command(self):
        # ActionDef for APPLY_MODIFIER "COST"
        action = ActionDef()
        action.type = EffectActionType.APPLY_MODIFIER
        action.str_val = "COST"
        action.value1 = 2 # reduce by 2
        action.value2 = 1 # duration 1 turn
        action.filter = FilterDef()
        action.filter.zones = ["HAND"]
        action.filter.civilizations = [Civilization.FIRE]

        # Execute action via GenericCardSystem
        # We need to manually invoke because we are testing implementation
        # dm_ai_module.GenericCardSystem.resolve_action(self.state, action, -1, {}, self.card_db)
        # However, bindings might not expose this fully.

        # We can check command history.
        history_size_before = len(self.state.command_history)
        modifiers_size_before = len(self.state.active_modifiers)

        # Use resolve_action_with_db (args: state, action, source_id, db, ctx)
        dm_ai_module.GenericCardSystem.resolve_action_with_db(self.state, action, 100, self.card_db, {})

        # Verify result
        self.assertEqual(len(self.state.active_modifiers), modifiers_size_before + 1)
        self.assertGreater(len(self.state.command_history), history_size_before)

        # Check command type
        last_cmd = self.state.command_history[-1]
        self.assertEqual(last_cmd.get_type(), dm_ai_module.CommandType.MUTATE)
        # MutateCommand bindings might verify mutation type if exposed, but we can verify effect.

        # Test Undo (Invert)
        last_cmd.invert(self.state)
        self.assertEqual(len(self.state.active_modifiers), modifiers_size_before)

    def test_apply_modifier_passive_uses_command(self):
        # ActionDef for APPLY_MODIFIER "POWER"
        action = ActionDef()
        action.type = EffectActionType.APPLY_MODIFIER
        action.str_val = "POWER"
        action.value1 = 5000
        action.value2 = 1
        action.filter = FilterDef()
        action.filter.zones = ["BATTLE_ZONE"]

        history_size_before = len(self.state.command_history)
        passives_size_before = len(self.state.passive_effects)

        dm_ai_module.GenericCardSystem.resolve_action_with_db(self.state, action, 100, self.card_db, {})

        self.assertEqual(len(self.state.passive_effects), passives_size_before + 1)
        self.assertGreater(len(self.state.command_history), history_size_before)

        last_cmd = self.state.command_history[-1]
        self.assertEqual(last_cmd.get_type(), dm_ai_module.CommandType.MUTATE)

        last_cmd.invert(self.state)
        self.assertEqual(len(self.state.passive_effects), passives_size_before)

    def test_grant_keyword_uses_command(self):
        # ActionDef for GRANT_KEYWORD
        action = ActionDef()
        action.type = EffectActionType.GRANT_KEYWORD
        action.str_val = "BLOCKER"
        action.value2 = 1
        action.filter = FilterDef()
        action.filter.zones = ["BATTLE_ZONE"]

        history_size_before = len(self.state.command_history)
        passives_size_before = len(self.state.passive_effects)

        dm_ai_module.GenericCardSystem.resolve_action_with_db(self.state, action, 100, self.card_db, {})

        self.assertEqual(len(self.state.passive_effects), passives_size_before + 1)
        self.assertGreater(len(self.state.command_history), history_size_before)

        # Verify passive details if possible, or just count
        last_cmd = self.state.command_history[-1]
        self.assertEqual(last_cmd.get_type(), dm_ai_module.CommandType.MUTATE)

        last_cmd.invert(self.state)
        self.assertEqual(len(self.state.passive_effects), passives_size_before)

if __name__ == '__main__':
    unittest.main()
