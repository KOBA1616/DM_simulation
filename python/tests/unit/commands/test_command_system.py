
import unittest
import dm_ai_module
from typing import Any
from dm_ai_module import GameState, GameCommand, MutateCommand, FlowCommand, CommandType, TargetScope, FilterDef, Zone, CardDefinition, CardKeywords, ConditionDef, MutationType, Civilization, CardType

class TestCommandSystem(unittest.TestCase):
    def setUp(self):
        self.state = GameState(100)
        self.state.setup_test_duel()

        # Create a dummy card
        self.card_id = 999
        self.instance_id = 0

        # Create a CardDefinition for testing
        keywords = CardKeywords()
        effects: list[Any] = []

        card_data = dm_ai_module.CardData(
            self.card_id, "Test Creature", 1, Civilization.FIRE, 1000, CardType.CREATURE, ["Dragon"], effects
        )
        dm_ai_module.register_card_data(card_data)

        # Place the card in battle zone
        self.state.add_test_card_to_battle(0, self.card_id, self.instance_id, False, False)

    def test_mutate_tap(self):
        # Create a TAP command
        cmd = MutateCommand(self.instance_id, MutationType.TAP)

        # Check initial state
        inst = self.state.get_card_instance(self.instance_id)
        self.assertFalse(inst.is_tapped)

        # Execute
        self.state.execute_command(cmd)

        # Check final state
        inst = self.state.get_card_instance(self.instance_id)
        self.assertTrue(inst.is_tapped)

    def test_mutate_untap(self):
        # Setup: Tap the card first
        inst = self.state.get_card_instance(self.instance_id)
        self.state.execute_command(MutateCommand(self.instance_id, MutationType.TAP))
        inst = self.state.get_card_instance(self.instance_id)
        self.assertTrue(inst.is_tapped)

        # Create UNTAP command
        cmd = MutateCommand(self.instance_id, MutationType.UNTAP)

        # Execute
        self.state.execute_command(cmd)

        # Check
        inst = self.state.get_card_instance(self.instance_id)
        self.assertFalse(inst.is_tapped)

    def test_mutate_power_mod(self):
        # Create POWER_MOD command (+5000)
        cmd = MutateCommand(self.instance_id, MutationType.POWER_MOD, 5000)

        # Execute
        self.state.execute_command(cmd)

        # Pass implies no crash

    def test_flow_condition_true(self):
        pass

if __name__ == '__main__':
    unittest.main()
