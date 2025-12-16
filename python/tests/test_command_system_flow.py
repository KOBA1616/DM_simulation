
import unittest
import dm_ai_module
from dm_ai_module import GameState, CommandSystem, CommandDef, JsonCommandType, TargetScope, FilterDef, Zone, CardDefinition, CardKeywords, ConditionDef

class TestCommandSystemFlow(unittest.TestCase):
    def setUp(self):
        self.state = GameState(100)
        self.state.setup_test_duel()
        self.card_id = 999
        self.instance_id = 0

        # Setup dummy card data
        effects = []
        card_data = dm_ai_module.CardData(
            self.card_id, "Test Creature", 1, "FIRE", 1000, "CREATURE", ["Dragon"], effects
        )
        dm_ai_module.register_card_data(card_data)
        self.state.add_test_card_to_battle(0, self.card_id, self.instance_id, False, False)

    def test_flow_condition_met(self):
        # Condition: DURING_YOUR_TURN (Should be true initially as active_player is 0)
        cond = ConditionDef()
        cond.type = "DURING_YOUR_TURN"

        # True Branch: Power +5000
        cmd_true = CommandDef()
        cmd_true.type = JsonCommandType.MUTATE
        cmd_true.target_group = TargetScope.SELF
        cmd_true.mutation_kind = "POWER_MOD"
        cmd_true.amount = 5000

        # False Branch: Untap (Should not run)
        cmd_false = CommandDef()
        cmd_false.type = JsonCommandType.MUTATE
        cmd_false.target_group = TargetScope.SELF
        cmd_false.mutation_kind = "UNTAP"

        # Flow Command
        flow_cmd = CommandDef()
        flow_cmd.type = JsonCommandType.FLOW
        flow_cmd.condition = cond
        flow_cmd.if_true = [cmd_true]
        flow_cmd.if_false = [cmd_false]

        # Execute
        CommandSystem.execute_command(self.state, flow_cmd, self.instance_id, 0)

        # Verify: Power mod should be 5000
        inst = self.state.get_card_instance(self.instance_id)
        self.assertEqual(inst.power_mod, 5000)

    def test_flow_condition_not_met(self):
        # Condition: DURING_OPPONENT_TURN (Should be false)
        cond = ConditionDef()
        cond.type = "DURING_OPPONENT_TURN"

        # True Branch: Power +5000
        cmd_true = CommandDef()
        cmd_true.type = JsonCommandType.MUTATE
        cmd_true.target_group = TargetScope.SELF
        cmd_true.mutation_kind = "POWER_MOD"
        cmd_true.amount = 5000

        # False Branch: Power +1000
        cmd_false = CommandDef()
        cmd_false.type = JsonCommandType.MUTATE
        cmd_false.target_group = TargetScope.SELF
        cmd_false.mutation_kind = "POWER_MOD"
        cmd_false.amount = 1000

        # Flow Command
        flow_cmd = CommandDef()
        flow_cmd.type = JsonCommandType.FLOW
        flow_cmd.condition = cond
        flow_cmd.if_true = [cmd_true]
        flow_cmd.if_false = [cmd_false]

        # Execute
        CommandSystem.execute_command(self.state, flow_cmd, self.instance_id, 0)

        # Verify: Power mod should be 1000
        inst = self.state.get_card_instance(self.instance_id)
        self.assertEqual(inst.power_mod, 1000)

if __name__ == '__main__':
    unittest.main()
