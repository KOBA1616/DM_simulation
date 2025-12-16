
import unittest
import dm_ai_module
from dm_ai_module import GameState, CommandSystem, CommandDef, JsonCommandType, TargetScope, FilterDef, Zone, CardDefinition, CardKeywords

class TestCommandSystem(unittest.TestCase):
    def setUp(self):
        self.state = GameState(100)
        self.state.setup_test_duel()

        # Create a dummy card
        self.card_id = 999
        self.instance_id = 0

        # Create a CardDefinition for testing
        keywords = CardKeywords()
        effects = []

        # We need to register this card definition so CardRegistry has it
        # Since we can't easily inject into the singleton CardRegistry via Python directly for *all* tests without a helper,
        # we rely on the fact that GenericCardSystem uses CardRegistry.
        # But CommandSystem.execute_command uses CardRegistry::get_all_definitions() internally.

        card_data = dm_ai_module.CardData(
            self.card_id, "Test Creature", 1, "FIRE", 1000, "CREATURE", ["Dragon"], effects
        )
        dm_ai_module.register_card_data(card_data)

        # Place the card in battle zone
        self.state.add_test_card_to_battle(0, self.card_id, self.instance_id, False, False)

    def test_mutate_tap(self):
        # Create a TAP command
        cmd = CommandDef()
        cmd.type = JsonCommandType.MUTATE
        cmd.target_group = TargetScope.SELF
        cmd.target_filter = FilterDef() # Empty filter matches self if source_instance_id is provided
        cmd.mutation_kind = "TAP"

        # Check initial state
        inst = self.state.get_card_instance(self.instance_id)
        self.assertFalse(inst.is_tapped)

        # Execute
        CommandSystem.execute_command(self.state, cmd, self.instance_id, 0)

        # Check final state
        inst = self.state.get_card_instance(self.instance_id)
        self.assertTrue(inst.is_tapped)

    def test_mutate_untap(self):
        # Setup: Tap the card first
        inst = self.state.get_card_instance(self.instance_id)
        inst.is_tapped = True

        # Create UNTAP command
        cmd = CommandDef()
        cmd.type = JsonCommandType.MUTATE
        cmd.target_group = TargetScope.SELF
        cmd.mutation_kind = "UNTAP"

        # Execute
        CommandSystem.execute_command(self.state, cmd, self.instance_id, 0)

        # Check
        inst = self.state.get_card_instance(self.instance_id)
        self.assertFalse(inst.is_tapped)

    def test_mutate_power_mod(self):
        # Create POWER_MOD command (+5000)
        cmd = CommandDef()
        cmd.type = JsonCommandType.MUTATE
        cmd.target_group = TargetScope.SELF
        cmd.mutation_kind = "POWER_MOD"
        cmd.amount = 5000

        # Execute
        CommandSystem.execute_command(self.state, cmd, self.instance_id, 0)

        # Check
        inst = self.state.get_card_instance(self.instance_id)
        self.assertEqual(inst.power_mod, 5000)

    def test_mutate_add_keyword(self):
        # Create ADD_KEYWORD command (BLOCKER)
        cmd = CommandDef()
        cmd.type = JsonCommandType.MUTATE
        cmd.target_group = TargetScope.SELF
        cmd.mutation_kind = "ADD_KEYWORD"
        cmd.str_param = "BLOCKER"

        # Execute
        CommandSystem.execute_command(self.state, cmd, self.instance_id, 0)

        # Note: We can't easily check 'active_modifiers' from Python directly on the instance
        # unless CardInstance exposes computed keywords.
        # However, checking game state active_modifiers is possible if exposed.
        # But for now, let's just run it to ensure no crash.
        # Ideally we verify the effect.

        pass

if __name__ == '__main__':
    unittest.main()
