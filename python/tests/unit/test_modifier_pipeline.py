
import unittest
import sys
import os

# Add the 'bin' directory to sys.path
sys.path.append(os.path.abspath('bin'))
sys.path.append(os.path.abspath('build'))

try:
    import dm_ai_module
except ImportError:
    print("Error: dm_ai_module not found. Please build the project first.")
    sys.exit(1)

class TestModifierPipeline(unittest.TestCase):
    def setUp(self):
        self.state = dm_ai_module.GameState(100)
        self.card_db = dm_ai_module.CardDatabase()
        # Setup dummy cards
        c1 = dm_ai_module.CardDefinition()
        c1.name = "TestCreature"
        c1.type = dm_ai_module.CardType.CREATURE
        c1.power = 3000
        self.card_db[1] = c1

        c2 = dm_ai_module.CardDefinition()
        c2.name = "TestModifierSource"
        c2.type = dm_ai_module.CardType.CREATURE
        self.card_db[2] = c2

    def test_compile_action_and_execute(self):
        # 1. Setup GameState
        state = self.state
        card_db = self.card_db

        # Add a creature to battle zone
        c_inst = dm_ai_module.CardInstance()
        c_inst.card_id = 1
        c_inst.instance_id = 0
        c_inst.owner = 0
        state.add_test_card_to_battle(0, 1, 0, False, False)

        # 2. Create an Action that maps to ModifierHandler
        action = dm_ai_module.ActionDef()
        action.type = dm_ai_module.EffectActionType.APPLY_MODIFIER
        action.str_val = "POWER"
        action.value1 = 5000
        action.value2 = 1

        # 3. Compile Action using EffectSystem (which calls ModifierHandler::compile_action)
        # We need a context for execution variables?
        # compile_action signature: (state, action, source_id, db, py_ctx)

        # EffectSystem.compile_action is static method bound to return List[Instruction]
        source_id = 1
        state.add_test_card_to_battle(0, 2, 1, False, False)

        # Note: python/tests/conftest.py may wrap GameState with a proxy. Native bindings require the native object.
        native_state = getattr(state, "_native", state)
        instructions = dm_ai_module.EffectSystem.compile_action(native_state, action, source_id, card_db, {})

        self.assertTrue(len(instructions) > 0)
        inst = instructions[0]
        self.assertEqual(inst.op, dm_ai_module.InstructionOp.MODIFY)
        self.assertEqual(inst.get_arg_str("str_value"), "POWER")
        self.assertEqual(inst.get_arg_int("value"), 5000)
        self.assertEqual(inst.get_arg_str("type"), "ADD_PASSIVE")

        # 4. Execute instructions via PipelineExecutor
        pipeline = dm_ai_module.PipelineExecutor()
        pipeline.execute(instructions, native_state, card_db)

        # 5. Verify effect applied (Global Filter based since no targets passed)
        passives = state.passive_effects
        found = False
        for p in passives:
            if p.type == dm_ai_module.PassiveType.POWER_MODIFIER and p.value == 5000:
                found = True
                break
        self.assertTrue(found, "Pipeline execution should have added POWER +5000 modifier")

if __name__ == '__main__':
    unittest.main()
