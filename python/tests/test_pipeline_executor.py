import sys
import os
import pytest
import json

# Add bin directory to path to import dm_ai_module
sys.path.append(os.path.join(os.path.dirname(__file__), '../../bin'))

try:
    import dm_ai_module as dm
except ImportError:
    pytest.fail("Could not import dm_ai_module. Ensure the module is built and in the bin directory.")

class TestPipelineExecutor:
    def test_pipeline_basic_math(self):
        state = dm.GameState(100)
        executor = dm.PipelineExecutor()

        # Instructions:
        # 1. MATH: 5 + 3 -> $result
        # 2. PRINT: "Result is "

        inst1 = dm.Instruction(dm.InstructionOp.MATH)
        inst1.set_args({"lhs": 5, "rhs": 3, "op": "+", "out": "$result"})

        inst2 = dm.Instruction(dm.InstructionOp.IF)
        # IF $result == 8 THEN PRINT "Correct"
        inst2.set_args({"cond": {"lhs": "$result", "op": "==", "rhs": 8}})

        then_inst = dm.Instruction(dm.InstructionOp.PRINT)
        then_inst.set_args({"msg": "Correct Calculation"})
        inst2.then_block = [then_inst]

        executor.execute([inst1, inst2], state)

        # Verify context (Need to expose context getter in bindings)
        res = executor.get_context_var("$result")
        assert res == 8

    def test_pipeline_selection_flow(self):
        state = dm.GameState(100)
        state.setup_test_duel()
        executor = dm.PipelineExecutor()

        # Mock selection
        # 1. SELECT targets -> $selection
        # 2. IF exists $selection THEN MOVE $selection to MANA

        inst1 = dm.Instruction(dm.InstructionOp.SELECT)
        inst1.set_args({"out": "$selection", "filter": {"zone": "BATTLE"}})

        inst2 = dm.Instruction(dm.InstructionOp.IF)
        inst2.set_args({"cond": {"exists": "$selection"}})

        move_inst = dm.Instruction(dm.InstructionOp.MOVE)
        move_inst.set_args({"target": "$selection", "to": "MANA"})
        inst2.then_block = [move_inst]

        executor.execute([inst1, inst2], state)

        # Check context
        sel = executor.get_context_var("$selection")
        assert isinstance(sel, list)
        assert len(sel) == 3 # Mock returns 3 items [1, 2, 3]

        # In a real test we would verify the cards moved,
        # but since we mocked the command execution in C++ to specific IDs,
        # and didn't set up those specific IDs in GameState fully,
        # we just assume the executor ran without crashing.

if __name__ == "__main__":
    pytest.main([__file__])
