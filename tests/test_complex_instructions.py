
import dm_ai_module
from dm_ai_module import GameState, EffectSystem, ActionDef, EffectActionType, InstructionOp

def test_complex_branching():
    print("Setup Complex Branching Test")
    state = GameState(200)
    card_db = {}

    # Populate DB so PipelineExecutor can find definitions
    for cid in [1]:
        # Minimal definition
        cdef = dm_ai_module.CardDefinition(cid, "Dummy", "FIRE", ["Dragon"], 5, 5000, dm_ai_module.CardKeywords(), [])
        card_db[cid] = cdef

    # Setup: 1 card in hand, 2 in deck.
    state.add_card_to_hand(0, 1, 100)
    state.add_card_to_deck(0, 1, 101)
    state.add_card_to_deck(0, 1, 102)

    # Test a Search Action that uses the new Pipeline logic
    # This verifies the bindings for Instruction args are working (JSON passing)

    action = ActionDef()
    action.type = EffectActionType.SEARCH_DECK_BOTTOM
    action.value1 = 1
    action.filter = dm_ai_module.FilterDef() # All

    ctx = {}
    print("Compiling Action...")
    insts = EffectSystem.compile_action(state, action, 100, card_db, ctx)

    print("Executing Pipeline...")
    pipeline = dm_ai_module.PipelineExecutor()
    pipeline.set_context_var("$source", 100)
    pipeline.execute(insts, state, card_db)

    # Verify: 1 card moved from deck bottom to hand.
    # Original hand: 1. Final hand: 2.
    assert len(state.players[0].hand) == 2
    print("Complex Branching / Pipeline Test Passed")

def test_instruction_args_binding():
    print("Testing Instruction Args Binding Stability")
    # Create an instruction with complex args via Python dict
    inst = dm_ai_module.Instruction(InstructionOp.PRINT)

    args = {
        "msg": "Hello World",
        "nested": {
            "key": "value",
            "val": 123
        },
        "list": [1, 2, 3]
    }

    try:
        inst.set_args(args)
        print("set_args successful")
    except Exception as e:
        print(f"set_args failed: {e}")
        raise e

    # Verify retrieval
    assert inst.get_arg_str("msg") == "Hello World"
    # Nested retrieval is not fully exposed via simple getters, but set_args shouldn't crash.

    print("Args Binding Test Passed")

if __name__ == "__main__":
    test_complex_branching()
    test_instruction_args_binding()
