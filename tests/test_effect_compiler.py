

import dm_ai_module
from dm_ai_module import GameState, EffectSystem, ActionDef, EffectActionType, InstructionOp

def test_draw_handler_compile():
    print("Setup Compile Test")
    state = GameState(100)
    card_db = {}

    state.add_card_to_hand(0, 1, 100)

    # Action: Draw 2 cards
    action = ActionDef()
    action.type = EffectActionType.DRAW_CARD
    action.value1 = 2

    ctx = {}

    print("Compiling...")
    instructions = EffectSystem.compile_action(state, action, 100, card_db, ctx)
    print(f"Compiled {len(instructions)} instructions")

    # Verify
    assert len(instructions) == 2

    inst0 = instructions[0]
    assert inst0.op == InstructionOp.IF
    # Check nested args using helper
    cond_type = inst0.get_arg_nested_str("cond", "type")
    assert cond_type == "DECK_EMPTY"

    # Verify Then Block (Win/Lose)
    assert len(inst0.then_block) == 1
    assert inst0.then_block[0].op == InstructionOp.GAME_ACTION
    assert inst0.then_block[0].get_arg_str("type") == "LOSE_GAME"

    # Verify Else Block (Move, Stat, Trigger)
    assert len(inst0.else_block) == 3
    assert inst0.else_block[0].op == InstructionOp.MOVE
    assert inst0.else_block[1].op == InstructionOp.MODIFY
    assert inst0.else_block[2].op == InstructionOp.GAME_ACTION

    print("test_draw_handler_compile passed")

def test_draw_handler_execution():
    print("Setup Execution Test")
    state = GameState(200)
    card_db = {}

    # Setup Player 0 with deck
    state.add_card_to_deck(0, 1, 101)
    state.add_card_to_deck(0, 1, 102)
    state.add_card_to_hand(0, 1, 100) # Source

    # Action: Draw 1 card
    action = ActionDef()
    action.type = EffectActionType.DRAW_CARD
    action.value1 = 1

    ctx = {}
    print("Compiling Execution...")
    instructions = EffectSystem.compile_action(state, action, 100, card_db, ctx)

    assert len(instructions) == 1

    # Execute
    print("Executing Pipeline...")
    pipeline = dm_ai_module.PipelineExecutor()
    pipeline.execute(instructions, state, card_db)
    print("Execution Finished")

    # Verify State
    assert len(state.players[0].hand) == 2
    assert len(state.players[0].deck) == 1
    assert state.players[0].hand[1].instance_id == 102

    # Verify Stats
    assert state.turn_stats.cards_drawn_this_turn == 1

    print("test_draw_handler_execution passed")

if __name__ == "__main__":
    test_draw_handler_compile()
    test_draw_handler_execution()
