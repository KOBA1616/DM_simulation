
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

    inst0 = instructions[0]
    assert inst0.get_then_block_size() == 1
    assert inst0.get_then_instruction(0).op == InstructionOp.GAME_ACTION
    assert inst0.get_then_instruction(0).get_arg_str("type") == "LOSE_GAME"

    assert inst0.get_else_block_size() == 3
    assert inst0.get_else_instruction(0).op == InstructionOp.MOVE
    assert inst0.get_else_instruction(1).op == InstructionOp.MODIFY
    assert inst0.get_else_instruction(2).op == InstructionOp.GAME_ACTION

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
    instructions = EffectSystem.compile_action(state, action, 100, card_db, ctx)

    pipeline = dm_ai_module.PipelineExecutor()
    pipeline.set_context_var("$source", 100) # Required for DeckEmpty condition
    pipeline.execute(instructions, state, card_db)

    # Verify State
    assert len(state.players[0].hand) == 2
    assert len(state.players[0].deck) == 1
    assert state.players[0].hand[1].instance_id == 102

    # Verify Stats
    assert state.turn_stats.cards_drawn_this_turn == 1

    print("test_draw_handler_execution passed")

def test_mana_handler_execution():
    print("Setup Mana Execution Test")
    state = GameState(200)
    card_db = {}

    state.add_card_to_deck(0, 1, 101)
    state.add_card_to_deck(0, 1, 102)
    state.add_card_to_hand(0, 1, 100)

    action = ActionDef()
    action.type = EffectActionType.ADD_MANA
    action.value1 = 1

    ctx = {}
    instructions = EffectSystem.compile_action(state, action, 100, card_db, ctx)
    assert len(instructions) == 1

    pipeline = dm_ai_module.PipelineExecutor()
    pipeline.set_context_var("$source", 100)
    pipeline.execute(instructions, state, card_db)

    assert len(state.players[0].mana_zone) == 1
    assert state.players[0].mana_zone[0].instance_id == 102
    print("test_mana_handler_execution passed")

def test_search_handler_execution():
    print("Setup Search Execution Test (Deck Bottom)")
    state = GameState(200)
    card_db = {}

    for cid in [1, 2, 3]:
        cdef = dm_ai_module.CardDefinition(cid, "Dummy", "FIRE", ["Dragon"], 5, 5000, dm_ai_module.CardKeywords(), [])
        card_db[cid] = cdef

    state.add_card_to_deck(0, 1, 101)
    state.add_card_to_deck(0, 2, 102)
    state.add_card_to_deck(0, 3, 103)

    action = ActionDef()
    action.type = EffectActionType.SEARCH_DECK_BOTTOM
    action.value1 = 2
    action.filter = dm_ai_module.FilterDef()

    ctx = {}
    instructions = EffectSystem.compile_action(state, action, 100, card_db, ctx)

    pipeline = dm_ai_module.PipelineExecutor()
    pipeline.set_context_var("$source", 100)
    pipeline.execute(instructions, state, card_db)

    assert len(state.players[0].hand) == 1
    assert state.players[0].hand[0].instance_id == 101
    assert len(state.players[0].deck) == 2
    assert state.players[0].deck[0].instance_id == 102

    print("test_search_handler_execution passed")

if __name__ == "__main__":
    test_draw_handler_compile()
    test_draw_handler_execution()
    test_mana_handler_execution()
    test_search_handler_execution()
