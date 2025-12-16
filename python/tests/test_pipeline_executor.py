import pytest
import dm_ai_module
import json

def test_pipeline_basics():
    state = dm_ai_module.GameState(100)
    executor = dm_ai_module.PipelineExecutor()
    executor.execute([], state, {})

    inst = dm_ai_module.Instruction(dm_ai_module.InstructionOp.MATH)
    inst.set_args({"op": "+", "lhs": 5, "rhs": 3, "out": "res"})
    executor.execute([inst], state, {})

    res = executor.get_context_var("res")
    assert res == 8

def test_legacy_adapter_conversion():
    f = dm_ai_module.FilterDef()
    a = dm_ai_module.ActionDef(dm_ai_module.EffectActionType.DRAW_CARD, dm_ai_module.TargetScope.NONE, f)
    a.value1 = 2

    c = dm_ai_module.ConditionDef()
    c.type = "NONE"

    e = dm_ai_module.EffectDef(dm_ai_module.TriggerType.NONE, c, [a])

    instructions = dm_ai_module.LegacyJsonAdapter.convert(e)

    assert len(instructions) == 1
    assert instructions[0].op == dm_ai_module.InstructionOp.MOVE

    state = dm_ai_module.GameState(100)
    state.add_card_to_deck(0, 1, 100)
    state.add_card_to_deck(0, 1, 101)
    state.add_card_to_deck(0, 1, 102)

    executor = dm_ai_module.PipelineExecutor()
    executor.execute(instructions, state, {})

    assert len(state.players[0].hand) == 2
    assert len(state.players[0].deck) == 1

def test_pipeline_select_and_destroy():
    # Scenario: Select 1 creature in Battle Zone and Destroy it
    f = dm_ai_module.FilterDef()
    f.zones = ["BATTLE_ZONE"]

    a = dm_ai_module.ActionDef(dm_ai_module.EffectActionType.DESTROY, dm_ai_module.TargetScope.TARGET_SELECT, f)
    a.value1 = 1

    c = dm_ai_module.ConditionDef()
    c.type = "NONE"

    e = dm_ai_module.EffectDef(dm_ai_module.TriggerType.NONE, c, [a])

    instructions = dm_ai_module.LegacyJsonAdapter.convert(e)

    assert len(instructions) == 2
    assert instructions[0].op == dm_ai_module.InstructionOp.SELECT
    assert instructions[1].op == dm_ai_module.InstructionOp.MOVE

    # Execution
    state = dm_ai_module.GameState(100)
    # Add a creature to Battle Zone
    state.add_test_card_to_battle(0, 1, 200, False, False)
    state.active_player_id = 0

    # Mock DB
    card_data = dm_ai_module.CardData(1, "TestCreature", 1, "FIRE", 1000, "CREATURE", [], [])
    dm_ai_module.register_card_data(card_data)

    k = dm_ai_module.CardKeywords()
    d = dm_ai_module.CardDefinition(1, "TestCreature", "FIRE", [], 1, 1000, k, [])
    card_db = {1: d}

    executor = dm_ai_module.PipelineExecutor()
    executor.execute(instructions, state, card_db)

    # Verify: Card moved to Graveyard
    assert len(state.players[0].battle_zone) == 0
    assert len(state.players[0].graveyard) == 1
    assert state.players[0].graveyard[0].instance_id == 200

def test_pipeline_search_deck():
    f = dm_ai_module.FilterDef()
    f.zones = ["DECK"]

    a = dm_ai_module.ActionDef(dm_ai_module.EffectActionType.SEARCH_DECK, dm_ai_module.TargetScope.NONE, f)
    a.value1 = 1

    # Ensure explicit NONE condition to avoid auto-IF
    c = dm_ai_module.ConditionDef()
    c.type = "NONE"

    e = dm_ai_module.EffectDef(dm_ai_module.TriggerType.NONE, c, [a])

    instructions = dm_ai_module.LegacyJsonAdapter.convert(e)

    assert len(instructions) == 2
    assert instructions[0].op == dm_ai_module.InstructionOp.SELECT
    assert instructions[1].op == dm_ai_module.InstructionOp.MOVE

    state = dm_ai_module.GameState(100)
    state.add_card_to_deck(0, 1, 100)
    state.active_player_id = 0

    card_data = dm_ai_module.CardData(1, "TestCard", 1, "FIRE", 1000, "CREATURE", [], [])
    dm_ai_module.register_card_data(card_data)

    k = dm_ai_module.CardKeywords()
    d = dm_ai_module.CardDefinition(1, "TestCard", "FIRE", [], 1, 1000, k, [])

    card_db = {1: d}

    executor = dm_ai_module.PipelineExecutor()
    executor.execute(instructions, state, card_db)

    assert len(state.players[0].hand) == 1
    assert len(state.players[0].deck) == 0
