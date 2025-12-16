import pytest
import dm_ai_module
from dm_ai_module import GameState, CardDefinition, CardType, Civilization, EffectDef, ActionDef, EffectActionType, TargetScope, FilterDef, TriggerType, PhaseManager, GenericCardSystem

def test_cip_pipeline_integration():
    """
    Verify that standard CIP (On Play) effects are processed via the Instruction Pipeline.
    We test this by defining a card with a simple Draw effect and verifying it works.
    We also test a conditional effect.
    """
    # 1. Setup
    state = GameState(10)
    card_db = {}

    # 2. Define Card with CIP Draw
    # "Bronze-Arm Tribe": On Play, put top card of deck to mana.
    bat_def = CardDefinition()
    bat_def.name = "Bronze-Arm Tribe"
    bat_def.cost = 3
    bat_def.power = 1000
    bat_def.civilizations = [Civilization.NATURE]
    bat_def.type = CardType.CREATURE
    bat_def.races = ["Beast Folk"]

    eff = EffectDef()
    eff.trigger = TriggerType.ON_PLAY

    act = ActionDef()
    act.type = EffectActionType.ADD_MANA
    act.value1 = 1 # Count

    eff.actions = [act]
    bat_def.effects = [eff]

    card_db[1] = bat_def

    # Register DB
    dm_ai_module.initialize_card_stats(state, card_db, 40)

    # 3. Setup Board
    # Add card to hand to play it (conceptually, but here we invoke resolve directly)
    state.add_card_to_deck(0, 999, 100) # Dummy card in deck to charge

    instance_id = 0
    state.add_test_card_to_battle(0, 1, instance_id, False, True) # Put BAT in Battle Zone

    # 4. Trigger Resolution (Simulate Engine Flow)
    # Normally EffectResolver -> GenericCardSystem.resolve_trigger -> PendingEffect -> GenericCardSystem.resolve_effect

    # Check if Mana Zone is empty
    assert len(state.players[0].mana_zone) == 0

    # Call resolve_effect_with_db directly (simulating PendingEffect execution)
    # This should now route through PipelineExecutor because TriggerType is ON_PLAY
    GenericCardSystem.resolve_effect_with_db(state, eff, instance_id, card_db)

    # 5. Verify Result
    # Should have moved 1 card from Deck to Mana
    assert len(state.players[0].mana_zone) == 1
    assert state.players[0].mana_zone[0].card_id == 999

def test_pipeline_conditional_cip():
    """
    Test a conditional CIP effect (e.g., Mana Armed) using the pipeline.
    """
    state = GameState(10)
    card_db = {}

    # Define "Mana Armed 3": If 3 Nature mana, Draw 1.
    def_ma = CardDefinition()
    def_ma.civilizations = [Civilization.NATURE]
    def_ma.type = CardType.CREATURE

    eff = EffectDef()
    eff.trigger = TriggerType.ON_PLAY

    # Condition: MANA_ARMED 3 (Nature implied by implementation/str_val usually)
    # Using `dm_ai_module.ConditionDef`
    cond = dm_ai_module.ConditionDef()
    cond.type = "MANA_ARMED"
    cond.value = 3
    cond.str_val = "NATURE"
    eff.condition = cond

    act = ActionDef()
    act.type = EffectActionType.DRAW_CARD
    act.value1 = 1
    eff.actions = [act]

    def_ma.effects = [eff]
    card_db[2] = def_ma

    # Setup
    instance_id = 0
    state.add_test_card_to_battle(0, 2, instance_id, False, True)

    # Case A: Condition Not Met (0 Mana)
    GenericCardSystem.resolve_effect_with_db(state, eff, instance_id, card_db)
    assert len(state.players[0].hand) == 0

    # Case B: Condition Met (3 Nature Mana)
    for i in range(3):
        state.add_card_to_mana(0, 2, 10+i) # Add Nature cards (ID 2 is Nature)

    state.add_card_to_deck(0, 999, 100) # Card to draw

    GenericCardSystem.resolve_effect_with_db(state, eff, instance_id, card_db)
    assert len(state.players[0].hand) == 1

if __name__ == "__main__":
    test_cip_pipeline_integration()
    test_pipeline_conditional_cip()
    print("All CIP Pipeline tests passed.")
