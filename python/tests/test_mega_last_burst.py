import pytest
import dm_ai_module

def test_mega_last_burst_logic():
    # Setup GameState
    state = dm_ai_module.GameState(100)

    # Create Twinpact Spell Side first
    spell_id = 1 # Same ID usually
    spell_side = dm_ai_module.CardData(
        spell_id,
        "Twinpact Spell",
        3,
        "FIRE",
        0,
        "SPELL",
        [],
        []
    )

    # Add an effect to spell side
    eff = dm_ai_module.EffectDef()
    eff.trigger = dm_ai_module.TriggerType.NONE
    act = dm_ai_module.ActionDef()
    act.type = dm_ai_module.EffectActionType.ADD_MANA
    act.value1 = 1
    act.source_zone = "DECK"
    eff.actions.append(act)
    spell_side.effects = [eff]

    # Create Twinpact Creature Side
    card_id = 1
    cdata = dm_ai_module.CardData(
        card_id,
        "Twinpact Test",
        5,
        "FIRE",
        5000,
        "CREATURE",
        [],
        []
    )

    # Set properties
    cdata.keywords = {"mega_last_burst": True}
    cdata.spell_side = spell_side

    dm_ai_module.register_card_data(cdata)

    # Setup board: Player 0 has the creature in Battle Zone
    state.add_test_card_to_battle(0, card_id, 0, False, False)
    # Ensure mana/deck has cards for spell effect (mana 0 initially)
    # add_card_to_deck(pid, card_id, instance_id)
    state.add_card_to_deck(0, 2, 1)
    state.add_card_to_deck(0, 2, 2)

    # Trigger Destruction via EffectSystem.resolve_action(DESTROY)
    act_destroy = dm_ai_module.ActionDef()
    act_destroy.type = dm_ai_module.EffectActionType.DESTROY
    act_destroy.scope = dm_ai_module.TargetScope.NONE
    act_destroy.filter.zones = ["BATTLE_ZONE"]
    act_destroy.filter.owner = "SELF"
    act_destroy.filter.count = 1

    # Execute Destroy
    dm_ai_module.EffectSystem.resolve_action(state, act_destroy, 0)

    # Check if pending effect for Mega Last Burst is queued
    pending = dm_ai_module.get_pending_effects_info(state)

    found_mlb = False
    mlb_idx = -1
    for i, p in enumerate(pending):
        # EffectType.TRIGGER_ABILITY = 15
        if p[0] == dm_ai_module.EffectType.TRIGGER_ABILITY and p[1] == 0:
             found_mlb = True
             mlb_idx = i
             break

    assert found_mlb, "Mega Last Burst pending effect not found"
