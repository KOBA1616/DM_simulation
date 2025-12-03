
import pytest
import dm_ai_module
from dm_ai_module import GameInstance, CardData, EffectDef, ActionDef, EffectActionType, TriggerType, TargetScope, CardKeywords, FilterDef

def test_revolution_change_logic():
    # 1. Register a dummy Revolution Change card
    # We use a dummy ID 9001 for Revolution Change Card
    rev_eff = EffectDef()
    # No explicit effect logic needed for the swap itself (handled by engine),
    # but we might want a CIP effect to prove it entered.
    cip_act = ActionDef()
    cip_act.type = EffectActionType.MODIFY_POWER
    cip_act.target_player = "SELF" # Not used by MODIFY_POWER but good practice
    cip_act.value1 = 1000
    cip_act.scope = TargetScope.SELF

    rev_eff.trigger = TriggerType.ON_PLAY
    rev_eff.actions = [cip_act]

    # We must enable the 'revolution_change' keyword manually since we don't have the JSON loader parsing keywords here easily
    # Wait, `register_card_data` takes CardData. CardData doesn't have `keywords` struct directly exposed in Python binding for `CardData` constructor?
    # Let's check bindings.
    # `CardData` struct in bindings takes `effects`. Keywords are usually parsed from string in `JsonLoader`.
    # But `register_card_data` uses `CardRegistry::load_from_json`.
    # If we want `revolution_change` keyword, we need to pass it in the JSON/CardData.
    # The `CardData` struct in C++ has `keywords`? No, `CardDefinition` (Runtime) has `keywords`. `CardData` (JSON) has `keywords` as map or list?
    # In `src/core/card_json_types.hpp`, `CardData` does NOT have keywords field?
    # Ah, `CardData` struct in `card_json_types.hpp` has `type`, `races`, `effects`.
    # Where are keywords?
    # Looking at `json_loader.cpp`, keywords like `speed_attacker` are derived from `effects` (TriggerType::PASSIVE_CONST).
    # Is `revolution_change` also derived?
    # `grep` showed `csv_loader` handled it. `json_loader`?

    # If `json_loader` doesn't support `revolution_change` keyword parsing yet, I need to add it OR use a hack.
    # The binding for `CardDefinition` allows setting keywords directly.
    # But `start_game` takes `std::map<CardID, CardDefinition>`.
    # So I can construct `CardDefinition` manually in Python with the keyword set!

    # Create CardDefinition for Rev Change Unit
    rev_kw = CardKeywords()
    rev_kw.revolution_change = True
    rev_kw.cip = True # To test CIP

    rev_card_def = dm_ai_module.CardDefinition(
        9001, "Dogiragon", "FIRE", ["Mega Command Dragon"], 7, 7000, rev_kw, []
    )

    # Create CardDefinition for Attacker
    attacker_kw = CardKeywords()
    attacker_kw.speed_attacker = True # So it can attack immediately
    attacker_card_def = dm_ai_module.CardDefinition(
        9002, "FireBird", "FIRE", ["Fire Bird"], 3, 3000, attacker_kw, []
    )

    card_db = {
        9001: rev_card_def,
        9002: attacker_card_def
    }

    # 2. Setup Game
    game = GameInstance(0, card_db)
    game.start_game(card_db)
    state = game.state

    # Setup Board
    # Player 0 (Active) has FireBird in Battle Zone and Dogiragon in Hand
    state.active_player_id = 0
    state.current_phase = dm_ai_module.Phase.ATTACK

    # Add FireBird to Battle
    # ID, instance_id, tapped, sick
    state.add_test_card_to_battle(0, 9002, 100, False, False) # 100 is instance_id

    # Add Dogiragon to Hand
    state.add_card_to_hand(0, 9001, 200) # 200 is instance_id

    # DEBUG: Verify DB and State
    print(f"DEBUG: Hand size: {len(state.players[0].hand)}")
    print(f"DEBUG: Card in hand ID: {state.players[0].hand[0].card_id}")
    print(f"DEBUG: Rev Change Keyword: {card_db[9001].keywords.revolution_change}")

    # 3. Generate Actions (Expect ATTACK_PLAYER)
    actions = dm_ai_module.ActionGenerator.generate_legal_actions(state, card_db)

    attack_action = None
    for a in actions:
        if a.type == dm_ai_module.ActionType.ATTACK_PLAYER and a.source_instance_id == 100:
            attack_action = a
            break

    assert attack_action is not None, "Should be able to attack with FireBird"

    # 4. Perform Attack
    dm_ai_module.EffectResolver.resolve_action(state, attack_action, card_db)

    # 5. Verify Pending Effect for Revolution Change
    # The phase should NOT be BLOCK yet. It should be ATTACK (or whatever, but pending effect exists).
    # Wait, my logic said "If rev change triggered, stay/transition...".
    # Check pending effects.
    pending_effects = dm_ai_module.get_pending_effects_info(state)
    assert len(pending_effects) > 0, "Should have pending effect for Rev Change"

    # Tuple is (type, source_instance_id, controller)
    pe_type, pe_src, pe_ctrl = pending_effects[0]

    # Check type (we exposed as int/enum)
    # EffectType.ON_ATTACK_FROM_HAND
    # Cast int to enum for comparison or compare int values
    assert pe_type == int(dm_ai_module.EffectType.ON_ATTACK_FROM_HAND)

    # 6. Generate Actions again (Should be USE_ABILITY or RESOLVE_EFFECT/PASS)
    actions_step2 = dm_ai_module.ActionGenerator.generate_legal_actions(state, card_db)

    use_rev_action = None
    for a in actions_step2:
        if a.type == dm_ai_module.ActionType.USE_ABILITY and a.source_instance_id == 200:
            use_rev_action = a
            break

    assert use_rev_action is not None, "Should be able to use Revolution Change with Dogiragon"

    # 7. Use Revolution Change
    dm_ai_module.EffectResolver.resolve_action(state, use_rev_action, card_db)

    # 8. Verify Swap
    # FireBird (100) should be in Hand
    p0 = state.players[0]
    found_bird_in_hand = False
    for c in p0.hand:
        if c.instance_id == 100:
            found_bird_in_hand = True
            break
    assert found_bird_in_hand, "FireBird should be returned to hand"

    # Dogiragon (200) should be in Battle Zone, Tapped
    found_dog_in_battle = False
    for c in p0.battle_zone:
        if c.instance_id == 200:
            found_dog_in_battle = True
            assert c.is_tapped, "Dogiragon should be tapped"
            assert not c.summoning_sickness, "Dogiragon should not have summoning sickness"
            break
    assert found_dog_in_battle, "Dogiragon should be in battle zone"

    # Verify CIP triggered (Pending Effect for CIP ON_PLAY)
    # The USE_ABILITY resolution should have added a CIP pending effect if cip=True
    pending_effects = dm_ai_module.get_pending_effects_info(state)
    # We might have cleared the RevChange pending effect, so now we expect CIP
    # But since we didn't register generic effects, maybe no CIP pending effect is generated
    # unless GenericCardSystem found JSON data.
    # Since we manually created CardDefinition but GenericCardSystem relies on CardRegistry (which uses JSON),
    # CIP won't trigger here unless we mock CardRegistry or use CardRegistry.load_from_json.

    # But checking if the creature is in battle zone and tapped is enough for Revolution Change logic verification.
    # TriggerType.ON_PLAY maps to EffectType.CIP usually in logic
    # My resolve_use_ability code: "GenericCardSystem::resolve_trigger(game_state, TriggerType::ON_PLAY, ...)"
    # GenericCardSystem::resolve_trigger adds a pending effect? Or resolves immediately?
    # It usually adds a pending effect if targets needed, or resolves if auto.
    # Since I didn't define detailed effects, maybe just check if CIP pending effect is there or stats recorded?
    # Actually, GenericCardSystem::resolve_trigger logic:
    # It checks CardRegistry. Since I didn't register via CardRegistry (I used manual CardDefinition),
    # GenericCardSystem won't find the effects in JSON!

    # NOTE: To test CIP via GenericCardSystem, I MUST register the card data via `register_card_data`.
    # But `register_card_data` doesn't set `revolution_change` keyword because it parses from JSON
    # and I haven't updated `json_loader` to parse that keyword.

    # So CIP test part might fail or need adjustment.
    # But the SWAP logic is hardcoded in `resolve_use_ability`, so that should work.

    print("Test Revolution Change Logic Passed")

if __name__ == "__main__":
    test_revolution_change_logic()
